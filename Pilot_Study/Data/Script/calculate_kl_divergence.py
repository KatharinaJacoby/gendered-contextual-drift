import pandas as pd
import numpy as np
import json
import os
from collections import Counter
from math import log
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_NAME = "meta-llama/Llama-3.1-8B"
EPSILON = 1e-10

FILE_LIST = [
    "run_sequence_female.csv",
    "run_sequence_male.csv",
    "male_pruned_8k.csv",
    "female_pruned_8k.csv"
]

OUTPUT_FILE = "kl_divergence_matrix.json"


def get_token_distribution(text, tokenizer):
    if not text or text.strip() == "":
        return {}
    encoded = tokenizer.encode(text, truncation=True, max_length=2048)
    if not encoded:
        return {}
    counts = Counter(encoded)
    total = sum(counts.values())
    return {tid: count / total for tid, count in counts.items()}


def kl_divergence(p, q):
    kl = 0.0
    for token, p_val in p.items():
        q_val = q.get(token, EPSILON)
        if p_val > 0:
            kl += p_val * log(p_val / q_val)
    return kl


def symmetric_kl(p, q):
    if not p or not q:
        return 0.0
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2


def load_dataframe_smart(file_path):
    """
    Attempts to load CSV with automatic delimiter detection.
    """
    if not os.path.exists(file_path):
        return None

    # 1. Detect delimiter
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    sep = ';' if ';' in first_line else ','

    try:
        df = pd.read_csv(file_path, sep=sep, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"  ❌ Error reading {file_path} (sep='{sep}'): {e}")
        return None

    # 2. Normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 3. Check required columns
    if 'turn' not in df.columns:
        print(f"  ❌ Error: No 'turn' column in {file_path}")
        return None

    resp_col = 'full_response' if 'full_response' in df.columns else ('response' if 'response' in df.columns else None)
    if not resp_col:
        print(f"  ❌ Error: No 'full_response' or 'response' in {file_path}")
        return None

    # 4. Determine variant column
    if 'variant' in df.columns:
        variant_col = 'variant'
    elif 'model' in df.columns:
        variant_col = 'model'
    else:
        # If no variant/model, assume the whole file is one run
        variant_col = None

    return df, variant_col, resp_col


def main():
    # 1. Load Tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load tokenizer. {e}")
        return

    all_results = []

    # 2. Process Each File
    for file_path in FILE_LIST:
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: File not found: {file_path}")
            continue

        print(f"\nProcessing: {file_path}")

        load_result = load_dataframe_smart(file_path)
        if load_result is None:
            continue

        df, variant_col, resp_col = load_result

        # Get unique variants
        variants = df[variant_col].unique()

        for variant in variants:
            subset = df[df[variant_col] == variant].sort_values(by='turn')

            # Parse turns, skipping NaN turns and NaN responses
            turns_map = {}
            for _, row in subset.iterrows():
                turn_str = str(row['turn']).strip()
                if turn_str == 'nan' or turn_str == '':
                    continue
                if turn_str.startswith('T'):
                    turn_num = int(turn_str[1:])
                else:
                    turn_num = int(turn_str)

                response = str(row[resp_col])
                if response == 'nan' or response.strip() == '':
                    continue

                turns_map[turn_num] = response

            if len(turns_map) < 2:
                print(f"  ⚠️  Skipping {variant}: Not enough valid turns.")
                continue

            # --- BASELINE = T01 ONLY ---
            sorted_turns = sorted(turns_map.keys())
            t1_text = turns_map[sorted_turns[0]]
            baseline_dist = get_token_distribution(t1_text, tokenizer)

            # Sanity Check: T1 vs T2
            if len(sorted_turns) > 1:
                dist_t2 = get_token_distribution(turns_map[sorted_turns[1]], tokenizer)
                stability = symmetric_kl(baseline_dist, dist_t2)
                print(f"  ✅ {variant}: Baseline Stability (T1 vs T2) = {stability:.4f}")
            else:
                print(f"  ⚠️  {variant}: Only 1 turn, using T01 as baseline.")

            # --- CALCULATE KL FOR ALL TURNS ---
            for turn_num in sorted_turns:
                if turn_num == sorted_turns[0]:
                    all_results.append({
                        "variant": variant,
                        "turn": turn_num,
                        "kl_score": 0.0,
                        "is_baseline": True
                    })
                    continue

                turn_text = turns_map[turn_num]
                turn_dist = get_token_distribution(turn_text, tokenizer)

                if not turn_dist:
                    kl_score = 0.0
                else:
                    kl_score = symmetric_kl(turn_dist, baseline_dist)

                all_results.append({
                    "variant": variant,
                    "turn": turn_num,
                    "kl_score": kl_score,
                    "is_baseline": False
                })

    # 3. Save Results
    if not all_results:
        print("\n❌ No results to save.")
        return

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n🎉 Success! Saved {len(all_results)} turn records to '{OUTPUT_FILE}'")

    # 4. Summary
    print("\n" + "=" * 75)
    print("SUMMARY: Peak KL Divergence per Variant (vs. T01 Baseline)")
    print("=" * 75)
    temp_df = pd.DataFrame(all_results)
    for variant in temp_df['variant'].unique():
        subset = temp_df[temp_df['variant'] == variant]
        peak_row = subset.loc[subset['kl_score'].idxmax()]
        contaminated = subset[subset['is_baseline'] == False]
        if len(contaminated) > 0:
            mean_kl = contaminated['kl_score'].mean()
            final_kl = contaminated['kl_score'].iloc[-1]
            recovery = final_kl / peak_row['kl_score'] if peak_row['kl_score'] > 0 else 0.0
        else:
            mean_kl = 0.0
            final_kl = 0.0
            recovery = 0.0
        print(f"{variant:<35} | Peak: {peak_row['kl_score']:.2f} @ T{peak_row['turn']} | Mean: {mean_kl:.2f} | Recovery: {recovery:.2f}")


if __name__ == "__main__":
    main()
