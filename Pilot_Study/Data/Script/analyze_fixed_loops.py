import pandas as pd
import os

def analyze_loops(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return []

    # Detect columns
    group_col = 'variant' if 'variant' in df.columns else ('model' if 'model' in df.columns else None)
    turn_col = 'turn' if 'turn' in df.columns else None
    resp_col = 'full_response' if 'full_response' in df.columns else None

    if not all([group_col, turn_col, resp_col]):
        print(f"⚠️ Missing columns in {csv_file}")
        return []

    results = []

    for variant, group in df.groupby(group_col):
        group = group.sort_values(by=turn_col)

        loop_start = None
        loop_length = 0
        loop_content_len = 0
        loop_preview = ""

        for i in range(1, len(group)):
            curr_turn = group.iloc[i][turn_col]
            prev_turn = group.iloc[i-1][turn_col]

            prev_text = str(group.iloc[i-1][resp_col]).strip()
            curr_text = str(group.iloc[i][resp_col]).strip()

            if not prev_text or not curr_text:
                continue

            if prev_text == curr_text:
                if loop_start is None:
                    loop_start = prev_turn
                loop_length += 1
                loop_content_len = len(curr_text)
                loop_preview = curr_text[:80] + "..." if len(curr_text) > 80 else curr_text
            else:
                # Loop just ended
                if loop_length > 0:
                    results.append({
                        'source_file': os.path.basename(csv_file),
                        'variant': variant,
                        'loop_start': loop_start,
                        'loop_end': group.iloc[i-1][turn_col],
                        'loop_duration_turns': loop_length + 1,
                        'loop_content_length_chars': loop_content_len,
                        'loop_content_preview': loop_preview,
                        'is_exact_match': True
                    })
                loop_start = None
                loop_length = 0
                loop_content_len = 0
                loop_preview = ""

        # Check if loop extends to end of file
        if loop_length > 0:
            results.append({
                'source_file': os.path.basename(csv_file),
                'variant': variant,
                'loop_start': loop_start,
                'loop_end': group.iloc[-1][turn_col],
                'loop_duration_turns': loop_length + 1,
                'loop_content_length_chars': loop_content_len,
                'loop_content_preview': loop_preview,
                'is_exact_match': True
            })

    return results

# ==========================================
# MAIN EXECUTION
# ==========================================

fixed_files = [
    'male_pruned_8k.csv',
    'run_sequence_female.csv',
    'run_sequence_male.csv',
    'female_pruned_8k.csv'
]

all_results = []

for f in fixed_files:
    if os.path.exists(f):
        print(f"Analyzing: {f}")
        results = analyze_loops(f)
        all_results.extend(results)
        print(f"  Found {len(results)} loop segment(s)")
    else:
        print(f"⚠️ File not found: {f}")

# Write summary CSV
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_file = 'loop_detection_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ Summary saved to {summary_file}")
    print(f"   Total loop segments found: {len(all_results)}")

    # Print concise summary
    print("\n" + "="*60)
    print("LOOP DETECTION SUMMARY")
    print("="*60)
    for _, row in summary_df.iterrows():
        print(f"  {row['variant']:20s} | {str(row['loop_start']):5s} → {str(row['loop_end']):5s} | {row['loop_duration_turns']:2d} turns | {row['loop_content_length_chars']:5d} chars")
else:
    print("\n⚠️ No loops detected in any file.")