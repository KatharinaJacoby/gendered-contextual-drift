import csv

def load_csv(filepath):
    """Loads the labels CSV."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyze_trigger(data):
    """Analyzes the shift at Turn 3 (Gendered Trigger)."""
    # Find Turn 3
    turn_3 = next((t for t in data if t['turn_id'] == 'T0003'), None)
    turn_2 = next((t for t in data if t['turn_id'] == 'T0002'), None)

    if not turn_3 or not turn_2:
        print("Error: Could not find Turn 2 or Turn 3.")
        return

    # Extract scores
    score_2 = int(turn_2['response_raw_bias_score'])
    score_3 = int(turn_3['response_raw_bias_score'])
    intensity_2 = int(turn_2['gender_bias_intensity'])
    intensity_3 = int(turn_3['gender_bias_intensity'])

    delta_score = score_3 - score_2
    delta_intensity = intensity_3 - intensity_2

    print("--- Gendered Trigger Analysis (Turn 3) ---")
    print(f"Turn 2 Bias Score: {score_2}")
    print(f"Turn 3 Bias Score: {score_3}")
    print(f"Delta Score: {delta_score}")
    print(f"Turn 2 Intensity: {intensity_2}")
    print(f"Turn 3 Intensity: {intensity_3}")
    print(f"Delta Intensity: {delta_intensity}")

    if delta_score > 0:
        print(f"\n[RESULT] Significant increase in bias score detected ({delta_score} points).")
    else:
        print("\n[RESULT] No significant increase in bias score.")

    if delta_intensity > 0:
        print(f"[RESULT] Shift in gender bias intensity detected ({delta_intensity} levels).")
    else:
        print("[RESULT] No shift in gender bias intensity.")

if __name__ == "__main__":
    input_file = "meta_drift_labels_anonymized.csv"
    if __import__('os').path.exists(input_file):
        data = load_csv(input_file)
        analyze_trigger(data)
    else:
        print(f"Error: {input_file} not found. Run calculate_bias_scores.py first.")