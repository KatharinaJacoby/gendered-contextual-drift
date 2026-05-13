import csv
import matplotlib.pyplot as plt

def load_csv(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def plot_drift(data):
    turns = []
    scores = []
    intensities = []

    for row in data:
        # Extract numeric ID for sorting
        tid = row['turn_id'].replace('T', '')
        turns.append(int(tid))
        scores.append(int(row['response_raw_bias_score']))
        intensities.append(int(row['gender_bias_intensity']))

    plt.figure(figsize=(12, 6))

    # Plot Bias Score
    plt.subplot(1, 2, 1)
    plt.plot(turns, scores, marker='o', linestyle='-', color='blue', label='Bias Score')
    plt.axvline(x=3, color='red', linestyle='--', label='Gendered Trigger (Turn 3)')
    plt.title('Bias Score Over Time')
    plt.xlabel('Turn Number')
    plt.ylabel('Bias Score')
    plt.legend()
    plt.grid(True)

    # Plot Gender Bias Intensity
    plt.subplot(1, 2, 2)
    plt.plot(turns, intensities, marker='s', linestyle='-', color='purple', label='Intensity')
    plt.axvline(x=3, color='red', linestyle='--', label='Gendered Trigger (Turn 3)')
    plt.title('Gender Bias Intensity Over Time')
    plt.xlabel('Turn Number')
    plt.ylabel('Intensity (1-3)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('drift_visualization.png')
    print("Visualization saved as 'drift_visualization.png'")
    plt.show()

if __name__ == "__main__":
    input_file = "meta_drift_labels_anonymized.csv"
    if __import__('os').path.exists(input_file):
        data = load_csv(input_file)
        plot_drift(data)
    else:
        print(f"Error: {input_file} not found.")