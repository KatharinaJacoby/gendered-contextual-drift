import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the data
try:
    df = pd.read_csv('loop_detection_summary.csv')
except FileNotFoundError:
    print("Error: 'loop_detection_summary.csv' not found. Please run 'analyze_fixed_loops.py' first.")
    exit()

# 2. Preprocess the data
# Extract context density (2k or 8k) from the variant name or source file
# We assume the source_file or variant name contains the density info.
# Based on your file names:
# - 'run_sequence_*.csv' usually contains 2k and 8k runs mixed.
# - '*_pruned_8k.csv' is 8k.
# Let's infer density from the 'variant' column if it contains 'C2' (2k) or 'C3/C4' (8k) or 'PRUNED_8k'.

def get_density(variant):
    v = str(variant).upper()
    if 'C2' in v: return '2k'
    if 'C3' in v or 'C4' in v or '8K' in v: return '8k'
    return 'Unknown'

df['Density'] = df['variant'].apply(get_density)

# Filter out 'Unknown' or non-loop rows if necessary (though this file should only have loops)
df = df[df['Density'] != 'Unknown']

# 3. Aggregate Data
# We want to show the MAXIMUM loop duration for each run to highlight the "Lock-in" vs "Fluctuation"
# Or the average. Let's do Max Duration per Run ID to show the worst-case scenario.
# First, group by a unique Run ID. Since 'variant' might have multiple entries per run (multiple loop segments),
# we take the max duration per variant.

run_stats = df.groupby(['Density', 'variant'])['loop_duration_turns'].max().reset_index()
run_stats.rename(columns={'loop_duration_turns': 'Max_Loop_Duration'}, inplace=True)

# 4. Plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors
colors = {'2k': '#FF6B6B', '8k': '#4ECDC4'}

# Create boxplots or swarmplots to show distribution
# Boxplot is good for showing the spread (Fluctuation vs Lock-in)
sns.boxplot(x='Density', y='Max_Loop_Duration', data=run_stats, palette=colors, ax=ax, width=0.5)

# Add individual data points (swarm) to show every run
sns.swarmplot(x='Density', y='Max_Loop_Duration', data=run_stats, color='black', size=8, ax=ax, alpha=0.7)

# Styling
ax.set_title('Phase Transition: Loop Duration by Context Density', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Context Density', fontsize=12)
ax.set_ylabel('Maximum Loop Duration (Turns)', fontsize=12)
ax.set_ylim(0, 30) # Set limit to clearly show the jump

# Add annotations for the "Phase Transition"
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Fluctuation Threshold')
ax.text(0.5, 28, 'STATIC ENTRAPMENT (8k)', fontsize=12, fontweight='bold', color='#4ECDC4', ha='center', va='bottom')
ax.text(0.5, 4, 'FLUCTUATING DRIFT (2k)', fontsize=12, fontweight='bold', color='#FF6B6B', ha='center', va='bottom')

# Remove legend if not needed, or customize
ax.legend().set_visible(False)

plt.tight_layout()
plt.savefig('phase_transition_plot.png', dpi=300)
print("✅ Plot saved as 'phase_transition_plot.png'")

# Optional: Print summary stats
print("\n--- Summary Statistics ---")
print(run_stats.groupby('Density')['Max_Loop_Duration'].describe())