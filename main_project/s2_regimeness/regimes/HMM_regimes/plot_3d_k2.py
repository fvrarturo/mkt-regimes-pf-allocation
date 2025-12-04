#!/usr/bin/env python3
"""
Create 3D plot showing AIC and BIC as vertical bars for K=2 scenario.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Add current directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Load results
results_file = SCRIPT_DIR / 'results_systematic' / 'all_model_results.csv'
results_df = pd.read_csv(results_file)

# Filter for K=2 only
k2_results = results_df[results_df['K'] == 2].copy()

# Create readable combination names
def create_readable_name(combo_name, variables):
    """Create a readable name for the combination."""
    if combo_name == 'all_4vars':
        return 'All 4 Variables'
    
    # Extract variable names from the variables string
    var_list = [v.strip().replace('_factor', '') for v in variables.split(',')]
    var_names = {
        'growth': 'Growth',
        'inflation': 'Inflation',
        'monetary_policy': 'Policy',
        'market_volatility': 'Volatility'
    }
    readable_vars = [var_names.get(v, v.title()) for v in var_list]
    return ' + '.join(readable_vars)

k2_results['combo_name_readable'] = k2_results.apply(
    lambda row: create_readable_name(row['combination'], row['variables']), axis=1
)

# Sort by AIC for better visualization
k2_results = k2_results.sort_values('AIC').reset_index(drop=True)

# Create 3D plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Prepare data
n_combinations = len(k2_results)
x_positions = np.arange(n_combinations)
y_positions_aic = np.zeros(n_combinations)  # AIC bars at y=0
y_positions_bic = np.ones(n_combinations)   # BIC bars at y=1
z_base = np.zeros(n_combinations)

# Get AIC and BIC values
aic_values = k2_results['AIC'].values
bic_values = k2_results['BIC'].values

# Bar width
bar_width = 0.3
bar_depth = 0.3

# Create color maps
aic_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_combinations))
bic_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_combinations))

# Plot AIC bars
for i, (x, aic) in enumerate(zip(x_positions, aic_values)):
    ax.bar3d(
        x - bar_width/2, y_positions_aic[i] - bar_depth/2, z_base[i],
        bar_width, bar_depth, aic,
        color=aic_colors[i],
        alpha=0.8,
        label='AIC' if i == 0 else ''
    )

# Plot BIC bars
for i, (x, bic) in enumerate(zip(x_positions, bic_values)):
    ax.bar3d(
        x - bar_width/2, y_positions_bic[i] - bar_depth/2, z_base[i],
        bar_width, bar_depth, bic,
        color=bic_colors[i],
        alpha=0.8,
        label='BIC' if i == 0 else ''
    )

# Set labels and title
ax.set_xlabel('Variable Combinations', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Metric Type', fontsize=12, fontweight='bold', labelpad=10)
ax.set_zlabel('AIC / BIC Value', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('HMM Model Comparison: AIC and BIC for K=2 Regimes', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis ticks and labels
ax.set_xticks(x_positions)
ax.set_xticklabels(k2_results['combo_name_readable'], rotation=45, ha='right', fontsize=9)

# Set y-axis ticks and labels
ax.set_yticks([0, 1])
ax.set_yticklabels(['AIC', 'BIC'], fontsize=11)

# Set z-axis limits to show all bars clearly
max_value = max(aic_values.max(), bic_values.max())
ax.set_zlim(0, max_value * 1.1)

# Add grid
ax.grid(True, alpha=0.3)

# Add legend
ax.legend(loc='upper left', fontsize=11)

# Add text annotations for min values
min_aic_idx = aic_values.argmin()
min_bic_idx = bic_values.argmin()

ax.text(x_positions[min_aic_idx], 0, aic_values[min_aic_idx] + max_value * 0.05,
        f'Min AIC\n{aic_values[min_aic_idx]:.1f}',
        fontsize=9, ha='center', color='darkred', fontweight='bold')

ax.text(x_positions[min_bic_idx], 1, bic_values[min_bic_idx] + max_value * 0.05,
        f'Min BIC\n{bic_values[min_bic_idx]:.1f}',
        fontsize=9, ha='center', color='darkblue', fontweight='bold')

# Adjust viewing angle for better visibility
ax.view_init(elev=25, azim=45)

# Adjust layout
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)

# Save plot
output_file = SCRIPT_DIR / 'results_systematic' / 'aic_bic_3d_k2.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved 3D plot to: {output_file}")

plt.close()

