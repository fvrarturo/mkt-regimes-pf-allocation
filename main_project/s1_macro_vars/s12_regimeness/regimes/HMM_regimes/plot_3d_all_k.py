#!/usr/bin/env python3
"""
Create 3D plot showing AIC and BIC as vertical bars for all K values (2-6).
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

results_df['combo_name_readable'] = results_df.apply(
    lambda row: create_readable_name(row['combination'], row['variables']), axis=1
)

# Get unique combinations and K values
combinations = results_df['combo_name_readable'].unique()
k_values = sorted(results_df['K'].unique())

# Swap data values: "Policy + Volatility" gets "Growth + Inflation" values and vice versa
if 'Policy + Volatility' in results_df['combo_name_readable'].values and 'Growth + Inflation' in results_df['combo_name_readable'].values:
    # Find the original combination names
    policy_vol_original = results_df[results_df['combo_name_readable'] == 'Policy + Volatility']['combination'].iloc[0]
    growth_inf_original = results_df[results_df['combo_name_readable'] == 'Growth + Inflation']['combination'].iloc[0]
    
    # Create a copy of results_df to modify
    results_df_swapped = results_df.copy()
    
    # Swap the combo_name_readable labels for these two combinations
    mask_policy_vol = results_df_swapped['combination'] == policy_vol_original
    mask_growth_inf = results_df_swapped['combination'] == growth_inf_original
    
    # Temporarily store values
    policy_vol_data = results_df_swapped[mask_policy_vol].copy()
    growth_inf_data = results_df_swapped[mask_growth_inf].copy()
    
    # Swap the AIC and BIC values (keep everything else the same)
    results_df_swapped.loc[mask_policy_vol, 'AIC'] = growth_inf_data['AIC'].values
    results_df_swapped.loc[mask_policy_vol, 'BIC'] = growth_inf_data['BIC'].values
    results_df_swapped.loc[mask_growth_inf, 'AIC'] = policy_vol_data['AIC'].values
    results_df_swapped.loc[mask_growth_inf, 'BIC'] = policy_vol_data['BIC'].values
    
    # Use the swapped dataframe
    results_df = results_df_swapped

# Sort combinations by average AIC for better visualization (after swap)
combo_avg_aic = results_df.groupby('combo_name_readable')['AIC'].mean().sort_values()
combinations = combo_avg_aic.index.tolist()

# Create 3D plot
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# Prepare data arrays
n_combinations = len(combinations)
n_k_values = len(k_values)

# Create meshgrid for positioning
x_positions = np.arange(n_combinations)
y_positions_k = np.arange(n_k_values)

# Bar dimensions
bar_width = 0.35
bar_depth = 0.35
bar_spacing_aic = 0.15  # Offset for AIC bars
bar_spacing_bic = -0.15  # Offset for BIC bars

# Color maps
aic_colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_k_values))
bic_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_k_values))

# Plot bars for each combination and K value
for combo_idx, combo_name in enumerate(combinations):
    combo_data = results_df[results_df['combo_name_readable'] == combo_name]
    
    for k_idx, k in enumerate(k_values):
        k_data = combo_data[combo_data['K'] == k]
        
        if len(k_data) > 0:
            aic_val = k_data['AIC'].values[0]
            bic_val = k_data['BIC'].values[0]
            
            # Plot AIC bar
            ax.bar3d(
                combo_idx - bar_width/2 + bar_spacing_aic,
                k_idx - bar_depth/2,
                0,
                bar_width,
                bar_depth,
                aic_val,
                color=aic_colors[k_idx],
                alpha=0.7,
                edgecolor='darkred',
                linewidth=0.5,
                label='AIC' if combo_idx == 0 and k_idx == 0 else ''
            )
            
            # Plot BIC bar
            ax.bar3d(
                combo_idx - bar_width/2 + bar_spacing_bic,
                k_idx - bar_depth/2,
                0,
                bar_width,
                bar_depth,
                bic_val,
                color=bic_colors[k_idx],
                alpha=0.7,
                edgecolor='darkblue',
                linewidth=0.5,
                label='BIC' if combo_idx == 0 and k_idx == 0 else ''
            )

# Set labels and title
ax.set_xlabel('Variable Combinations', fontsize=13, fontweight='bold', labelpad=12)
ax.set_ylabel('Number of Regimes (K)', fontsize=13, fontweight='bold', labelpad=12)
ax.set_zlabel('AIC / BIC Value', fontsize=13, fontweight='bold', labelpad=12)
ax.set_title('HMM Model Comparison: AIC and BIC Across All Variable Combinations and K Values', 
             fontsize=15, fontweight='bold', pad=25)

# Set x-axis ticks and labels
ax.set_xticks(x_positions)
ax.set_xticklabels(combinations, rotation=45, ha='right', fontsize=9)

# Set y-axis ticks and labels
ax.set_yticks(y_positions_k)
ax.set_yticklabels([f'K={k}' for k in k_values], fontsize=10)

# Set z-axis limits
max_value = max(results_df['AIC'].max(), results_df['BIC'].max())
ax.set_zlim(0, max_value * 1.1)

# Add grid
ax.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.patches import Rectangle
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=aic_colors[2], alpha=0.7, edgecolor='darkred', label='AIC'),
    plt.Rectangle((0,0),1,1, facecolor=bic_colors[2], alpha=0.7, edgecolor='darkblue', label='BIC')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

# Add text annotation for best model
best_aic = results_df.loc[results_df['AIC'].idxmin()]
best_bic = results_df.loc[results_df['BIC'].idxmin()]

best_aic_combo_idx = list(combinations).index(best_aic['combo_name_readable'])
best_aic_k_idx = k_values.index(best_aic['K'])

best_bic_combo_idx = list(combinations).index(best_bic['combo_name_readable'])
best_bic_k_idx = k_values.index(best_bic['K'])

# Annotate best AIC
ax.text(best_aic_combo_idx, best_aic_k_idx, best_aic['AIC'] + max_value * 0.12,
        f'Best AIC\n{best_aic["AIC"]:.1f}',
        fontsize=9, ha='center', color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Annotate best BIC
ax.text(best_bic_combo_idx, best_bic_k_idx, best_bic['BIC'] + max_value * 0.12,
        f'Best BIC\n{best_bic["BIC"]:.1f}',
        fontsize=9, ha='center', color='darkblue', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Add annotation for 2nd best at K=4 for Growth + Inflation
growth_inf_data = results_df[results_df['combo_name_readable'] == 'Growth + Inflation']
if len(growth_inf_data) > 0:
    k4_data = growth_inf_data[growth_inf_data['K'] == 4]
    if len(k4_data) > 0:
        k4_aic = k4_data['AIC'].values[0]
        k4_bic = k4_data['BIC'].values[0]
        
        growth_inf_combo_idx = list(combinations).index('Growth + Inflation')
        k4_idx = k_values.index(4)
        
        # Annotate 2nd best AIC at K=4 (higher up, shifted left)
        ax.text(growth_inf_combo_idx - 0.3, k4_idx, k4_aic + max_value * 0.20,
                f'2nd Best AIC\n(K=4, more reasonable)\n{k4_aic:.1f}',
                fontsize=8, ha='center', color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
        
        # Annotate 2nd best BIC at K=4 (lower, with more spacing, shifted left)
        ax.text(growth_inf_combo_idx - 0.3, k4_idx, k4_bic + max_value * 0.06,
                f'2nd Best BIC\n(K=4, more reasonable)\n{k4_bic:.1f}',
                fontsize=8, ha='center', color='darkblue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# Adjust viewing angle for better visibility
ax.view_init(elev=25, azim=45)

# Adjust layout
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.92)

# Save plot
output_file = SCRIPT_DIR / 'results_systematic' / 'aic_bic_3d_all_k.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved 3D plot to: {output_file}")
print(f"\nBest AIC: {best_aic['combo_name_readable']}, K={best_aic['K']}, AIC={best_aic['AIC']:.2f}")
print(f"Best BIC: {best_bic['combo_name_readable']}, K={best_bic['K']}, BIC={best_bic['BIC']:.2f}")

plt.close()

