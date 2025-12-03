#!/usr/bin/env python3
"""
Create a comprehensive coefficient table showing:
- Columns: combination, K, regime, and all regression variables
- Combination/K/regime columns: Color-coded for visual distinction
- Variable columns: Show coefficients, highlight significant ones (green=positive, red=negative)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("white")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def create_coefficient_table(
    results_df: pd.DataFrame,
    output_dir: Path,
    significance_threshold: float = 0.05
):
    """
    Create comprehensive coefficient table with color coding.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Regression results
    output_dir : Path
        Output directory
    significance_threshold : float
        P-value threshold for significance
    """
    # Get all unique variables (include ALL variables, not just significant ones)
    all_variables = sorted(results_df['variable'].unique())
    
    # Get all unique combinations, K values, and regimes
    all_combinations = sorted(results_df['combination'].unique())
    all_k_values = sorted(results_df['K'].unique())
    
    # Create a comprehensive table
    table_data = []
    
    for combo in all_combinations:
        for k in all_k_values:
            combo_k_data = results_df[
                (results_df['combination'] == combo) &
                (results_df['K'] == k)
            ]
            
            if len(combo_k_data) == 0:
                continue
            
            # Get regimes that have regression results
            regimes_with_results = sorted(combo_k_data['regime'].unique())
            
            # For K=k, we expect regimes 0, 1, ..., k-1
            # Include ALL expected regimes, even if they don't have results
            expected_regimes = list(range(k))
            
            for regime in expected_regimes:
                # Check if this regime has regression results
                regime_data = combo_k_data[combo_k_data['regime'] == regime]
                
                # Get average weight (regime probability) and R²
                avg_weight = 0.0
                r_squared = np.nan
                if len(regime_data) > 0:
                    # Get avg_weight and r_squared from any variable (should be same for all variables in same regime)
                    if 'avg_weight' in regime_data.columns:
                        avg_weight = regime_data['avg_weight'].iloc[0]
                    if 'r_squared' in regime_data.columns:
                        r_squared = regime_data['r_squared'].iloc[0]
                else:
                    # No regression results - regime was skipped
                    avg_weight = 0.0  # Will be highlighted in orange
                
                # Create row
                row = {
                    'combination': combo,
                    'K': k,
                    'regime': regime,
                    'avg_weight': avg_weight,
                    'r_squared': r_squared,
                    'has_results': len(regime_data) > 0
                }
                
                # Add coefficient for each variable (if exists)
                for var in all_variables:
                    var_data = regime_data[regime_data['variable'] == var]
                    if len(var_data) > 0:
                        coef = var_data['coefficient'].iloc[0]
                        pval = var_data['p_value'].iloc[0]
                        row[var] = coef
                        row[f'{var}_pval'] = pval
                        row[f'{var}_sig'] = (pval < significance_threshold) if pd.notna(pval) else False
                    else:
                        # No regression results for this regime-variable pair
                        row[var] = np.nan
                        row[f'{var}_pval'] = np.nan
                        row[f'{var}_sig'] = False
                
                table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    if len(table_df) == 0:
        print("No data to create table")
        return
    
    # Sort by combination, K, regime
    table_df = table_df.sort_values(['combination', 'K', 'regime']).reset_index(drop=True)
    
    # Track first row index of each regime for R² display (merge cells effect)
    # Each regime has multiple rows (one per variable), we only show R² in the first row
    regime_first_row_indices = set()
    current_key = None
    for idx, row in table_df.iterrows():
        key = (row['combination'], row['K'], row['regime'])
        if key != current_key:
            regime_first_row_indices.add(idx)
            current_key = key
    
    print(f"\nCreated table with {len(table_df)} rows (combination-K-regime combinations)")
    print(f"Variables: {len(all_variables)}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(max(20, len(all_variables) * 1.5), max(10, len(table_df) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create color maps for combination, K, regime
    unique_combos = table_df['combination'].unique()
    combo_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_combos)))
    combo_color_map = {combo: combo_colors[i] for i, combo in enumerate(unique_combos)}
    
    unique_k = sorted(table_df['K'].unique())
    k_colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_k)))
    k_color_map = {k: k_colors[i] for i, k in enumerate(unique_k)}
    
    unique_regimes = sorted(table_df['regime'].unique())
    regime_colors = plt.cm.Pastel2(np.linspace(0, 1, len(unique_regimes)))
    regime_color_map = {r: regime_colors[i] for i, r in enumerate(unique_regimes)}
    
    # Create table data for display
    display_data = []
    
    for idx, row in table_df.iterrows():
        # Check if this is the first row of this regime (for R² display)
        is_first_row_of_regime = idx in regime_first_row_indices
        
        display_row = [
            row['combination'],
            f"K={row['K']}",
            f"R{row['regime']}",
            f"{row['avg_weight']:.3f}" if pd.notna(row['avg_weight']) else "0.000",
            f"{row['r_squared']:.3f}" if (pd.notna(row['r_squared']) and is_first_row_of_regime) else ""
        ]
        
        # Add coefficients for each variable
        for var in all_variables:
            coef = row[var]
            is_sig = row[f'{var}_sig']
            
            if pd.isna(coef):
                display_row.append('')
            else:
                display_row.append(f'{coef:.4f}')
        
        display_data.append(display_row)
    
    # Format column headers: split underscores and wrap long names
    def format_header(name):
        """Format header: replace underscores, wrap long names."""
        # Replace underscores with spaces
        formatted = name.replace('_', ' ')
        
        # If longer than 10 characters, try to split into two lines
        if len(formatted) > 10:
            # Try to split at a space (prefer middle)
            words = formatted.split()
            if len(words) >= 2:
                # Split roughly in half
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                return f'{line1}\n{line2}'
            else:
                # Single long word - split at character 10
                return f'{formatted[:10]}\n{formatted[10:]}'
        return formatted
    
    # Format headers
    formatted_headers = ['Combination', 'K', 'Regime', 'Avg\nWeight', 'R²'] + [format_header(var) for var in all_variables]
    headers = ['Combination', 'K', 'Regime', 'avg_weight', 'r_squared'] + all_variables  # Keep original for data access
    
    # Create table with formatted headers
    table = ax.table(
        cellText=display_data,
        colLabels=formatted_headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style the table
    # Header row - no rotation, allow multi-line text, reduced height
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', ha='center', va='center', fontsize=6)
        # Further reduced height for headers
        cell.set_height(0.03)
    
    # Color code combination, K, regime columns
    for row_idx in range(len(table_df)):
        # Combination column
        cell = table[(row_idx + 1, 0)]
        combo = table_df.iloc[row_idx]['combination']
        cell.set_facecolor(combo_color_map[combo])
        
        # K column
        cell = table[(row_idx + 1, 1)]
        k = table_df.iloc[row_idx]['K']
        cell.set_facecolor(k_color_map[k])
        
        # Regime column
        cell = table[(row_idx + 1, 2)]
        regime = table_df.iloc[row_idx]['regime']
        cell.set_facecolor(regime_color_map[regime])
        
        # Avg Weight column
        cell = table[(row_idx + 1, 3)]
        avg_weight = table_df.iloc[row_idx]['avg_weight']
        has_results = table_df.iloc[row_idx]['has_results']
        
        # Highlight in orange if average weight < 0.1 (low probability regime)
        if pd.isna(avg_weight) or avg_weight < 0.1:
            cell.set_facecolor('#FFA500')  # Orange
            cell.set_text_props(weight='bold', color='darkred')
        else:
            cell.set_facecolor('#E8F5E9')  # Light green
            cell.set_text_props(color='black')
        
        # R² column (merged effect - only show in first row of each regime)
        cell = table[(row_idx + 1, 4)]
        r_squared = table_df.iloc[row_idx]['r_squared']
        is_first_row_of_regime = row_idx in regime_first_row_indices
        
        # Set background color for all rows of the regime
        if pd.notna(r_squared):
            cell.set_facecolor('#E8F4F8')  # Light blue-gray
            if is_first_row_of_regime:
                cell.set_text_props(color='black', fontsize=7, weight='bold')
            else:
                # Empty cell but same background (merged effect)
                cell.set_text_props(color='#E8F4F8')  # Same as background
        else:
            cell.set_facecolor('#F0F0F0')
            cell.set_text_props(color='gray')
        
        # Variable columns - highlight significant coefficients
        for var_idx, var in enumerate(all_variables):
            col_idx = var_idx + 5  # Offset by 5 (combination, K, regime, avg_weight, r_squared)
            cell = table[(row_idx + 1, col_idx)]
            
            coef = table_df.iloc[row_idx][var]
            is_sig = table_df.iloc[row_idx][f'{var}_sig']
            
            if pd.isna(coef):
                cell.set_facecolor('#F0F0F0')  # Light gray for missing
                cell.set_text_props(color='gray')
            elif is_sig:
                # Significant: green for positive, red for negative
                if coef > 0:
                    cell.set_facecolor('#90EE90')  # Light green
                    cell.set_text_props(weight='bold', color='darkgreen')
                else:
                    cell.set_facecolor('#FFB6C1')  # Light red
                    cell.set_text_props(weight='bold', color='darkred')
            else:
                # Not significant: white background
                cell.set_facecolor('white')
                cell.set_text_props(color='black')
    
    # Set title
    plt.title('Coefficient Table: All Combinations × K × Regimes\n(Using Soft Regime Assignments - Weighted Regressions)\n(Green=Significant Positive, Red=Significant Negative, Orange=Low Weight)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='black', label='Significant Positive'),
        Rectangle((0, 0), 1, 1, facecolor='#FFB6C1', edgecolor='black', label='Significant Negative'),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Not Significant'),
        Rectangle((0, 0), 1, 1, facecolor='#F0F0F0', edgecolor='black', label='Missing Data'),
        Rectangle((0, 0), 1, 1, facecolor='#FFA500', edgecolor='black', label='Low Weight (<0.1)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    # Save
    output_file = output_dir / 'coefficient_table_comprehensive.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved comprehensive coefficient table to: {output_file}")
    
    # Also save as CSV for reference
    csv_output = output_dir / 'coefficient_table_comprehensive.csv'
    # Create CSV with readable format
    csv_data = table_df[['combination', 'K', 'regime', 'avg_weight', 'r_squared'] + all_variables].copy()
    csv_data.to_csv(csv_output, index=False)
    print(f"✓ Saved CSV table to: {csv_output}")
    
    # Create a summary showing only significant coefficients
    sig_table_data = []
    for _, row in table_df.iterrows():
        sig_row = {
            'combination': row['combination'],
            'K': row['K'],
            'regime': row['regime']
        }
        
        sig_vars = []
        for var in all_variables:
            if row[f'{var}_sig']:
                coef = row[var]
                pval = row[f'{var}_pval']
                sig_vars.append(f"{var}={coef:.4f} (p={pval:.3f})")
        
        sig_row['significant_variables'] = '; '.join(sig_vars) if sig_vars else 'None'
        sig_table_data.append(sig_row)
    
    sig_table_df = pd.DataFrame(sig_table_data)
    sig_csv_output = output_dir / 'coefficient_table_significant_only.csv'
    sig_table_df.to_csv(sig_csv_output, index=False)
    print(f"✓ Saved significant-only summary to: {sig_csv_output}")
    
    return table_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python create_coefficient_table.py <results_csv_path> [output_dir]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent
    
    results_df = pd.read_csv(results_path)
    create_coefficient_table(results_df, output_dir)

