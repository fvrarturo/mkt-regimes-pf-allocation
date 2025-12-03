#!/usr/bin/env python3
"""
Create ranking plot of top 10 combination-K pairs by number of significant variables.
Shows which variables are significant, their coefficients, p-values, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def create_top_combinations_ranking(
    results_df: pd.DataFrame,
    output_dir: Path,
    significance_threshold: float = 0.05,
    top_n: int = 10
):
    """
    Create ranking plot of top combination-K pairs by number of significant variables.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Regression results
    output_dir : Path
        Output directory
    significance_threshold : float
        P-value threshold for significance
    top_n : int
        Number of top combinations to show
    """
    # Filter significant results
    significant = results_df[
        (results_df['p_value'] < significance_threshold) &
        (results_df['p_value'].notna())
    ].copy()
    
    if len(significant) == 0:
        print("No significant variables found")
        return
    
    # Group by combination and K, count unique significant variables
    ranking = significant.groupby(['combination', 'K']).agg({
        'variable': 'nunique',
        'regime': 'nunique',
        'coefficient': ['mean', 'std'],
        'p_value': 'mean',
        't_statistic': lambda x: np.abs(x).mean()
    }).reset_index()
    
    # Flatten column names
    ranking.columns = ['combination', 'K', 'n_significant_vars', 'n_regimes', 
                       'mean_coef', 'std_coef', 'mean_pval', 'mean_abs_tstat']
    
    # Create label for ranking
    ranking['label'] = ranking.apply(
        lambda row: f"{row['combination']}, K={row['K']}", axis=1
    )
    
    # Sort by number of significant variables (descending)
    ranking = ranking.sort_values('n_significant_vars', ascending=False).head(top_n)
    
    print(f"\nTop {top_n} combination-K pairs by number of significant variables:")
    print(ranking[['label', 'n_significant_vars', 'n_regimes', 'mean_pval']].to_string(index=False))
    
    # Get detailed variable information for top combinations
    top_labels = ranking['label'].values
    detailed_info = []
    
    for _, row in ranking.iterrows():
        combo = row['combination']
        k = row['K']
        
        # Get variables for this combination-K pair
        combo_k_data = significant[
            (significant['combination'] == combo) &
            (significant['K'] == k)
        ]
        
        # Aggregate by variable (across regimes)
        var_summary = combo_k_data.groupby('variable').agg({
            'coefficient': 'mean',
            'p_value': 'mean',
            't_statistic': lambda x: np.abs(x).mean(),
            'regime': 'nunique'
        }).reset_index()
        
        var_summary = var_summary.sort_values('p_value')
        
        # Add combination-K info
        var_summary['combination'] = combo
        var_summary['K'] = k
        var_summary['label'] = row['label']
        var_summary['rank'] = row.name + 1
        
        detailed_info.append(var_summary)
    
    detailed_df = pd.concat(detailed_info, ignore_index=True)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Bar chart of top combinations
    ax1 = fig.add_subplot(gs[0, :])
    
    y_pos = np.arange(len(ranking))
    bars = ax1.barh(y_pos, ranking['n_significant_vars'], 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (idx, row) in enumerate(ranking.iterrows()):
        ax1.text(row['n_significant_vars'] + 0.2, i, 
                f"{int(row['n_significant_vars'])} vars",
                va='center', fontsize=10, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ranking['label'], fontsize=10)
    ax1.set_xlabel('Number of Significant Variables', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Combination-K Pairs by Number of Significant Variables', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: Coefficient signs distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Count positive vs negative coefficients for top combinations
    coef_signs = []
    for _, row in ranking.iterrows():
        combo = row['combination']
        k = row['K']
        combo_k_data = significant[
            (significant['combination'] == combo) &
            (significant['K'] == k)
        ]
        
        positive = (combo_k_data['coefficient'] > 0).sum()
        negative = (combo_k_data['coefficient'] < 0).sum()
        
        coef_signs.append({
            'label': row['label'],
            'positive': positive,
            'negative': negative
        })
    
    signs_df = pd.DataFrame(coef_signs)
    signs_df = signs_df.set_index('label')
    
    x_pos = np.arange(len(signs_df))
    width = 0.35
    
    ax2.barh(x_pos - width/2, signs_df['positive'], width, 
            label='Positive', color='green', alpha=0.7, edgecolor='black')
    ax2.barh(x_pos + width/2, signs_df['negative'], width, 
            label='Negative', color='red', alpha=0.7, edgecolor='black')
    
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(signs_df.index, fontsize=9)
    ax2.set_xlabel('Number of Variables', fontsize=11, fontweight='bold')
    ax2.set_title('Coefficient Signs Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Plot 3: Variable frequency heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Count how many times each variable appears in top combinations
    var_counts = detailed_df.groupby('variable').agg({
        'label': 'nunique',
        'coefficient': lambda x: (x > 0).sum() - (x < 0).sum()  # Net positive count
    }).reset_index()
    var_counts.columns = ['variable', 'n_combinations', 'net_positive']
    var_counts = var_counts.sort_values('n_combinations', ascending=False).head(15)
    
    y_pos_vars = np.arange(len(var_counts))
    colors_vars = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                   for x in var_counts['net_positive']]
    
    bars_vars = ax3.barh(y_pos_vars, var_counts['n_combinations'], 
                         color=colors_vars, alpha=0.7, edgecolor='black')
    
    ax3.set_yticks(y_pos_vars)
    ax3.set_yticklabels(var_counts['variable'], fontsize=9)
    ax3.set_xlabel('Number of Top Combinations', fontsize=11, fontweight='bold')
    ax3.set_title('Most Frequent Significant Variables\n(Green=Net Positive, Red=Net Negative)', 
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    plt.suptitle('Top Combination-K Pairs: Ranking and Statistics', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = output_dir / 'top_combinations_ranking.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✓ Saved ranking plot to: {output_file}")
    
    # Create detailed table with variable breakdown for each top combination
    detailed_table = detailed_df[['rank', 'label', 'combination', 'K', 'variable', 
                                  'coefficient', 'p_value', 't_statistic', 'regime']].copy()
    detailed_table.columns = ['Rank', 'Label', 'Combination', 'K', 'Variable', 
                             'Coefficient', 'P-value', 'Avg |t-stat|', 'N Regimes']
    detailed_table = detailed_table.sort_values(['Rank', 'P-value'])
    
    output_table = output_dir / 'top_combinations_detailed.csv'
    detailed_table.to_csv(output_table, index=False)
    print(f"✓ Saved detailed table to: {output_table}")
    
    # Create summary statistics table
    summary_stats = ranking[['label', 'combination', 'K', 'n_significant_vars', 
                            'n_regimes', 'mean_coef', 'std_coef', 'mean_pval', 'mean_abs_tstat']].copy()
    summary_stats.columns = ['Label', 'Combination', 'K', 'N Significant Vars', 
                            'N Regimes', 'Mean Coef', 'Std Coef', 'Mean P-value', 'Mean |t-stat|']
    
    output_summary = output_dir / 'top_combinations_summary.csv'
    summary_stats.to_csv(output_summary, index=False)
    print(f"✓ Saved summary statistics to: {output_summary}")
    
    return ranking, detailed_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python plot_top_combinations.py <results_csv_path> [output_dir] [top_n]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    results_df = pd.read_csv(results_path)
    create_top_combinations_ranking(results_df, output_dir, top_n=top_n)

