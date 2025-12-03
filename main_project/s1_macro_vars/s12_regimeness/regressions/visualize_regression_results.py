#!/usr/bin/env python3
"""
Visualization module for conditional regression results.

Creates various plots to visualize:
- Coefficient heatmaps by regime
- Statistical significance heatmaps
- Variable importance across regimes
- Best predictors by regime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_coefficient_heatmap(
    results_df: pd.DataFrame,
    combination: str,
    k: int,
    output_dir: Path,
    significance_threshold: float = 0.05
):
    """
    Create coefficient heatmap for a specific combination and K.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Regression results
    combination : str
        Variable combination name
    k : int
        Number of regimes
    output_dir : Path
        Output directory
    significance_threshold : float
        P-value threshold for significance
    """
    # Filter data
    subset = results_df[
        (results_df['combination'] == combination) &
        (results_df['K'] == k)
    ].copy()
    
    if len(subset) == 0:
        return
    
    # Create pivot table: regimes x variables
    pivot_coef = subset.pivot_table(
        index='regime',
        columns='variable',
        values='coefficient',
        aggfunc='mean'
    )
    
    # Create significance mask
    pivot_pval = subset.pivot_table(
        index='regime',
        columns='variable',
        values='p_value',
        aggfunc='mean'
    )
    significant_mask = pivot_pval < significance_threshold
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, pivot_coef.shape[1] * 0.5), max(6, pivot_coef.shape[0] * 0.8)))
    
    # Plot heatmap
    sns.heatmap(
        pivot_coef,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        mask=~significant_mask,  # Only show significant coefficients
        cbar_kws={'label': 'Coefficient'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f'Coefficient Heatmap: {combination}, K={k}\n(Significant only, p<{significance_threshold})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Macro Variables', fontsize=12)
    ax.set_ylabel('Regime', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    filename = f'coefficient_heatmap_{combination.replace(" ", "_")}_K{k}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def plot_significance_heatmap(
    results_df: pd.DataFrame,
    combination: str,
    k: int,
    output_dir: Path,
    significance_threshold: float = 0.05
):
    """
    Create significance heatmap (p-values) for a specific combination and K.
    """
    # Filter data
    subset = results_df[
        (results_df['combination'] == combination) &
        (results_df['K'] == k)
    ].copy()
    
    if len(subset) == 0:
        return
    
    # Create pivot table: regimes x variables
    pivot_pval = subset.pivot_table(
        index='regime',
        columns='variable',
        values='p_value',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, pivot_pval.shape[1] * 0.5), max(6, pivot_pval.shape[0] * 0.8)))
    
    # Plot heatmap (lower p-values = more significant)
    sns.heatmap(
        pivot_pval,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=significance_threshold,
        cbar_kws={'label': 'P-value'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f'Significance Heatmap (P-values): {combination}, K={k}\n(Lower = More Significant)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Macro Variables', fontsize=12)
    ax.set_ylabel('Regime', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    filename = f'significance_heatmap_{combination.replace(" ", "_")}_K{k}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def plot_top_predictors_by_regime(
    results_df: pd.DataFrame,
    combination: str,
    k: int,
    output_dir: Path,
    top_n: int = 10,
    significance_threshold: float = 0.05
):
    """
    Plot top predictors for each regime.
    """
    # Filter data
    subset = results_df[
        (results_df['combination'] == combination) &
        (results_df['K'] == k) &
        (results_df['p_value'] < significance_threshold) &
        (results_df['p_value'].notna())
    ].copy()
    
    if len(subset) == 0:
        return
    
    # Get unique regimes
    regimes = sorted(subset['regime'].unique())
    n_regimes = len(regimes)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 8))
    if n_regimes == 1:
        axes = [axes]
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        
        # Get top predictors for this regime
        regime_data = subset[subset['regime'] == regime].copy()
        regime_data = regime_data.sort_values('p_value').head(top_n)
        
        # Plot coefficients
        y_pos = np.arange(len(regime_data))
        colors = ['green' if c > 0 else 'red' for c in regime_data['coefficient']]
        
        ax.barh(y_pos, regime_data['coefficient'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regime_data['variable'], fontsize=9)
        ax.set_xlabel('Coefficient', fontsize=11)
        ax.set_title(f'Regime {regime}\n(Top {top_n} Significant Predictors)', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Top Predictors by Regime: {combination}, K={k}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filename = f'top_predictors_{combination.replace(" ", "_")}_K{k}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")


def plot_variable_importance_across_regimes(
    results_df: pd.DataFrame,
    output_dir: Path,
    significance_threshold: float = 0.05,
    top_n_vars: int = 15
):
    """
    Plot overall variable importance across all regimes.
    """
    # Filter significant results
    significant = results_df[
        (results_df['p_value'] < significance_threshold) &
        (results_df['p_value'].notna())
    ].copy()
    
    if len(significant) == 0:
        return
    
    # Calculate importance: average absolute t-statistic
    importance = significant.groupby('variable')['t_statistic'].apply(
        lambda x: np.abs(x).mean()
    ).sort_values(ascending=False).head(top_n_vars)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(importance))
    ax.barh(y_pos, importance.values, alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance.index, fontsize=10)
    ax.set_xlabel('Average |t-statistic|', fontsize=12)
    ax.set_title(f'Variable Importance Across All Regimes\n(Top {top_n_vars} Variables, Significant Only)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'variable_importance_overall.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: variable_importance_overall.png")


def create_all_visualizations(
    results_df: pd.DataFrame,
    output_dir: Path,
    significance_threshold: float = 0.05
):
    """
    Create all visualizations for regression results.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Get unique combinations and K values
    combinations = results_df['combination'].unique()
    k_values = sorted(results_df['K'].unique())
    
    print(f"\nCreating visualizations for:")
    print(f"  Combinations: {len(combinations)}")
    print(f"  K values: {k_values}")
    
    # Create heatmaps for each combination and K
    for combo in combinations:
        for k in k_values:
            subset = results_df[
                (results_df['combination'] == combo) &
                (results_df['K'] == k)
            ]
            
            if len(subset) == 0:
                continue
            
            print(f"\n{combo}, K={k}...")
            plot_coefficient_heatmap(results_df, combo, k, output_dir, significance_threshold)
            plot_significance_heatmap(results_df, combo, k, output_dir, significance_threshold)
            plot_top_predictors_by_regime(results_df, combo, k, output_dir, top_n=10, 
                                         significance_threshold=significance_threshold)
    
    # Overall visualizations
    print("\nCreating overall visualizations...")
    plot_variable_importance_across_regimes(results_df, output_dir, significance_threshold)
    
    print("\n✓ All visualizations created!")


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_regression_results.py <results_csv_path> [output_dir]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent
    
    results_df = pd.read_csv(results_path)
    create_all_visualizations(results_df, output_dir)

