"""
Plotting functions for conditional extremeness analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_marginal_effects(marginal_effects_dict, output_dir):
    """
    Plot marginal effects of macro variables in normal vs extreme states by regime.
    
    Parameters:
    -----------
    marginal_effects_dict : dict
        Dictionary mapping regime -> marginal effects DataFrame
    output_dir : Path
        Output directory
    """
    n_regimes = len(marginal_effects_dict)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6*n_regimes, 6))
    
    if n_regimes == 1:
        axes = [axes]
    
    for idx, (regime, me_df) in enumerate(marginal_effects_dict.items()):
        ax = axes[idx]
        
        variables = me_df['variable'].values
        normal_effects = me_df['effect_normal'].values
        extreme_effects = me_df['effect_extreme'].values
        
        x = np.arange(len(variables))
        width = 0.35
        
        ax.bar(x - width/2, normal_effects, width, label='Normal State', alpha=0.8)
        ax.bar(x + width/2, extreme_effects, width, label='Extreme State', alpha=0.8)
        
        ax.set_xlabel('Macro Variable', fontsize=11)
        ax.set_ylabel('Marginal Effect on ERP', fontsize=11)
        ax.set_title(f'Regime {regime}: Normal vs Extreme Effects', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace('_factor', '') for v in variables], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "marginal_effects_by_regime.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved marginal effects plot to {output_path}")
    plt.close()


def plot_regime_fragility_heatmap(effects_df, output_dir):
    """
    Create heatmap showing which regimes become "fragile" under extremeness.
    
    Parameters:
    -----------
    effects_df : pd.DataFrame
        DataFrame from extract_key_effects
    output_dir : Path
        Output directory
    """
    # Pivot to create heatmap: regimes x variables, values = interaction effect (delta)
    pivot_data = effects_df.pivot(index='regime', columns='variable', values='delta_interaction')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Interaction Effect (δ)'}, ax=ax)
    
    ax.set_title('Regime Fragility Under Extremeness\n(Interaction Effects: δ_r)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Macro Variable', fontsize=11)
    ax.set_ylabel('Regime', fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "regime_fragility_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved fragility heatmap to {output_path}")
    plt.close()


def plot_beta_comparison(effects_df, output_dir):
    """
    Plot β_r (normal) vs β_r + δ_r (extreme) comparison.
    
    Parameters:
    -----------
    effects_df : pd.DataFrame
        DataFrame from extract_key_effects
    output_dir : Path
        Output directory
    """
    n_vars = len(effects_df['variable'].unique())
    n_regimes = len(effects_df['regime'].unique())
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, var in enumerate(effects_df['variable'].unique()):
        ax = axes[idx]
        var_data = effects_df[effects_df['variable'] == var]
        
        regimes = var_data['regime'].values
        beta_normal = var_data['beta_normal'].values
        beta_extreme = var_data['beta_extreme'].values
        
        x = np.arange(len(regimes))
        width = 0.35
        
        ax.bar(x - width/2, beta_normal, width, label='β (Normal)', alpha=0.8)
        ax.bar(x + width/2, beta_extreme, width, label='β + δ (Extreme)', alpha=0.8)
        
        ax.set_xlabel('Regime', fontsize=11)
        ax.set_ylabel('Coefficient Value', fontsize=11)
        ax.set_title(f'{var.replace("_factor", "")}: Normal vs Extreme State Coefficients', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "beta_comparison_normal_vs_extreme.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved beta comparison plot to {output_path}")
    plt.close()

