"""
Plotting and Visualization Functions for Regime-Conditional Regression Results

This module contains functions to create comprehensive visualizations of
regression results including heatmaps, coefficient plots, and statistical summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_heatmaps(regressor, output_dir: Optional[Path] = None):
    """
    Create heatmaps for coefficients and t-statistics by horizon.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("\nCreating heatmaps...")
    
    if not regressor.coefficient_tables:
        regressor.create_coefficient_tables()
    
    horizons = sorted(regressor.regression_results['horizon'].unique())
    
    for horizon in horizons:
        # Coefficient heatmap
        coef_table = regressor.coefficient_tables.get(f'h{horizon}_coefficients')
        if coef_table is not None and len(coef_table) > 0:
            plt.figure(figsize=(12, max(8, len(coef_table) * 0.5)))
            sns.heatmap(
                coef_table,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Coefficient'}
            )
            plt.title(f'ERP Regression Coefficients by Regime (Horizon: {horizon} month(s))', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Regime', fontsize=12)
            plt.ylabel('Macro Variable', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / f'coefficient_heatmap_h{horizon}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: coefficient_heatmap_h{horizon}.png")
        
        # T-statistic heatmap
        tstat_table = regressor.coefficient_tables.get(f'h{horizon}_tstats')
        if tstat_table is not None and len(tstat_table) > 0:
            plt.figure(figsize=(12, max(8, len(tstat_table) * 0.5)))
            sns.heatmap(
                tstat_table,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                cbar_kws={'label': 't-statistic'}
            )
            plt.title(f'T-Statistics by Regime (Horizon: {horizon} month(s))', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Regime', fontsize=12)
            plt.ylabel('Macro Variable', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / f'tstat_heatmap_h{horizon}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: tstat_heatmap_h{horizon}.png")


def plot_significance_by_variable(regressor, output_dir: Optional[Path] = None):
    """
    Plot number of significant coefficients by variable across all regimes and horizons.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating significance summary plot...")
    
    results = regressor.regression_results.copy()
    results['significant'] = results['p_value'] < 0.05
    
    # Count significant results by variable
    sig_counts = results.groupby('variable')['significant'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sig_counts.plot(kind='barh', color='steelblue')
    plt.xlabel('Number of Significant Results (p < 0.05)', fontsize=12)
    plt.ylabel('Macro Variable', fontsize=12)
    plt.title('Total Significant Coefficients by Variable\n(across all regimes and horizons)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'significance_by_variable.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: significance_by_variable.png")


def plot_coefficient_distribution(regressor, output_dir: Optional[Path] = None):
    """
    Plot distribution of coefficients by significance level.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating coefficient distribution plot...")
    
    results = regressor.regression_results.copy()
    results['significance'] = 'Not Significant'
    results.loc[results['p_value'] < 0.01, 'significance'] = 'p < 0.01'
    results.loc[(results['p_value'] >= 0.01) & (results['p_value'] < 0.05), 'significance'] = 'p < 0.05'
    results.loc[(results['p_value'] >= 0.05) & (results['p_value'] < 0.10), 'significance'] = 'p < 0.10'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, horizon in enumerate(sorted(results['horizon'].unique())):
        horizon_data = results[results['horizon'] == horizon]
        
        for sig_level in ['p < 0.01', 'p < 0.05', 'p < 0.10', 'Not Significant']:
            sig_data = horizon_data[horizon_data['significance'] == sig_level]['coefficient']
            if len(sig_data) > 0:
                axes[i].hist(sig_data, alpha=0.6, label=sig_level, bins=30)
        
        axes[i].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[i].set_xlabel('Coefficient', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_title(f'Horizon: {horizon} month(s)', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Regression Coefficients by Significance Level', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: coefficient_distribution.png")


def plot_r_squared_by_regime(regressor, output_dir: Optional[Path] = None):
    """
    Plot R-squared values by regime and horizon.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating R-squared plot...")
    
    results = regressor.regression_results.copy()
    
    # Average R-squared by regime and horizon
    r2_summary = results.groupby(['regime', 'horizon', 'regime_name'])['r_squared'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    horizons = sorted(results['horizon'].unique())
    regimes = sorted(results['regime'].unique())
    x = np.arange(len(horizons))
    width = 0.8 / len(regimes)
    
    for i, regime in enumerate(regimes):
        regime_data = r2_summary[r2_summary['regime'] == regime]
        regime_name = regime_data['regime_name'].iloc[0] if len(regime_data) > 0 else f"Regime {regime}"
        r2_values = [regime_data[regime_data['horizon'] == h]['r_squared'].values[0] 
                     if len(regime_data[regime_data['horizon'] == h]) > 0 else 0 
                     for h in horizons]
        ax.bar(x + i * width, r2_values, width, label=regime_name, alpha=0.8)
    
    ax.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax.set_ylabel('Average R²', fontsize=12)
    ax.set_title('Average R² by Regime and Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'r_squared_by_regime.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: r_squared_by_regime.png")


def plot_top_predictors(regressor, output_dir: Optional[Path] = None, top_n: int = 10):
    """
    Plot top N predictors by absolute t-statistic for each horizon.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    top_n : int
        Number of top predictors to show
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating top predictors plot...")
    
    results = regressor.regression_results.copy()
    results['abs_tstat'] = results['t_statistic'].abs()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, horizon in enumerate(sorted(results['horizon'].unique())):
        horizon_data = results[results['horizon'] == horizon].copy()
        top_predictors = horizon_data.nlargest(top_n, 'abs_tstat')
        
        y_pos = np.arange(len(top_predictors))
        axes[i].barh(y_pos, top_predictors['t_statistic'], 
                    color=['green' if x > 0 else 'red' for x in top_predictors['t_statistic']])
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([f"{row['variable']} (R{row['regime']})" 
                                  for _, row in top_predictors.iterrows()], fontsize=8)
        axes[i].set_xlabel('t-statistic', fontsize=10)
        axes[i].set_title(f'Top {top_n} Predictors (Horizon: {horizon} month(s))', 
                         fontsize=12, fontweight='bold')
        axes[i].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Top Predictors by Absolute t-statistic', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_predictors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: top_predictors.png")


def plot_coefficient_differences(regressor, output_dir: Optional[Path] = None):
    """
    Plot significant coefficient differences across regimes.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    if regressor.statistical_tests is None or len(regressor.statistical_tests) == 0:
        print("  No coefficient difference tests available")
        return
    
    print("  Creating coefficient differences plot...")
    
    sig_tests = regressor.statistical_tests[regressor.statistical_tests['significant'] == True].copy()
    
    if len(sig_tests) == 0:
        print("    No significant differences found")
        return
    
    # Group by variable and count significant differences
    var_counts = sig_tests.groupby('variable').size().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    var_counts.head(20).plot(kind='barh', color='coral')
    plt.xlabel('Number of Significant Coefficient Differences', fontsize=12)
    plt.ylabel('Macro Variable', fontsize=12)
    plt.title('Variables with Significant Coefficient Differences Across Regimes\n(Top 20)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: coefficient_differences.png")


def plot_coefficient_evolution(regressor, output_dir: Optional[Path] = None):
    """
    Plot how coefficients change across forecast horizons for top variables.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating coefficient evolution plot...")
    
    results = regressor.regression_results.copy()
    results['abs_tstat'] = results['t_statistic'].abs()
    
    # Get top variables by average absolute t-statistic
    top_vars = results.groupby('variable')['abs_tstat'].mean().nlargest(8).index.tolist()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, var in enumerate(top_vars):
        var_data = results[results['variable'] == var].copy()
        
        for regime in sorted(var_data['regime'].unique()):
            regime_data = var_data[var_data['regime'] == regime]
            regime_name = regime_data['regime_name'].iloc[0] if 'regime_name' in regime_data.columns else f"R{regime}"
            
            horizons = sorted(regime_data['horizon'].unique())
            coefs = [regime_data[regime_data['horizon'] == h]['coefficient'].values[0] 
                     if len(regime_data[regime_data['horizon'] == h]) > 0 else np.nan
                     for h in horizons]
            pvals = [regime_data[regime_data['horizon'] == h]['p_value'].values[0] 
                     if len(regime_data[regime_data['horizon'] == h]) > 0 else 1.0
                     for h in horizons]
            
            # Color by significance
            colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'gray' for p in pvals]
            axes[idx].plot(horizons, coefs, marker='o', label=regime_name, linewidth=2, markersize=8)
            for i, (h, c, p) in enumerate(zip(horizons, coefs, pvals)):
                axes[idx].scatter(h, c, c=colors[i], s=100, zorder=5, edgecolors='black', linewidth=1)
        
        axes[idx].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[idx].set_xlabel('Horizon (months)', fontsize=10)
        axes[idx].set_ylabel('Coefficient', fontsize=10)
        axes[idx].set_title(f'{var}', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=7, loc='best')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Coefficient Evolution Across Forecast Horizons\n(Top 8 Variables)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: coefficient_evolution.png")


def plot_significance_heatmap(regressor, output_dir: Optional[Path] = None):
    """
    Create heatmap of significance levels by variable and regime.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating significance heatmap...")
    
    results = regressor.regression_results.copy()
    
    # Create significance score: 3 = p<0.01, 2 = p<0.05, 1 = p<0.10, 0 = not significant
    results['sig_score'] = 0
    results.loc[results['p_value'] < 0.01, 'sig_score'] = 3
    results.loc[(results['p_value'] >= 0.01) & (results['p_value'] < 0.05), 'sig_score'] = 2
    results.loc[(results['p_value'] >= 0.05) & (results['p_value'] < 0.10), 'sig_score'] = 1
    
    # Average significance score by variable and regime (across all horizons)
    sig_matrix = results.groupby(['variable', 'regime'])['sig_score'].mean().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(sig_matrix) * 0.4)))
    sns.heatmap(
        sig_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Significance Score\n(3=p<0.01, 2=p<0.05, 1=p<0.10, 0=ns)'},
        vmin=0,
        vmax=3
    )
    plt.title('Average Significance Score by Variable and Regime\n(Across All Horizons)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Regime', fontsize=12)
    plt.ylabel('Macro Variable', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'significance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: significance_heatmap.png")


def plot_sample_sizes(regressor, output_dir: Optional[Path] = None):
    """
    Plot sample sizes by regime and horizon.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating sample size plot...")
    
    results = regressor.regression_results.copy()
    
    # Average sample size by regime and horizon
    sample_summary = results.groupby(['regime', 'horizon', 'regime_name'])['n_observations'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    horizons = sorted(results['horizon'].unique())
    regimes = sorted(results['regime'].unique())
    x = np.arange(len(horizons))
    width = 0.8 / len(regimes)
    
    for i, regime in enumerate(regimes):
        regime_data = sample_summary[sample_summary['regime'] == regime]
        regime_name = regime_data['regime_name'].iloc[0] if len(regime_data) > 0 else f"Regime {regime}"
        n_obs = [regime_data[regime_data['horizon'] == h]['n_observations'].values[0] 
                 if len(regime_data[regime_data['horizon'] == h]) > 0 else 0 
                 for h in horizons]
        ax.bar(x + i * width, n_obs, width, label=regime_name, alpha=0.8)
    
    ax.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax.set_ylabel('Average Sample Size', fontsize=12)
    ax.set_title('Average Sample Size by Regime and Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: sample_sizes.png")


def plot_best_predictors_by_regime(regressor, output_dir: Optional[Path] = None, top_n: int = 5):
    """
    Plot best predictors for each regime.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    top_n : int
        Number of top predictors per regime
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating best predictors by regime plot...")
    
    results = regressor.regression_results.copy()
    results['abs_tstat'] = results['t_statistic'].abs()
    
    # Get best predictors for each regime (across all horizons)
    regimes = sorted(results['regime'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, regime in enumerate(regimes):
        regime_data = results[results['regime'] == regime].copy()
        regime_name = regime_data['regime_name'].iloc[0] if 'regime_name' in regime_data.columns else f"Regime {regime}"
        
        # Average absolute t-statistic by variable
        var_avg_tstat = regime_data.groupby('variable')['abs_tstat'].mean().sort_values(ascending=False)
        top_vars = var_avg_tstat.head(top_n)
        
        y_pos = np.arange(len(top_vars))
        colors = ['green' if t > 2 else 'orange' if t > 1.5 else 'gray' for t in top_vars.values]
        
        axes[idx].barh(y_pos, top_vars.values, color=colors, alpha=0.7)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(top_vars.index, fontsize=9)
        axes[idx].set_xlabel('Average |t-statistic|', fontsize=10)
        axes[idx].set_title(f'{regime_name}\n(Top {top_n} Predictors)', fontsize=12, fontweight='bold')
        axes[idx].axvline(1.96, color='red', linestyle='--', linewidth=1, alpha=0.7, label='p=0.05 threshold')
        axes[idx].axvline(2.58, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='p=0.01 threshold')
        axes[idx].legend(fontsize=7)
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Best Predictors by Regime\n(Average |t-statistic| across all horizons)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'best_predictors_by_regime.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: best_predictors_by_regime.png")


def plot_pvalue_distribution(regressor, output_dir: Optional[Path] = None):
    """
    Plot distribution of p-values to check for multiple testing issues.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating p-value distribution plot...")
    
    results = regressor.regression_results.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, horizon in enumerate(sorted(results['horizon'].unique())):
        horizon_data = results[results['horizon'] == horizon]
        
        axes[i].hist(horizon_data['p_value'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
        axes[i].axvline(0.01, color='darkred', linestyle='--', linewidth=2, label='p=0.01')
        axes[i].set_xlabel('p-value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_title(f'Horizon: {horizon} month(s)', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add text with counts
        n_sig_05 = (horizon_data['p_value'] < 0.05).sum()
        n_sig_01 = (horizon_data['p_value'] < 0.01).sum()
        axes[i].text(0.7, 0.9, f'p<0.05: {n_sig_05}\np<0.01: {n_sig_01}', 
                    transform=axes[i].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Distribution of p-values by Forecast Horizon', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'pvalue_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: pvalue_distribution.png")


def create_all_plots(regressor, output_dir: Optional[Path] = None):
    """
    Create all visualization plots for regression results.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for plots
    """
    print("\nCreating comprehensive visualizations...")
    
    if output_dir is None:
        output_dir = regressor.output_dir
    
    # Create all plots
    create_heatmaps(regressor, output_dir)
    plot_significance_by_variable(regressor, output_dir)
    plot_coefficient_distribution(regressor, output_dir)
    plot_r_squared_by_regime(regressor, output_dir)
    plot_top_predictors(regressor, output_dir)
    plot_coefficient_differences(regressor, output_dir)
    
    # New enhanced plots
    plot_coefficient_evolution(regressor, output_dir)
    plot_significance_heatmap(regressor, output_dir)
    plot_sample_sizes(regressor, output_dir)
    plot_best_predictors_by_regime(regressor, output_dir)
    plot_pvalue_distribution(regressor, output_dir)
    
    print("\nAll visualizations created!")

