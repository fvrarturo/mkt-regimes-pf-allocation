"""
Plotting and output functions for regression results visualization.

Functions:
- create_results_tables: Create CSV tables with regression results
- create_plots: Create all visualization plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_results_tables(results, output_dir):
    """
    Create CSV tables with regression results.
    
    Parameters:
    -----------
    results : dict
        Regression results dictionary
    output_dir : Path
        Output directory for CSV files
    
    Returns:
    --------
    tuple
        (results_df, summary_df, importance_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create combined table for all horizons
    all_results = []
    
    for h, res_dict in results.items():
        reg_res = res_dict['results']
        
        for var, coef, se, t_stat, p_val in zip(
            reg_res['variable_names'],
            reg_res['coefficients'],
            reg_res['std_errors'],
            reg_res['t_stats'],
            reg_res['p_values']
        ):
            all_results.append({
                'horizon': h,
                'variable': var,
                'coefficient': coef,
                'std_error': se,
                't_stat': t_stat,
                'p_value': p_val,
                'abs_t_stat': abs(t_stat),
                'significant_5pct': 1 if p_val < 0.05 else 0,
                'significant_1pct': 1 if p_val < 0.01 else 0
            })
        
        # Add intercept row
        all_results.append({
            'horizon': h,
            'variable': 'intercept',
            'coefficient': reg_res['intercept'],
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'abs_t_stat': np.nan,
            'significant_5pct': 0,
            'significant_1pct': 0
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save combined table
    combined_path = output_dir / "regression_results_all_horizons.csv"
    results_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined results table to {combined_path}")
    
    # Create summary table (one row per horizon)
    summary_rows = []
    for h, res_dict in results.items():
        reg_res = res_dict['results']
        summary_rows.append({
            'horizon': h,
            'r_squared': reg_res['r_squared'],
            'n_obs': reg_res['n_obs'],
            'n_variables': len(reg_res['variable_names'])
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "regression_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")
    
    # Create variable importance table (ranked by |t-stat|)
    importance_rows = []
    for h, res_dict in results.items():
        reg_res = res_dict['results']
        
        # Create DataFrame for this horizon
        horizon_df = pd.DataFrame({
            'variable': reg_res['variable_names'],
            'abs_t_stat': np.abs(reg_res['t_stats']),
            'coefficient': reg_res['coefficients'],
            't_stat': reg_res['t_stats'],
            'p_value': reg_res['p_values']
        })
        
        # Sort by absolute t-stat
        horizon_df = horizon_df.sort_values('abs_t_stat', ascending=False)
        horizon_df['rank'] = range(1, len(horizon_df) + 1)
        horizon_df['horizon'] = h
        
        importance_rows.append(horizon_df)
    
    if len(importance_rows) > 0:
        importance_df = pd.concat(importance_rows, ignore_index=True)
    else:
        importance_df = pd.DataFrame()
    
    importance_path = output_dir / "variable_importance_ranking.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved variable importance ranking to {importance_path}")
    
    return results_df, summary_df, importance_df


def create_plots(results, output_dir):
    """
    Create visualization plots.
    
    Parameters:
    -----------
    results : dict
        Regression results dictionary
    output_dir : Path
        Output directory for PNG files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Variable importance plot (|t-stat|) for each horizon
    n_horizons = len(results)
    fig, axes = plt.subplots(n_horizons, 1, figsize=(12, 5 * n_horizons))
    if n_horizons == 1:
        axes = [axes]
    
    for idx, (h, res_dict) in enumerate(sorted(results.items())):
        reg_res = res_dict['results']
        
        # Sort by absolute t-stat
        var_order = sorted(
            zip(reg_res['variable_names'], np.abs(reg_res['t_stats'])),
            key=lambda x: x[1],
            reverse=True
        )
        vars_sorted = [v[0] for v in var_order]
        t_stats_sorted = [v[1] for v in var_order]
        
        # Create bar plot
        ax = axes[idx]
        bars = ax.barh(range(len(vars_sorted)), t_stats_sorted)
        ax.set_yticks(range(len(vars_sorted)))
        ax.set_yticklabels(vars_sorted)
        ax.set_xlabel('|t-statistic|', fontsize=12)
        ax.set_title(f'Variable Importance (Horizon h = {h} months)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Color bars by significance
        for i, (bar, var) in enumerate(zip(bars, vars_sorted)):
            var_idx = reg_res['variable_names'].index(var)
            p_val = reg_res['p_values'][var_idx]
            if p_val < 0.01:
                bar.set_color('darkgreen')
            elif p_val < 0.05:
                bar.set_color('green')
            elif p_val < 0.10:
                bar.set_color('orange')
            else:
                bar.set_color('gray')
        
        # Add value labels
        for i, (bar, t_stat) in enumerate(zip(bars, t_stats_sorted)):
            ax.text(t_stat + max(t_stats_sorted) * 0.01, i, f'{t_stat:.2f}',
                   va='center', fontsize=9)
    
    plt.tight_layout()
    importance_plot_path = output_dir / "variable_importance_ranking.png"
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved variable importance plot to {importance_plot_path}")
    plt.close()
    
    # 2. Coefficient comparison across horizons
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all unique variables
    all_vars = set()
    for res_dict in results.values():
        all_vars.update(res_dict['results']['variable_names'])
    all_vars = sorted(list(all_vars))
    
    # Prepare data for grouped bar chart
    x = np.arange(len(all_vars))
    width = 0.15
    horizons_sorted = sorted(results.keys())
    
    for i, h in enumerate(horizons_sorted):
        reg_res = results[h]['results']
        coefs = []
        for var in all_vars:
            if var in reg_res['variable_names']:
                idx = reg_res['variable_names'].index(var)
                coefs.append(reg_res['coefficients'][idx])
            else:
                coefs.append(0)
        
        offset = (i - len(horizons_sorted) / 2 + 0.5) * width
        ax.bar(x + offset, coefs, width, label=f'h = {h}m', alpha=0.8)
    
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title('Coefficient Comparison Across Horizons', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_vars, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    coef_plot_path = output_dir / "coefficient_comparison.png"
    plt.savefig(coef_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved coefficient comparison plot to {coef_plot_path}")
    plt.close()
    
    # 3. R-squared by horizon
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons_sorted = sorted(results.keys())
    r_squareds = [results[h]['results']['r_squared'] for h in horizons_sorted]
    
    bars = ax.bar(horizons_sorted, r_squareds, alpha=0.7, color='steelblue')
    ax.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Model Fit (R²) by Forecast Horizon', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, r2 in zip(bars, r_squareds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{r2:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    r2_plot_path = output_dir / "r_squared_by_horizon.png"
    plt.savefig(r2_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved R² plot to {r2_plot_path}")
    plt.close()

