"""
Plotting functions for cross-model comparison.

Functions:
- plot_performance_comparison: Compare RMSE/MAE across models
- plot_forecast_comparison: Compare forecasts vs actuals for all models
- plot_dm_test_results: Visualize Diebold-Mariano test results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_performance_comparison(
    performance_df: pd.DataFrame,
    metric: str = 'rmse',
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot performance comparison across models.
    
    Parameters:
    -----------
    performance_df : pd.DataFrame
        Performance table with columns: model, variable, horizon, rmse, mae
    metric : str
        Metric to plot ('rmse' or 'mae')
    output_dir : Path, optional
        Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    variables = ['Growth', 'Inflation']
    horizons = [1, 3, 6]
    
    # Model colors
    model_colors = {
        'TVP-VAR': '#1f77b4',
        'XGBoost (Macro)': '#ff7f0e',
        'XGBoost (Macro+Sent)': '#2ca02c',
        'LSTM': '#d62728'
    }
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        
        # Filter data for this variable
        var_data = performance_df[performance_df['variable'] == var]
        
        # Prepare data for plotting
        x = np.arange(len(horizons))
        width = 0.2
        
        models = var_data['model'].unique()
        for i, model in enumerate(models):
            model_data = var_data[var_data['model'] == model]
            values = []
            for h in horizons:
                h_data = model_data[model_data['horizon'] == h]
                if len(h_data) > 0:
                    values.append(h_data.iloc[0][metric])
                else:
                    values.append(np.nan)
            
            ax.bar(x + i * width, values, width, label=model, 
                  color=model_colors.get(model, f'C{i}'), alpha=0.8)
        
        ax.set_xlabel('Forecast Horizon (months)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{var} Forecast Performance ({metric.upper()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f'h={h}m' for h in horizons])
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"performance_comparison_{metric}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {output_dir / filename}")
    
    plt.close()


def plot_heatmap_performance(
    rmse_table: pd.DataFrame,
    mae_table: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create heatmap of performance metrics.
    
    Parameters:
    -----------
    rmse_table : pd.DataFrame
        Pivoted RMSE table
    mae_table : pd.DataFrame
        Pivoted MAE table
    output_dir : Path, optional
        Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # RMSE heatmap
    sns.heatmap(rmse_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'RMSE'}, ax=axes[0], linewidths=0.5)
    axes[0].set_title('RMSE Comparison Across Models', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Forecast Horizon', fontsize=12)
    axes[0].set_ylabel('Model & Variable', fontsize=12)
    
    # MAE heatmap
    sns.heatmap(mae_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'MAE'}, ax=axes[1], linewidths=0.5)
    axes[1].set_title('MAE Comparison Across Models', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Forecast Horizon', fontsize=12)
    axes[1].set_ylabel('Model & Variable', fontsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "performance_heatmap.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance heatmap to {output_dir / filename}")
    
    plt.close()


def plot_forecast_comparison_all_models(
    forecasts_dict: Dict[str, pd.Series],
    actuals: pd.Series,
    variable_name: str,
    horizon: int,
    start_date: str = "2008-01-01",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot forecasts from all models against actuals.
    
    Parameters:
    -----------
    forecasts_dict : dict
        Dictionary mapping model names to forecast Series
    actuals : pd.Series
        Actual values
    variable_name : str
        Name of variable
    horizon : int
        Forecast horizon
    start_date : str
        Start date for plotting
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Filter by start date
    actuals_filtered = actuals[actuals.index >= start_date]
    
    # Plot actuals
    ax.plot(actuals_filtered.index, actuals_filtered.values, 
           label='Actual', linewidth=3, color='black', alpha=0.9, zorder=10)
    
    # Model colors and styles
    model_styles = {
        'TVP-VAR': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2},
        'XGBoost (Macro)': {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 2},
        'XGBoost (Macro+Sent)': {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 2},
        'LSTM': {'color': '#d62728', 'linestyle': '-.', 'linewidth': 2}
    }
    
    # Plot forecasts
    for model_name, forecast_series in forecasts_dict.items():
        # Align forecasts with actuals (forecasts are at origin dates)
        plot_data = []
        for forecast_date in forecast_series.index:
            target_date = forecast_date + pd.DateOffset(months=horizon)
            if target_date in actuals_filtered.index:
                plot_data.append({
                    'date': target_date,
                    'forecast': forecast_series.loc[forecast_date]
                })
        
        if len(plot_data) > 0:
            plot_df = pd.DataFrame(plot_data).set_index('date').sort_index()
            style = model_styles.get(model_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
            ax.plot(plot_df.index, plot_df['forecast'], 
                   label=model_name, alpha=0.8, **style)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(variable_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{variable_name.replace("_", " ").title()} Forecast Comparison (h={horizon}m)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"forecast_comparison_all_{variable_name.replace('_', '_')}_h{horizon}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved forecast comparison plot to {output_dir / filename}")
    
    plt.close()


def plot_dm_test_results(
    dm_results: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Visualize Diebold-Mariano test results.
    
    Parameters:
    -----------
    dm_results : pd.DataFrame
        DM test results with columns: model1, model2, variable, horizon, 
        dm_statistic, p_value
    output_dir : Path, optional
        Directory to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: DM statistics
    ax1 = axes[0]
    pivot_stat = dm_results.pivot_table(
        index=['model1', 'model2'],
        columns=['variable', 'horizon'],
        values='dm_statistic',
        aggfunc='first'
    )
    
    sns.heatmap(pivot_stat, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, cbar_kws={'label': 'DM Statistic'}, 
                ax=ax1, linewidths=0.5)
    ax1.set_title('Diebold-Mariano Test Statistics', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Variable & Horizon', fontsize=12)
    ax1.set_ylabel('Model Comparison', fontsize=12)
    
    # Plot 2: P-values (significance)
    ax2 = axes[1]
    pivot_pval = dm_results.pivot_table(
        index=['model1', 'model2'],
        columns=['variable', 'horizon'],
        values='p_value',
        aggfunc='first'
    )
    
    # Create significance mask (p < 0.05)
    significance_mask = pivot_pval < 0.05
    
    sns.heatmap(pivot_pval, annot=True, fmt='.3f', cmap='YlOrRd', 
                mask=~significance_mask, cbar_kws={'label': 'P-value'}, 
                ax=ax2, linewidths=0.5)
    ax2.set_title('Diebold-Mariano Test P-values (Significant: p < 0.05)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Variable & Horizon', fontsize=12)
    ax2.set_ylabel('Model Comparison', fontsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "dm_test_results.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved DM test results plot to {output_dir / filename}")
    
    plt.close()

