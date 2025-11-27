"""
Plotting module for TVP-VAR results visualization.

Functions:
- plot_forecast_vs_realized: Plot forecasts against actual values
- plot_time_varying_coefficients: Plot time-varying coefficients
- plot_impulse_responses: Plot impulse response functions
- plot_forecast_performance: Plot forecast performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_forecast_vs_realized(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: List[int] = [1, 3, 6],
    variable_name: str = "Variable",
    start_date: Optional[str] = "2008-01-01",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot forecasts against actual values for all horizons in one plot.
    
    Parameters:
    -----------
    forecasts : pd.DataFrame
        Forecasts with columns h_1, h_3, h_6, indexed by forecast date
    actuals : pd.Series
        Actual values, indexed by date
    horizons : list
        Forecast horizons
    variable_name : str
        Name of variable being forecasted
    start_date : str, optional
        Start date for plotting (default: "2008-01-01")
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color map for horizons
    horizon_colors = plt.cm.viridis(np.linspace(0, 1, len(horizons)))
    
    # Filter actuals by start date
    if start_date:
        start_dt = pd.to_datetime(start_date)
        actuals_filtered = actuals[actuals.index >= start_dt]
    else:
        actuals_filtered = actuals
    
    # Plot actuals
    ax.plot(actuals_filtered.index, actuals_filtered.values, label='Actual', linewidth=2.5, 
           color='black', alpha=0.9, zorder=10)
    
    # Plot forecasts for each horizon
    for idx, h in enumerate(horizons):
        col_name = f'h_{h}'
        if col_name not in forecasts.columns:
            continue
        
        # Prepare data for plotting
        plot_data = []
        for forecast_date in forecasts.index:
            target_date = forecast_date + pd.DateOffset(months=h)
            
            if target_date in actuals.index and pd.notna(forecasts.loc[forecast_date, col_name]):
                # Filter by start date
                if start_date and target_date < pd.to_datetime(start_date):
                    continue
                    
                plot_data.append({
                    'date': target_date,
                    'forecast': forecasts.loc[forecast_date, col_name]
                })
        
        if len(plot_data) == 0:
            continue
        
        plot_df = pd.DataFrame(plot_data).set_index('date').sort_index()
        
        # Plot forecast line
        ax.plot(plot_df.index, plot_df['forecast'], 
               label=f'Forecast h={h}m', 
               linewidth=2, 
               alpha=0.75, 
               linestyle='--',
               color=horizon_colors[idx],
               marker='o',
               markersize=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{variable_name}', fontsize=12)
    ax.set_title(f'{variable_name} Forecast vs Actual (All Horizons)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"forecast_vs_realized_{variable_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved forecast plot to {output_dir / filename}")
    
    plt.close()


def plot_time_varying_coefficients(
    coefficients: pd.DataFrame,
    variable_name: str,
    coefficient_labels: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot time-varying coefficients for a variable.
    
    Parameters:
    -----------
    coefficients : pd.DataFrame
        Coefficients over time, indexed by date
    variable_name : str
        Name of variable
    coefficient_labels : list, optional
        Labels for coefficients. If None, uses column names.
    output_dir : Path, optional
        Directory to save plot
    """
    n_coefs = len(coefficients.columns)
    
    # Select a subset of coefficients to plot (to avoid clutter)
    # Plot intercept and a few key coefficients
    if n_coefs > 5:
        # Plot intercept and first few coefficients
        cols_to_plot = ['coef_0'] + [f'coef_{i}' for i in range(1, min(5, n_coefs))]
        plot_data = coefficients[cols_to_plot].copy()
    else:
        plot_data = coefficients.copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for col in plot_data.columns:
        ax.plot(plot_data.index, plot_data[col], label=col, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title(f'Time-Varying Coefficients: {variable_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"time_varying_coefficients_{variable_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved coefficient plot to {output_dir / filename}")
    
    plt.close()


def plot_impulse_responses(
    irf_data: pd.DataFrame,
    shock_var: str,
    response_vars: List[str],
    forecast_date: pd.Timestamp,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot impulse response functions.
    
    Parameters:
    -----------
    irf_data : pd.DataFrame
        IRF data with periods as index and response variables as columns
    shock_var : str
        Name of shocked variable
    response_vars : list
        Names of response variables
    forecast_date : pd.Timestamp
        Date for which IRF was computed
    output_dir : Path, optional
        Directory to save plot
    """
    n_responses = len(response_vars)
    fig, axes = plt.subplots(n_responses, 1, figsize=(12, 4 * n_responses))
    
    if n_responses == 1:
        axes = [axes]
    
    for idx, response_var in enumerate(response_vars):
        if response_var not in irf_data.columns:
            continue
        
        ax = axes[idx]
        ax.plot(irf_data.index, irf_data[response_var], linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Periods Ahead', fontsize=11)
        ax.set_ylabel('Response', fontsize=11)
        ax.set_title(f'Response of {response_var} to {shock_var} Shock\n(As of {forecast_date.date()})', 
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"irf_{shock_var.lower()}_shock_{forecast_date.strftime('%Y%m%d')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved IRF plot to {output_dir / filename}")
    
    plt.close()


def plot_aggregated_irfs(
    irf_data_dict: Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    shock_var: str,
    response_vars: List[str],
    forecast_dates: List[pd.Timestamp],
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot aggregated impulse response functions from multiple dates with confidence intervals.
    
    Parameters:
    -----------
    irf_data_dict : dict
        Dictionary mapping forecast dates to (irf_df, lower_ci_df, upper_ci_df) tuples
    shock_var : str
        Name of shocked variable
    response_vars : list
        Names of response variables
    forecast_dates : list
        List of forecast dates (for labeling)
    output_dir : Path, optional
        Directory to save plot
    """
    n_responses = len(response_vars)
    fig, axes = plt.subplots(n_responses, 1, figsize=(14, 5 * n_responses))
    
    if n_responses == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(forecast_dates)))
    
    for idx, response_var in enumerate(response_vars):
        ax = axes[idx]
        
        # Plot IRFs from each date
        for date_idx, forecast_date in enumerate(forecast_dates):
            if forecast_date not in irf_data_dict:
                continue
            
            irf_df, lower_ci_df, upper_ci_df = irf_data_dict[forecast_date]
            
            if response_var not in irf_df.columns:
                continue
            
            irf_values = irf_df[response_var].values
            periods = len(irf_values)
            
            # Plot confidence intervals if available
            if lower_ci_df is not None and upper_ci_df is not None:
                if response_var in lower_ci_df.columns and response_var in upper_ci_df.columns:
                    lower_ci = lower_ci_df[response_var].values
                    upper_ci = upper_ci_df[response_var].values
                    
                    # Shaded confidence interval
                    ax.fill_between(
                        range(periods),
                        lower_ci,
                        upper_ci,
                        alpha=0.2,
                        color=colors[date_idx],
                        label=f'{forecast_date.date()} (95% CI)' if date_idx == 0 else None
                    )
            
            # Plot point estimate
            label = f'{forecast_date.date()}' if date_idx < 3 else None  # Only label first 3
            ax.plot(
                range(periods),
                irf_values,
                linewidth=2,
                marker='o',
                markersize=3,
                color=colors[date_idx],
                label=label,
                alpha=0.8
            )
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Periods Ahead', fontsize=12)
        ax.set_ylabel('Response', fontsize=12)
        ax.set_title(f'Response of {response_var.replace("_", " ").title()} to {shock_var.replace("_", " ").title()} Shock\n(Aggregated Across Multiple Dates)', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"aggregated_irf_{shock_var.lower().replace(' ', '_')}_shock.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved aggregated IRF plot to {output_dir / filename}")
    
    plt.close()


def plot_aggregated_forecast_comparison(
    forecasts_dict: Dict[str, pd.DataFrame],
    actuals: pd.Series,
    horizons: List[int] = [1, 3, 6],
    variable_name: str = "Variable",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot aggregated forecast comparison: all models and horizons in one plot.
    Lines are color-coded by horizon length.
    
    Parameters:
    -----------
    forecasts_dict : dict
        Dictionary mapping model names to forecast DataFrames
    actuals : pd.Series
        Actual values
    horizons : list
        Forecast horizons
    variable_name : str
        Name of variable
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color map for horizons
    horizon_colors = plt.cm.viridis(np.linspace(0, 1, len(horizons)))
    
    # Plot actuals
    ax.plot(actuals.index, actuals.values, label='Actual', linewidth=2.5, 
           color='black', alpha=0.9, zorder=10)
    
    # Plot forecasts: for each model and each horizon
    linestyles = ['--', '-.', ':']  # Different line styles for different models
    model_styles = {model_name: linestyles[i % len(linestyles)] 
                   for i, model_name in enumerate(forecasts_dict.keys())}
    
    for model_idx, (model_name, forecasts) in enumerate(forecasts_dict.items()):
        for h_idx, h in enumerate(horizons):
            col_name = f'h_{h}'
            if col_name not in forecasts.columns:
                continue
            
            # Prepare data
            plot_data = []
            for forecast_date in forecasts.index:
                target_date = forecast_date + pd.DateOffset(months=h)
                if target_date in actuals.index and pd.notna(forecasts.loc[forecast_date, col_name]):
                    plot_data.append({
                        'date': target_date,
                        'forecast': forecasts.loc[forecast_date, col_name]
                    })
            
            if len(plot_data) == 0:
                continue
            
            plot_df = pd.DataFrame(plot_data).set_index('date').sort_index()
            
            # Label format: Model - h=Xm
            label = f'{model_name} - h={h}m'
            ax.plot(plot_df.index, plot_df['forecast'], 
                   label=label,
                   linewidth=2, 
                   alpha=0.75, 
                   linestyle=model_styles[model_name],
                   color=horizon_colors[h_idx],
                   marker='o',
                   markersize=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{variable_name}', fontsize=12)
    ax.set_title(f'{variable_name} Forecast Comparison: TVP-VAR vs Static VAR (All Horizons)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"aggregated_forecast_comparison_{variable_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved aggregated forecast comparison plot to {output_dir / filename}")
    
    plt.close()


def plot_forecast_performance(
    growth_metrics: pd.DataFrame,
    inflation_metrics: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot forecast performance metrics (RMSE, MAE) for both growth and inflation.
    Shows bars side by side for comparison.
    
    Parameters:
    -----------
    growth_metrics : pd.DataFrame
        Metrics for growth with columns: horizon, rmse, mae
    inflation_metrics : pd.DataFrame
        Metrics for inflation with columns: horizon, rmse, mae
    output_dir : Path, optional
        Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for grouped bar chart
    horizons = sorted(growth_metrics['horizon'].unique())
    x = np.arange(len(horizons))
    width = 0.35  # Width of bars
    
    # RMSE plot
    growth_rmse = [growth_metrics[growth_metrics['horizon'] == h]['rmse'].values[0] 
                   if len(growth_metrics[growth_metrics['horizon'] == h]) > 0 else 0 
                   for h in horizons]
    inflation_rmse = [inflation_metrics[inflation_metrics['horizon'] == h]['rmse'].values[0] 
                      if len(inflation_metrics[inflation_metrics['horizon'] == h]) > 0 else 0 
                      for h in horizons]
    
    bars1 = ax1.bar(x - width/2, growth_rmse, width, label='Growth', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, inflation_rmse, width, label='Inflation', alpha=0.8, color='coral')
    
    ax1.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE by Horizon: Growth vs Inflation', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{h}m' for h in horizons])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # MAE plot
    growth_mae = [growth_metrics[growth_metrics['horizon'] == h]['mae'].values[0] 
                  if len(growth_metrics[growth_metrics['horizon'] == h]) > 0 else 0 
                  for h in horizons]
    inflation_mae = [inflation_metrics[inflation_metrics['horizon'] == h]['mae'].values[0] 
                     if len(inflation_metrics[inflation_metrics['horizon'] == h]) > 0 else 0 
                     for h in horizons]
    
    bars3 = ax2.bar(x - width/2, growth_mae, width, label='Growth', alpha=0.8, color='steelblue')
    bars4 = ax2.bar(x + width/2, inflation_mae, width, label='Inflation', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('MAE by Horizon: Growth vs Inflation', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{h}m' for h in horizons])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "forecast_performance_comparison.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance plot to {output_dir / filename}")
    
    plt.close()


def plot_forecast_comparison(
    forecasts_dict: Dict[str, pd.DataFrame],
    actuals: pd.Series,
    horizons: List[int] = [1, 3, 6],
    variable_name: str = "Variable",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot forecasts from multiple models for comparison.
    
    Parameters:
    -----------
    forecasts_dict : dict
        Dictionary mapping model names to forecast DataFrames
    actuals : pd.Series
        Actual values
    horizons : list
        Forecast horizons
    variable_name : str
        Name of variable
    output_dir : Path, optional
        Directory to save plot
    """
    for h in horizons:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot actuals
        ax.plot(actuals.index, actuals.values, label='Actual', linewidth=2.5, 
               color='black', alpha=0.8)
        
        # Plot forecasts from each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts_dict)))
        for (model_name, forecasts), color in zip(forecasts_dict.items(), colors):
            col_name = f'h_{h}'
            if col_name not in forecasts.columns:
                continue
            
            # Prepare data
            plot_data = []
            for forecast_date in forecasts.index:
                target_date = forecast_date + pd.DateOffset(months=h)
                if target_date in actuals.index and pd.notna(forecasts.loc[forecast_date, col_name]):
                    plot_data.append({
                        'date': target_date,
                        'forecast': forecasts.loc[forecast_date, col_name]
                    })
            
            if len(plot_data) == 0:
                continue
            
            plot_df = pd.DataFrame(plot_data).set_index('date').sort_index()
            ax.plot(plot_df.index, plot_df['forecast'], label=model_name, 
                   linewidth=1.5, alpha=0.7, linestyle='--', color=color)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{variable_name}', fontsize=12)
        ax.set_title(f'{variable_name} Forecast Comparison (Horizon h = {h} months)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"forecast_comparison_{variable_name.lower().replace(' ', '_')}_h{h}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {output_dir / filename}")
        
        plt.close()

