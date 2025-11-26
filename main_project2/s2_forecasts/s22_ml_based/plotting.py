"""
Plotting module for XGBoost results visualization.

Functions:
- plot_feature_importance: Plot feature importance
- plot_forecast_comparison: Compare macro vs macro+sentiment forecasts
- plot_shap_values: Plot SHAP values (if available)
- plot_partial_dependence: Plot partial dependence plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_feature_importance(
    importance_df: pd.DataFrame,
    variable: str,
    horizon: int,
    top_n: int = 20,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance DataFrame with columns 'feature' and 'importance'
    variable : str
        Variable name
    horizon : int
        Forecast horizon
    top_n : int
        Number of top features to plot
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'].values, alpha=0.7, color='steelblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Feature Importance: {variable.capitalize()} (Horizon h={horizon}m)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"feature_importance_{variable}_h{horizon}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {output_dir / filename}")
    
    plt.close()


def plot_forecast_comparison(
    forecasts_macro: pd.Series,
    forecasts_sentiment: pd.Series,
    actuals: pd.Series,
    variable: str,
    horizon: int,
    start_date: Optional[str] = "2008-01-01",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot forecast comparison: macro-only vs macro+sentiment.
    
    Parameters:
    -----------
    forecasts_macro : pd.Series
        Forecasts from macro-only model
    forecasts_sentiment : pd.Series
        Forecasts from macro+sentiment model
    actuals : pd.Series
        Actual values
    variable : str
        Variable name
    horizon : int
        Forecast horizon
    start_date : str, optional
        Start date for plotting
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Filter by start date
    if start_date:
        start_dt = pd.to_datetime(start_date)
        actuals_filtered = actuals[actuals.index >= start_dt]
    else:
        actuals_filtered = actuals
    
    # Plot actuals
    ax.plot(actuals_filtered.index, actuals_filtered.values, 
           label='Actual', linewidth=2.5, color='black', alpha=0.9, zorder=10)
    
    # Prepare forecast data
    plot_data_macro = []
    plot_data_sentiment = []
    
    for forecast_date in forecasts_macro.index:
        if forecast_date not in forecasts_sentiment.index:
            continue
        
        target_date = forecast_date + pd.DateOffset(months=horizon)
        
        if start_date and target_date < pd.to_datetime(start_date):
            continue
        
        if target_date in actuals.index:
            if pd.notna(forecasts_macro.loc[forecast_date]):
                plot_data_macro.append({
                    'date': target_date,
                    'forecast': forecasts_macro.loc[forecast_date]
                })
            if pd.notna(forecasts_sentiment.loc[forecast_date]):
                plot_data_sentiment.append({
                    'date': target_date,
                    'forecast': forecasts_sentiment.loc[forecast_date]
                })
    
    if len(plot_data_macro) > 0:
        df_macro = pd.DataFrame(plot_data_macro).set_index('date').sort_index()
        ax.plot(df_macro.index, df_macro['forecast'], 
               label='XGBoost (Macro-only)', linewidth=2, alpha=0.75, 
               linestyle='--', color='steelblue', marker='o', markersize=2)
    
    if len(plot_data_sentiment) > 0:
        df_sentiment = pd.DataFrame(plot_data_sentiment).set_index('date').sort_index()
        ax.plot(df_sentiment.index, df_sentiment['forecast'], 
               label='XGBoost (Macro+Sentiment)', linewidth=2, alpha=0.75, 
               linestyle='--', color='coral', marker='s', markersize=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{variable.capitalize()}', fontsize=12)
    ax.set_title(f'{variable.capitalize()} Forecast Comparison: Macro vs Macro+Sentiment (h={horizon}m)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"forecast_comparison_{variable}_h{horizon}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved forecast comparison plot to {output_dir / filename}")
    
    plt.close()


def plot_rmse_comparison(
    metrics_macro: pd.DataFrame,
    metrics_sentiment: pd.DataFrame,
    variable: str,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot side-by-side RMSE comparison bar chart.
    
    Parameters:
    -----------
    metrics_macro : pd.DataFrame
        Metrics for macro-only model (columns: horizon, rmse, mae)
    metrics_sentiment : pd.DataFrame
        Metrics for macro+sentiment model (columns: horizon, rmse, mae)
    variable : str
        Variable name
    output_dir : Path, optional
        Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    horizons = sorted(metrics_macro['horizon'].unique())
    x = np.arange(len(horizons))
    width = 0.35
    
    # RMSE comparison
    macro_rmse = [metrics_macro[metrics_macro['horizon'] == h]['rmse'].values[0] 
                  if len(metrics_macro[metrics_macro['horizon'] == h]) > 0 else 0 
                  for h in horizons]
    sent_rmse = [metrics_sentiment[metrics_sentiment['horizon'] == h]['rmse'].values[0] 
                if len(metrics_sentiment[metrics_sentiment['horizon'] == h]) > 0 else 0 
                for h in horizons]
    
    bars1 = ax1.bar(x - width/2, macro_rmse, width, label='Macro-only', 
                   alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, sent_rmse, width, label='Macro+Sentiment', 
                   alpha=0.8, color='coral')
    
    ax1.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title(f'RMSE Comparison: {variable.capitalize()}', fontsize=13, fontweight='bold')
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
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MAE comparison
    macro_mae = [metrics_macro[metrics_macro['horizon'] == h]['mae'].values[0] 
                if len(metrics_macro[metrics_macro['horizon'] == h]) > 0 else 0 
                for h in horizons]
    sent_mae = [metrics_sentiment[metrics_sentiment['horizon'] == h]['mae'].values[0] 
               if len(metrics_sentiment[metrics_sentiment['horizon'] == h]) > 0 else 0 
               for h in horizons]
    
    bars3 = ax2.bar(x - width/2, macro_mae, width, label='Macro-only', 
                   alpha=0.8, color='steelblue')
    bars4 = ax2.bar(x + width/2, sent_mae, width, label='Macro+Sentiment', 
                   alpha=0.8, color='coral')
    
    ax2.set_xlabel('Forecast Horizon (months)', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title(f'MAE Comparison: {variable.capitalize()}', fontsize=13, fontweight='bold')
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
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"rmse_mae_comparison_{variable}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved RMSE/MAE comparison plot to {output_dir / filename}")
    
    plt.close()

