"""
Plotting module for LSTM results visualization.

Functions:
- plot_learning_curves: Plot training/validation loss curves
- plot_forecast_vs_realized: Plot forecasts against actual values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_learning_curves(
    history: Dict,
    horizon: int,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot learning curves (training and validation loss).
    
    Parameters:
    -----------
    history : dict
        Training history from Keras model
    horizon : int
        Forecast horizon
    output_dir : Path, optional
        Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['loss'], label='Training Loss', linewidth=2, alpha=0.8)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'Model Loss: Horizon h={horizon}m', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # MAE plot
    if 'mae' in history:
        ax2.plot(epochs, history['mae'], label='Training MAE', linewidth=2, alpha=0.8)
        if 'val_mae' in history:
            ax2.plot(epochs, history['val_mae'], label='Validation MAE', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title(f'Model MAE: Horizon h={horizon}m', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"learning_curve_lstm_h{horizon}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved learning curve plot to {output_dir / filename}")
    
    plt.close()


def plot_forecast_vs_realized_lstm(
    forecasts: Dict[str, Dict[int, pd.Series]],
    actuals: Dict[str, Dict[int, pd.Series]],
    horizons: List[int] = [1, 3, 6],
    start_date: Optional[str] = "2008-01-01",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot LSTM forecasts against actual values for all horizons.
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary mapping variable names to dict of {horizon: forecast_series}
    actuals : dict
        Dictionary mapping variable names to dict of {horizon: actual_series}
    horizons : list
        Forecast horizons
    start_date : str, optional
        Start date for plotting
    output_dir : Path, optional
        Directory to save plot
    """
    for var_name in forecasts.keys():
        if var_name not in actuals:
            continue
        
        var_forecasts = forecasts[var_name]
        var_actuals = actuals[var_name]
        
        # Create figure with subplots for each horizon
        n_horizons = len(horizons)
        fig, axes = plt.subplots(n_horizons, 1, figsize=(16, 5 * n_horizons))
        
        if n_horizons == 1:
            axes = [axes]
        
        # Color map for horizons
        horizon_colors = plt.cm.viridis(np.linspace(0, 1, len(horizons)))
        
        for idx, h in enumerate(horizons):
            ax = axes[idx]
            
            if h not in var_forecasts or h not in var_actuals:
                continue
            
            forecast_series = var_forecasts[h]
            actual_series = var_actuals[h]
            
            # Align forecasts with actuals (forecasts are at origin dates, actuals at target dates)
            plot_data = []
            for forecast_date in forecast_series.index:
                target_date = forecast_date + pd.DateOffset(months=h)
                if target_date in actual_series.index:
                    plot_data.append({
                        'date': target_date,
                        'forecast': forecast_series.loc[forecast_date],
                        'actual': actual_series.loc[target_date]
                    })
            
            if len(plot_data) == 0:
                continue
            
            plot_df = pd.DataFrame(plot_data).set_index('date').sort_index()
            
            # Filter by start date
            if start_date:
                start_dt = pd.to_datetime(start_date)
                plot_df = plot_df[plot_df.index >= start_dt]
            
            # Plot
            ax.plot(plot_df.index, plot_df['actual'], label='Actual', 
                   linewidth=2.5, color='black', alpha=0.9, zorder=10)
            ax.plot(plot_df.index, plot_df['forecast'], label=f'LSTM Forecast h={h}m', 
                   linewidth=2, alpha=0.75, linestyle='--', 
                   color=horizon_colors[idx], marker='o', markersize=2)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(f'{var_name.replace("_", " ").title()}', fontsize=12)
            ax.set_title(f'{var_name.replace("_", " ").title()} Forecast vs Actual (Horizon h={h}m)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(alpha=0.3)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"forecast_vs_realized_lstm_{var_name}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved LSTM forecast plot to {output_dir / filename}")
        
        plt.close()

