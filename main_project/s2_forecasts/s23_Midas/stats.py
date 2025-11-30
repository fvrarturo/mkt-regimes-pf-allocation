"""
Statistical analysis module for MIDAS forecast evaluation.

Functions:
- compute_forecast_metrics: Compute RMSE, MAE for forecasts
- compare_forecasts: Compare forecast performance across models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


def compute_forecast_metrics(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    horizons: list = [1, 3, 6]
) -> pd.DataFrame:
    """
    Compute forecast accuracy metrics (RMSE, MAE) for each horizon.
    
    Parameters:
    -----------
    forecasts : pd.DataFrame
        Forecasts with columns h_1, h_3, h_6, etc., indexed by forecast date
    actuals : pd.Series
        Actual values, indexed by date
    horizons : list
        Forecast horizons in months
    
    Returns:
    --------
    pd.DataFrame
        Metrics table with columns: horizon, rmse, mae, n_forecasts
    """
    metrics = []
    
    for h in horizons:
        col_name = f'h_{h}'
        if col_name not in forecasts.columns:
            continue
        
        # Align forecasts with actuals
        # For horizon h, forecast made at date t predicts value at t+h
        forecast_values = []
        actual_values = []
        
        for forecast_date in forecasts.index:
            target_date = forecast_date + pd.DateOffset(months=h)
            
            if target_date in actuals.index and pd.notna(forecasts.loc[forecast_date, col_name]):
                forecast_values.append(forecasts.loc[forecast_date, col_name])
                actual_values.append(actuals.loc[target_date])
        
        if len(forecast_values) == 0:
            continue
        
        forecast_values = np.array(forecast_values)
        actual_values = np.array(actual_values)
        
        # Compute errors
        errors = forecast_values - actual_values
        
        # RMSE
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # MAE
        mae = np.mean(np.abs(errors))
        
        metrics.append({
            'horizon': h,
            'rmse': rmse,
            'mae': mae,
            'n_forecasts': len(forecast_values)
        })
    
    return pd.DataFrame(metrics)


def compare_forecasts(
    forecasts_dict: Dict[str, pd.DataFrame],
    actuals: pd.Series,
    horizons: list = [1, 3, 6]
) -> pd.DataFrame:
    """
    Compare forecast performance across multiple models.
    
    Parameters:
    -----------
    forecasts_dict : dict
        Dictionary of forecast DataFrames, keyed by model name
    actuals : pd.Series
        Actual values
    horizons : list
        Forecast horizons
    
    Returns:
    --------
    pd.DataFrame
        Comparison table with models as rows, metrics as columns
    """
    comparison = []
    
    for model_name, forecasts in forecasts_dict.items():
        metrics = compute_forecast_metrics(forecasts, actuals, horizons=horizons)
        
        for _, row in metrics.iterrows():
            comparison.append({
                'model': model_name,
                'horizon': int(row['horizon']),
                'rmse': row['rmse'],
                'mae': row['mae'],
                'n_forecasts': int(row['n_forecasts'])
            })
    
    return pd.DataFrame(comparison)


def diebold_mariano_test(
    forecast_1: np.ndarray,
    forecast_2: np.ndarray,
    actuals: np.ndarray,
    loss: str = 'mse'
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for forecast comparison.
    
    Parameters:
    -----------
    forecast_1 : np.ndarray
        First forecast
    forecast_2 : np.ndarray
        Second forecast
    actuals : np.ndarray
        Actual values
    loss : str
        Loss function: 'mse' or 'mae'
    
    Returns:
    --------
    tuple
        (dm_statistic, p_value)
    """
    # Compute losses
    error_1 = forecast_1 - actuals
    error_2 = forecast_2 - actuals
    
    if loss == 'mse':
        loss_1 = error_1 ** 2
        loss_2 = error_2 ** 2
    elif loss == 'mae':
        loss_1 = np.abs(error_1)
        loss_2 = np.abs(error_2)
    else:
        raise ValueError(f"Unknown loss function: {loss}")
    
    # DM statistic
    d = loss_1 - loss_2
    mean_d = np.mean(d)
    var_d = np.var(d)
    
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value
