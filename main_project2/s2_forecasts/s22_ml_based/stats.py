"""
Statistical analysis module for XGBoost forecast evaluation.

Functions:
- compute_forecast_metrics: Compute RMSE, MAE for forecasts
- diebold_mariano_test: Diebold-Mariano test for forecast comparison
- compare_models: Compare macro-only vs macro+sentiment models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


def compute_forecast_metrics(
    forecasts: pd.Series,
    actuals: pd.Series,
    horizon: int
) -> Dict[str, float]:
    """
    Compute forecast accuracy metrics.
    
    Parameters:
    -----------
    forecasts : pd.Series
        Forecasts indexed by forecast date
    actuals : pd.Series
        Actual values indexed by date
    horizon : int
        Forecast horizon in months
    
    Returns:
    --------
    dict
        Dictionary with RMSE, MAE, and other metrics
    """
    # Align forecasts with actuals
    errors = []
    
    for forecast_date in forecasts.index:
        target_date = forecast_date + pd.DateOffset(months=horizon)
        
        if target_date in actuals.index and pd.notna(forecasts.loc[forecast_date]):
            forecast_val = forecasts.loc[forecast_date]
            actual_val = actuals.loc[target_date]
            errors.append(forecast_val - actual_val)
    
    if len(errors) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'n_forecasts': 0}
    
    errors = np.array(errors)
    
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'n_forecasts': len(errors)
    }


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    power: int = 2
) -> Dict[str, float]:
    """
    Diebold-Mariano test for forecast accuracy comparison.
    
    Tests H0: E[L(e1)] = E[L(e2)] vs H1: E[L(e1)] != E[L(e2)]
    where L is the loss function (power=2 for MSE, power=1 for MAE).
    
    Parameters:
    -----------
    errors1 : np.ndarray
        Forecast errors from model 1
    errors2 : np.ndarray
        Forecast errors from model 2
    h : int
        Forecast horizon (for adjusting variance)
    power : int
        Power of loss function (1 for MAE, 2 for MSE)
    
    Returns:
    --------
    dict
        Dictionary with test statistic, p-value, and other statistics
    """
    if len(errors1) != len(errors2):
        raise ValueError("Error arrays must have same length")
    
    # Compute loss differential
    if power == 1:
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    elif power == 2:
        loss1 = errors1 ** 2
        loss2 = errors2 ** 2
    else:
        loss1 = np.abs(errors1) ** power
        loss2 = np.abs(errors2) ** power
    
    d = loss1 - loss2
    d_bar = np.mean(d)
    
    # Sample variance (with Newey-West adjustment for serial correlation)
    n = len(d)
    
    # Compute autocovariances
    gamma = []
    for k in range(h):
        if k == 0:
            gamma_k = np.var(d)
        else:
            gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma.append(gamma_k)
    
    # Newey-West variance estimator
    var_d = gamma[0] + 2 * sum(gamma[1:])
    se_d = np.sqrt(var_d / n) if var_d > 0 else 0
    
    # Test statistic
    if se_d > 0:
        dm_stat = d_bar / se_d
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        dm_stat = 0.0
        p_value = 1.0
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_bar,
        'se_loss_diff': se_d,
        'n_obs': n
    }


def compare_models(
    forecasts1: pd.Series,
    forecasts2: pd.Series,
    actuals: pd.Series,
    horizon: int,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, float]:
    """
    Compare two forecast models using Diebold-Mariano test.
    
    Parameters:
    -----------
    forecasts1 : pd.Series
        Forecasts from model 1
    forecasts2 : pd.Series
        Forecasts from model 2
    actuals : pd.Series
        Actual values
    horizon : int
        Forecast horizon
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2
    
    Returns:
    --------
    dict
        Comparison results with DM test statistics
    """
    # Align forecasts with actuals
    errors1 = []
    errors2 = []
    
    for forecast_date in forecasts1.index:
        if forecast_date not in forecasts2.index:
            continue
        
        target_date = forecast_date + pd.DateOffset(months=horizon)
        
        if (target_date in actuals.index and
            pd.notna(forecasts1.loc[forecast_date]) and
            pd.notna(forecasts2.loc[forecast_date])):
            
            actual_val = actuals.loc[target_date]
            errors1.append(forecasts1.loc[forecast_date] - actual_val)
            errors2.append(forecasts2.loc[forecast_date] - actual_val)
    
    if len(errors1) < 10:
        return {
            'dm_statistic': np.nan,
            'p_value': np.nan,
            'mean_loss_diff': np.nan,
            'n_obs': len(errors1)
        }
    
    errors1 = np.array(errors1)
    errors2 = np.array(errors2)
    
    # Run DM test
    dm_result = diebold_mariano_test(errors1, errors2, h=horizon, power=2)
    
    return dm_result

