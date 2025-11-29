"""
Statistical analysis module for forecast evaluation.

Functions:
- compute_forecast_metrics: Compute RMSE, MAE for forecasts
- diebold_mariano_test: Diebold-Mariano test for forecast comparison
- compare_static_var: Compare TVP-VAR to static VAR
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR


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
        Metrics table with columns: horizon, rmse, mae
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
        # MAE
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    elif power == 2:
        # MSE
        loss1 = errors1 ** 2
        loss2 = errors2 ** 2
    else:
        loss1 = np.abs(errors1) ** power
        loss2 = np.abs(errors2) ** power
    
    d = loss1 - loss2
    
    # Sample mean
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
    
    # Standard error
    se_d = np.sqrt(var_d / n) if var_d > 0 else 0
    
    # Test statistic
    if se_d > 0:
        dm_stat = d_bar / se_d
        # Two-sided p-value
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


def compare_forecasts(
    forecasts1: pd.DataFrame,
    forecasts2: pd.DataFrame,
    actuals: pd.Series,
    horizons: list = [1, 3, 6],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> pd.DataFrame:
    """
    Compare two forecast models using Diebold-Mariano tests.
    
    Parameters:
    -----------
    forecasts1 : pd.DataFrame
        Forecasts from model 1
    forecasts2 : pd.DataFrame
        Forecasts from model 2
    actuals : pd.Series
        Actual values
    horizons : list
        Forecast horizons
    model1_name : str
        Name of model 1
    model2_name : str
        Name of model 2
    
    Returns:
    --------
    pd.DataFrame
        Comparison results with DM test statistics and p-values
    """
    results = []
    
    for h in horizons:
        col_name = f'h_{h}'
        if col_name not in forecasts1.columns or col_name not in forecasts2.columns:
            continue
        
        # Align forecasts with actuals
        errors1 = []
        errors2 = []
        
        for forecast_date in forecasts1.index:
            if forecast_date not in forecasts2.index:
                continue
            
            target_date = forecast_date + pd.DateOffset(months=h)
            
            if (target_date in actuals.index and
                pd.notna(forecasts1.loc[forecast_date, col_name]) and
                pd.notna(forecasts2.loc[forecast_date, col_name])):
                
                actual_val = actuals.loc[target_date]
                errors1.append(forecasts1.loc[forecast_date, col_name] - actual_val)
                errors2.append(forecasts2.loc[forecast_date, col_name] - actual_val)
        
        if len(errors1) < 10:  # Need minimum observations
            continue
        
        errors1 = np.array(errors1)
        errors2 = np.array(errors2)
        
        # Run DM test (MSE-based)
        dm_result = diebold_mariano_test(errors1, errors2, h=h, power=2)
        
        results.append({
            'horizon': h,
            'model1': model1_name,
            'model2': model2_name,
            'dm_statistic': dm_result['dm_statistic'],
            'p_value': dm_result['p_value'],
            'mean_loss_diff': dm_result['mean_loss_diff'],
            'n_obs': dm_result['n_obs']
        })
    
    return pd.DataFrame(results)


def fit_static_var(
    data: pd.DataFrame,
    lag_order: int
) -> VAR:
    """
    Fit a static VAR model (for comparison with TVP-VAR).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    lag_order : int
        Lag order
    
    Returns:
    --------
    VAR
        Fitted VAR model
    """
    var_model = VAR(data)
    fitted_model = var_model.fit(maxlags=lag_order, ic=None)
    return fitted_model

