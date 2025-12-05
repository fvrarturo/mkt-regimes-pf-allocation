"""
Model training functions for 3-month rolling regression.

This module handles rolling window forecasting and LASSO selection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rolling_forecast(y_series, X_frame, window):
    """
    Rolling-window 1-step-ahead forecasts.
    
    Parameters
    ----------
    y_series : pd.Series
        Target series
    X_frame : pd.DataFrame
        Feature dataframe
    window : int
        Rolling window size
    
    Returns
    -------
    pd.Series
        Forecast series
    """
    y_vals = y_series.values
    X_vals = X_frame.values
    idx = y_series.index
    
    preds = []
    pred_idx = []
    
    for i in range(window, len(y_series)):
        # Training window: i-window ... i-1
        X_train = X_vals[i-window:i]
        y_train = y_vals[i-window:i]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict y at time i
        y_pred = model.predict(X_vals[i].reshape(1, -1))[0]
        preds.append(y_pred)
        pred_idx.append(idx[i])
    
    return pd.Series(preds, index=pred_idx)


def oos_metrics(y_true, y_pred, label=""):
    """
    Calculate out-of-sample metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    label : str
        Label for printing
    
    Returns
    -------
    tuple
        (rmse, mae, r2)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    sse = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - y_true.mean())**2)
    r2_oos = 1 - sse / sst
    
    print(f"\n=== OOS METRICS: {label} ===")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R^2  : {r2_oos:.4f}")
    
    return rmse, mae, r2_oos


def fit_lasso_selection(X, y, min_features=6, max_features=10, alphas=None, tol=1e-6):
    """
    Fit LASSO model with automatic feature selection.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    min_features : int
        Minimum number of features
    max_features : int
        Maximum number of features
    alphas : array-like or None
        Alpha values to try
    tol : float
        Tolerance for nonzero coefficients
    
    Returns
    -------
    dict
        Dictionary with selected variables and model info
    """
    if alphas is None:
        alphas = np.logspace(-3, 0, 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    best_alpha = None
    best_coef = None
    best_k = None
    
    for a in sorted(alphas, reverse=True):
        lasso = Lasso(alpha=a, max_iter=50000, random_state=0)
        lasso.fit(X_scaled, y.values)
        coefs = lasso.coef_
        k = np.sum(np.abs(coefs) > tol)
        
        if min_features <= k <= max_features:
            best_alpha = a
            best_coef = coefs
            best_k = k
            break
    
    # Fallback: smallest alpha
    if best_alpha is None:
        a = min(alphas)
        lasso = Lasso(alpha=a, max_iter=50000, random_state=0)
        lasso.fit(X_scaled, y.values)
        best_alpha = a
        best_coef = lasso.coef_
        best_k = np.sum(np.abs(best_coef) > tol)
    
    selected_vars = [name for name, c in zip(X.columns, best_coef) if abs(c) > tol]
    
    return {
        "alpha": best_alpha,
        "n_features": best_k,
        "selected_vars": selected_vars,
    }

