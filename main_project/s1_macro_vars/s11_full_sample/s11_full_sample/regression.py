"""
Regression functions for full-sample regression analysis.

Functions:
- run_regression: Run OLS regression and compute diagnostics
- create_forward_erp: Create forward-looking ERP for horizon h
- run_full_sample_regressions: Run regressions for multiple horizons
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats


def run_regression(y, X, variable_names):
    """
    Run OLS regression and compute diagnostics.
    
    Parameters:
    -----------
    y : np.array
        Dependent variable
    X : np.array
        Independent variables (already standardized)
    variable_names : list
        Names of independent variables
    
    Returns:
    --------
    dict
        Dictionary with regression results
    """
    n = len(y)
    p = X.shape[1]
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Standard errors and t-stats
    X_with_intercept = np.column_stack([np.ones(n), X])
    residuals = y - y_pred
    df_resid = n - p - 1
    sse = np.sum(residuals ** 2)
    mse = sse / df_resid if df_resid > 0 else 0
    
    # Variance-covariance matrix
    try:
        XTX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        var_b = mse * XTX_inv.diagonal()
        std_errors = np.sqrt(var_b)[1:]  # Exclude intercept
    except np.linalg.LinAlgError:
        std_errors = np.full(p, np.nan)
    
    # Coefficients, t-stats, p-values
    coefs = model.coef_
    t_stats = coefs / std_errors if np.all(std_errors > 0) else np.full(p, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid)) if df_resid > 0 else np.full(p, np.nan)
    
    return {
        'intercept': model.intercept_,
        'coefficients': coefs,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'n_obs': n,
        'variable_names': variable_names
    }


def create_forward_erp(erp_series, horizon):
    """
    Create forward-looking ERP for horizon h.
    
    Parameters:
    -----------
    erp_series : pd.Series
        ERP series
    horizon : int
        Horizon in months
    
    Returns:
    --------
    pd.Series
        Forward ERP series
    """
    return erp_series.shift(-horizon)


def run_full_sample_regressions(df, horizons=[1, 3, 6, 12, 24]):
    """
    Run full-sample regressions for multiple horizons.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with ERP and predictors
    horizons : list
        List of forecast horizons (in months)
    
    Returns:
    --------
    dict
        Dictionary with results for each horizon
    """
    # Define macro variables
    macro_vars = [
        'inflation_factor',
        'growth_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    # Check which variables are available
    available_vars = [v for v in macro_vars if v in df.columns]
    if len(available_vars) < len(macro_vars):
        raise ValueError(f"Missing macro variables. Available: {df.columns.tolist()}")
    
    results = {}
    
    for h in horizons:
        print(f"\n{'='*60}")
        print(f"Running regression for horizon h = {h} months")
        print(f"{'='*60}")
        
        # Create forward ERP
        erp_forward = create_forward_erp(df['ERP'], h)
        
        # Align data (drop NaN from forward ERP)
        valid_idx = ~erp_forward.isna()
        y = erp_forward[valid_idx].values
        X_df = df.loc[valid_idx, available_vars]
        X = X_df.values
        
        if len(y) == 0:
            print(f"Warning: No valid observations for horizon {h}")
            continue
        
        # Standardize predictors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run regression
        reg_results = run_regression(y, X_scaled, available_vars)
        
        # Store results
        results[h] = {
            'results': reg_results,
            'scaler': scaler,
            'n_valid': len(y)
        }
        
        # Print summary
        print(f"\nR² = {reg_results['r_squared']:.4f}")
        print(f"N = {reg_results['n_obs']}")
        print(f"\n{'Variable':<25} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10}")
        print("-" * 75)
        for var, coef, se, t, p in zip(
            reg_results['variable_names'],
            reg_results['coefficients'],
            reg_results['std_errors'],
            reg_results['t_stats'],
            reg_results['p_values']
        ):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{var:<25} {coef:>12.6f} {se:>12.6f} {t:>10.2f} {p:>10.4f} {sig}")
        print(f"{'Intercept':<25} {reg_results['intercept']:>12.6f}")
    
    return results

