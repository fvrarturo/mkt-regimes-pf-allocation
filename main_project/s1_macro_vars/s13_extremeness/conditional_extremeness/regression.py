"""
Conditional extremeness regression analysis.

Estimates: ERP_{t+h} = α_r + β_r X_t + γ_r · Ext_t + (δ_r X_t · Ext_t) + ε_t

Focuses on:
- γ_r: extremeness main effect
- δ_r: interaction effect (how extremeness changes macro-ERP relationships)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def estimate_regime_extremeness_regression(df, regime_col='regime', horizon=1):
    """
    Estimate conditional extremeness regression for each regime.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe with regime, extremeness, macro variables, and ERP
    regime_col : str
        Column name for regime assignment
    horizon : int
        Forecast horizon (for future ERP)
    
    Returns:
    --------
    dict
        Dictionary with regression results for each regime
    """
    results = {}
    
    # Macro variables
    macro_vars = ['growth_factor', 'inflation_factor', 'monetary_policy_factor', 'market_volatility_factor']
    macro_vars = [v for v in macro_vars if v in df.columns]
    
    # Get unique regimes
    regimes = sorted(df[regime_col].unique())
    
    for regime in regimes:
        regime_data = df[df[regime_col] == regime].copy()
        
        if len(regime_data) < 10:  # Need minimum observations
            print(f"Warning: Regime {regime} has only {len(regime_data)} observations, skipping")
            continue
        
        # Prepare features
        X_base = regime_data[macro_vars].values
        extreme = regime_data['extreme'].values.reshape(-1, 1)
        
        # Create interaction terms: X * extreme
        X_interactions = X_base * extreme
        
        # Combine: [X, extreme, X*extreme]
        X = np.hstack([X_base, extreme, X_interactions])
        
        # Target: ERP
        y = regime_data['erp'].values
        
        # Remove NaN
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid]
        y_clean = y[valid]
        
        if len(X_clean) < 10:
            continue
        
        # Fit regression
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Predictions
        y_pred = model.predict(X_clean)
        residuals = y_clean - y_pred
        
        # Compute statistics
        n = len(y_clean)
        k = X_clean.shape[1]
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        
        # Standard errors
        XtX_inv = np.linalg.pinv(X_clean.T @ X_clean)
        se = np.sqrt(np.diag(XtX_inv) * mse)
        
        # t-statistics
        t_stats = model.coef_ / se
        
        # p-values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Store results
        feature_names = macro_vars + ['extreme'] + [f'{v}_x_extreme' for v in macro_vars]
        
        results[regime] = {
            'n_obs': n,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'se': se,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'rmse': rmse,
            'feature_names': feature_names,
            'macro_vars': macro_vars
        }
    
    return results


def extract_key_effects(results):
    """
    Extract extremeness main effects (γ_r) and interaction effects (δ_r).
    
    Parameters:
    -----------
    results : dict
        Regression results from estimate_regime_extremeness_regression
    
    Returns:
    --------
    pd.DataFrame
        Table with key effects by regime
    """
    rows = []
    
    for regime, res in results.items():
        feature_names = res['feature_names']
        coefs = res['coefficients']
        p_vals = res['p_values']
        
        # Find extremeness main effect (γ_r)
        extreme_idx = feature_names.index('extreme')
        gamma = coefs[extreme_idx]
        gamma_pval = p_vals[extreme_idx]
        
        # Find interaction effects (δ_r) for each macro variable
        for var in res['macro_vars']:
            interaction_name = f'{var}_x_extreme'
            if interaction_name in feature_names:
                var_idx = res['macro_vars'].index(var)
                base_coef = coefs[var_idx]  # β_r (normal state)
                
                interaction_idx = feature_names.index(interaction_name)
                delta = coefs[interaction_idx]  # δ_r (interaction)
                delta_pval = p_vals[interaction_idx]
                
                # Combined effect in extreme state: β_r + δ_r
                extreme_coef = base_coef + delta
                
                rows.append({
                    'regime': regime,
                    'variable': var,
                    'beta_normal': base_coef,
                    'gamma_extreme': gamma,
                    'gamma_pvalue': gamma_pval,
                    'delta_interaction': delta,
                    'delta_pvalue': delta_pval,
                    'beta_extreme': extreme_coef,
                    'n_obs': res['n_obs'],
                    'r_squared': res['r_squared']
                })
    
    return pd.DataFrame(rows)


def compute_marginal_effects(results, regime):
    """
    Compute marginal effects of macro variables in normal vs extreme states.
    
    Parameters:
    -----------
    results : dict
        Regression results
    regime : int
        Regime number
    
    Returns:
    --------
    pd.DataFrame
        Marginal effects table
    """
    if regime not in results:
        return pd.DataFrame()
    
    res = results[regime]
    feature_names = res['feature_names']
    coefs = res['coefficients']
    
    rows = []
    for var in res['macro_vars']:
        var_idx = res['macro_vars'].index(var)
        beta_normal = coefs[var_idx]
        
        interaction_name = f'{var}_x_extreme'
        if interaction_name in feature_names:
            interaction_idx = feature_names.index(interaction_name)
            delta = coefs[interaction_idx]
            beta_extreme = beta_normal + delta
        else:
            delta = 0
            beta_extreme = beta_normal
        
        rows.append({
            'variable': var,
            'effect_normal': beta_normal,
            'effect_extreme': beta_extreme,
            'difference': delta
        })
    
    return pd.DataFrame(rows)

