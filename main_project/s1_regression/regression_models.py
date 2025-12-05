"""
Model training functions for quarterly regression analysis.

This module handles OLS, LASSO, and out-of-sample forecasting.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error


def fit_multivariate_ols(y, X):
    """
    Fit multivariate OLS regression.
    
    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables
    
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model


def fit_univariate_ols(y, X_var):
    """
    Fit univariate OLS regression for each variable.
    
    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X_var : pd.Series
        Single independent variable
    
    Returns
    -------
    dict
        Dictionary with regression results
    """
    tmp = pd.concat([y, X_var], axis=1).dropna()
    if tmp.shape[0] < 10:
        return None
    
    y_s = tmp.iloc[:, 0]
    X_s = sm.add_constant(tmp.iloc[:, 1])
    res = sm.OLS(y_s, X_s).fit()
    
    return {
        "coef": res.params.iloc[1],
        "t_stat": res.tvalues.iloc[1],
        "p_value": res.pvalues.iloc[1],
        "R_squared": res.rsquared,
        "n_obs": int(res.nobs),
        "sign": "positive" if res.params.iloc[1] > 0 else "negative"
    }


def fit_lasso_selection(X, y, min_features=6, max_features=10, alphas=None, tol=1e-6):
    """
    Fit LASSO model with automatic feature selection.
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
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
        Dictionary with model, selected variables, and alpha
    """
    if alphas is None:
        alphas = np.logspace(-3, 0, 100)
    
    best_alpha = None
    best_coefs = None
    best_k = None
    best_pipe = None
    
    for a in sorted(alphas, reverse=True):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=a, max_iter=50000, random_state=0))
        ])
        pipe.fit(X, y)
        coefs = pipe.named_steps["lasso"].coef_
        k = np.sum(np.abs(coefs) > tol)
        
        if min_features <= k <= max_features:
            best_alpha = a
            best_coefs = coefs
            best_k = k
            best_pipe = pipe
            break
    
    if best_alpha is None:
        # Fallback: smallest alpha
        a = min(alphas)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=a, max_iter=50000, random_state=0))
        ])
        pipe.fit(X, y)
        best_alpha = a
        best_coefs = pipe.named_steps["lasso"].coef_
        best_k = np.sum(np.abs(best_coefs) > tol)
        best_pipe = pipe
    
    # Get selected variable names
    if isinstance(X, pd.DataFrame):
        selected_vars = [X.columns[i] for i, c in enumerate(best_coefs) if abs(c) > tol]
    else:
        selected_vars = None
    
    return {
        "model": best_pipe,
        "alpha": best_alpha,
        "n_features": best_k,
        "coefs": best_coefs,
        "selected_vars": selected_vars,
    }


def expanding_window_forecast(X, y, split_date, selected_vars=None):
    """
    Perform expanding window out-of-sample forecasting.
    
    Parameters
    ----------
    X : pd.DataFrame
        Full feature set
    y : pd.Series
        Full target series
    split_date : str or pd.Timestamp
        Split date for train/test
    selected_vars : list or None
        Selected variables (if None, uses all)
    
    Returns
    -------
    pd.Series
        Out-of-sample predictions
    """
    if selected_vars is not None:
        X_sel = X[selected_vars]
    else:
        X_sel = X
    
    train_idx = X_sel.index <= split_date
    test_idx = X_sel.index > split_date
    
    X_train = X_sel.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X_sel.loc[test_idx]
    y_test = y.loc[test_idx]
    
    n0 = len(X_train)
    y_pred = []
    test_points = X_test.index
    
    for t in range(len(test_points)):
        model = LinearRegression()
        
        # Expanding window
        X_train_t = X_sel.iloc[:(n0 + t)]
        y_train_t = y.iloc[:(n0 + t)]
        
        model.fit(X_train_t, y_train_t)
        
        # Predict
        x_t = X_test.iloc[[t]]
        y_pred.append(model.predict(x_t)[0])
    
    return pd.Series(y_pred, index=test_points)

