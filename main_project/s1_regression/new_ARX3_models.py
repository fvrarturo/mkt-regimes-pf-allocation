"""
Model training and evaluation for ARX(3) model.

This module handles:
- Linear Regression (OLS)
- LASSO regression
- Model evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def summary_metrics(y_true, y_pred, label=""):
    """
    Calculate and print summary metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    label : str
        Label for printing
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label}   R^2={r2:.4f},  RMSE={rmse:.4f},  MAE={mae:.4f}")


def train_linear_model(X_train, y_train, X_val, y_val, X_test, y_test, scaler=None):
    """
    Train and evaluate Linear Regression (OLS) model.
    
    Parameters
    ----------
    X_train, X_val, X_test : array-like
        Training, validation, and test features
    y_train, y_val, y_test : array-like
        Training, validation, and test targets
    scaler : StandardScaler or None
        If None, creates a new scaler. Otherwise uses provided scaler.
    
    Returns
    -------
    dict
        Dictionary with model, scaler, and predictions
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
    else:
        X_train_sc = scaler.transform(X_train)
    
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    
    # Train model
    lin_model = LinearRegression()
    lin_model.fit(X_train_sc, y_train)
    
    # Predictions
    y_hat_train = lin_model.predict(X_train_sc)
    y_hat_val = lin_model.predict(X_val_sc)
    y_hat_test = lin_model.predict(X_test_sc)
    
    print("=== ARX(3) Linear Regression – Performance ===")
    summary_metrics(y_train, y_hat_train, "Train")
    summary_metrics(y_val, y_hat_val, "Val")
    summary_metrics(y_test, y_hat_test, "Test")
    
    return {
        "model": lin_model,
        "scaler": scaler,
        "y_hat_train": y_hat_train,
        "y_hat_val": y_hat_val,
        "y_hat_test": y_hat_test,
    }


def train_lasso_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                     min_features=6, alphas=None, tol=1e-6):
    """
    Train and evaluate LASSO model with automatic alpha selection.
    
    Parameters
    ----------
    X_train, X_val, X_test : array-like
        Training, validation, and test features
    y_train, y_val, y_test : array-like
        Training, validation, and test targets
    min_features : int
        Minimum number of nonzero coefficients required
    alphas : array-like or None
        Alpha values to try. If None, uses logspace(-3, 0, 40)
    tol : float
        Tolerance for considering coefficient as nonzero
    
    Returns
    -------
    dict
        Dictionary with model, scaler, predictions, and best alpha
    """
    if alphas is None:
        alphas = np.logspace(-3, 0, 40)  # from 0.001 to 1
    
    best_alpha = None
    best_k = None
    best_pipe = None
    
    for a in sorted(alphas, reverse=True):  # try larger penalties first
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=a, max_iter=50000, random_state=0))
        ])
        pipe.fit(X_train, y_train)
        coefs = pipe.named_steps["lasso"].coef_
        k = np.sum(np.abs(coefs) > tol)
        
        if k >= min_features:  # at least min_features nonzero predictors
            best_alpha = a
            best_k = k
            best_pipe = pipe
            break
    
    if best_pipe is None:
        # fallback: weakest penalty
        a = min(alphas)
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=a, max_iter=50000, random_state=0))
        ])
        best_pipe.fit(X_train, y_train)
        coefs = best_pipe.named_steps["lasso"].coef_
        best_alpha = a
        best_k = np.sum(np.abs(coefs) > tol)
    
    print("\n=== LASSO ARX(3) ===")
    print("alpha:", best_alpha)
    print("nonzero coefficients:", best_k)
    
    # Evaluate LASSO
    y_hat_train_L = best_pipe.predict(X_train)
    y_hat_val_L = best_pipe.predict(X_val)
    y_hat_test_L = best_pipe.predict(X_test)
    
    summary_metrics(y_train, y_hat_train_L, "Train (LASSO)")
    summary_metrics(y_val, y_hat_val_L, "Val   (LASSO)")
    summary_metrics(y_test, y_hat_test_L, "Test  (LASSO)")
    
    return {
        "model": best_pipe,
        "alpha": best_alpha,
        "n_features": best_k,
        "y_hat_train": y_hat_train_L,
        "y_hat_val": y_hat_val_L,
        "y_hat_test": y_hat_test_L,
    }

