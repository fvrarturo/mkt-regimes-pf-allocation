"""
Feature engineering for ARX(3) model.

This module handles building ARX(3) features with lagged predictors.
"""

import pandas as pd
import numpy as np


def build_arx_features(df, target_col="erp_next", lags=[0, 1, 2]):
    """
    Build ARX(3) design matrix with lagged features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with predictors and target
    target_col : str
        Name of target column
    lags : list of int
        Lags to include (e.g., [0, 1, 2] for ARX(3))
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ARX features, target, and next-month returns
    """
    # X_base: all "information set" variables at month t
    exclude_for_X = ["erp_next", "sp500_ret_next", "rf_ret_next"]
    X_base = df.drop(columns=exclude_for_X)
    y = df[target_col]
    
    # Build lags for ARX(3): [X_t, X_{t-1}, X_{t-2}]
    X_lagged_list = []
    
    for L in lags:
        tmp = X_base.shift(L).add_suffix(f"_L{L+1}")
        X_lagged_list.append(tmp)
    
    X_arx = pd.concat(X_lagged_list, axis=1)
    
    # Retain next-month returns for trading
    ret_next = df[["sp500_ret_next", "rf_ret_next"]]
    
    # Combine into one big DataFrame and drop NaNs (from lags / shifts)
    arx_df = pd.concat([X_arx, y, ret_next], axis=1).dropna()
    
    return arx_df


def split_train_val_test(arx_df, train_end="2012-12", val_end="2016-12"):
    """
    Split data into chronological train/validation/test sets.
    
    Parameters
    ----------
    arx_df : pd.DataFrame
        ARX dataframe with features, target, and returns
    train_end : str
        End date for training set (inclusive)
    val_end : str
        End date for validation set (inclusive)
    
    Returns
    -------
    dict
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test,
        ret_next_train, ret_next_val, ret_next_test
    """
    idx = arx_df.index
    
    train_mask = idx <= train_end
    val_mask = (idx > train_end) & (idx <= val_end)
    test_mask = idx > val_end
    
    train_df = arx_df.loc[train_mask]
    val_df = arx_df.loc[val_mask]
    test_df = arx_df.loc[test_mask]
    
    # Split into X / y / returns
    feature_cols = [c for c in arx_df.columns 
                    if c not in ["erp_next", "sp500_ret_next", "rf_ret_next"]]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df["erp_next"].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df["erp_next"].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df["erp_next"].copy()
    
    ret_next_train = train_df[["sp500_ret_next", "rf_ret_next"]].copy()
    ret_next_val = val_df[["sp500_ret_next", "rf_ret_next"]].copy()
    ret_next_test = test_df[["sp500_ret_next", "rf_ret_next"]].copy()
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "ret_next_train": ret_next_train,
        "ret_next_val": ret_next_val,
        "ret_next_test": ret_next_test,
    }

