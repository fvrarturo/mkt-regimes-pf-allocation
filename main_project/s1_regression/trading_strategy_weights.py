"""
Trading strategy weight functions.

This module implements weight calculation functions for different strategies.
"""

import pandas as pd
import numpy as np


def stock_weight_from_z(z_val):
    """
    Banded weight function based on z-score.
    
    Parameters
    ----------
    z_val : float
        Z-score value
    
    Returns
    -------
    float
        Stock weight in [0, 1]
    """
    if z_val >= 2:
        return 1.00
    elif z_val >= 1:
        return 0.75
    elif z_val > -1:
        return 0.50
    elif z_val > -2:
        return 0.25
    else:
        return 0.00


def compute_strategy_weights(z_scores, lag=1):
    """
    Compute strategy weights from z-scores with lag.
    
    Parameters
    ----------
    z_scores : pd.Series
        Z-scores
    lag : int
        Number of periods to lag weights
    
    Returns
    -------
    pd.DataFrame
        DataFrame with w_stock and w_cash columns
    """
    w_stock = z_scores.apply(stock_weight_from_z)
    w_cash = 1 - w_stock
    
    # Use last month's z-score to set this month's weights (no look-ahead)
    w_stock = w_stock.shift(lag).fillna(0.5)
    w_cash = w_cash.shift(lag).fillna(0.5)
    
    return pd.DataFrame({
        "w_stock": w_stock,
        "w_cash": w_cash
    })

