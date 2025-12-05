"""
Trading strategies for ARX(3) model predictions.

This module implements:
- Banded weight strategy (Strategy A)
- Sign-based strategy (Strategy B)
- Benchmark strategies
"""

import numpy as np
import pandas as pd


def weight_bands(z):
    """
    Banded weight function based on z-score.
    
    Parameters
    ----------
    z : float
        Z-score of predicted ERP
    
    Returns
    -------
    float
        Weight in [0, 1] for stock allocation
    """
    if z >= 2:
        return 1.00
    elif z >= 1:
        return 0.75
    elif z > -1:
        return 0.50
    elif z > -2:
        return 0.25
    else:
        return 0.00


def weight_sign(erp_pred):
    """
    Sign-based weight function.
    
    Parameters
    ----------
    erp_pred : float
        Predicted ERP
    
    Returns
    -------
    float
        Weight: 1.0 if erp_pred > 0, else 0.0
    """
    return 1.0 if erp_pred > 0 else 0.0


def compute_trading_strategies(y_hat_train, y_hat_test, ret_next_test, 
                               strategy_type="bands"):
    """
    Compute trading strategy returns.
    
    Parameters
    ----------
    y_hat_train : array-like
        Training set predictions (for computing z-score normalization)
    y_hat_test : array-like or pd.Series
        Test set predictions
    ret_next_test : pd.DataFrame
        DataFrame with columns sp500_ret_next and rf_ret_next
    strategy_type : str
        "bands" for Strategy A, "sign" for Strategy B
    
    Returns
    -------
    dict
        Dictionary with strategy returns and weights
    """
    y_hat_test = pd.Series(y_hat_test, index=ret_next_test.index)
    
    if strategy_type == "bands":
        # Z-score normalization based on training distribution
        mean_hat_train = np.mean(y_hat_train)
        std_hat_train = np.std(y_hat_train)
        z_pred_test = (y_hat_test - mean_hat_train) / std_hat_train
        
        w_stock = z_pred_test.apply(weight_bands)
    elif strategy_type == "sign":
        w_stock = y_hat_test.apply(weight_sign)
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")
    
    w_rf = 1 - w_stock
    
    # Realized next-month returns
    sp_next = ret_next_test["sp500_ret_next"]
    rf_next = ret_next_test["rf_ret_next"]
    
    # Portfolio returns
    ret_strategy = w_stock * sp_next + w_rf * rf_next
    
    return {
        "returns": ret_strategy,
        "weights_stock": w_stock,
        "weights_rf": w_rf,
    }


def compute_benchmarks(ret_next_test, avg_weight=None):
    """
    Compute benchmark portfolio returns.
    
    Parameters
    ----------
    ret_next_test : pd.DataFrame
        DataFrame with columns sp500_ret_next and rf_ret_next
    avg_weight : float or None
        Average stock weight for dynamic mix benchmark
    
    Returns
    -------
    dict
        Dictionary with benchmark returns
    """
    sp_next = ret_next_test["sp500_ret_next"]
    rf_next = ret_next_test["rf_ret_next"]
    
    # 1. Buy & Hold (100% S&P)
    ret_bench_100 = sp_next.copy()
    
    # 2. 50/50 static portfolio
    ret_bench_50 = 0.50 * sp_next + 0.50 * rf_next
    
    # 3. Constant mix using average dynamic weight
    if avg_weight is not None:
        ret_bench_avg = avg_weight * sp_next + (1 - avg_weight) * rf_next
    else:
        ret_bench_avg = None
    
    return {
        "bench_100": ret_bench_100,
        "bench_50": ret_bench_50,
        "bench_avg": ret_bench_avg,
    }


def evaluate_strategies(strategy_A, strategy_B, benchmarks):
    """
    Print evaluation metrics for all strategies.
    
    Parameters
    ----------
    strategy_A : dict
        Strategy A results from compute_trading_strategies
    strategy_B : dict
        Strategy B results from compute_trading_strategies
    benchmarks : dict
        Benchmark results from compute_benchmarks
    """
    print("Mean monthly returns (test):")
    print(f"Strategy A:     {strategy_A['returns'].mean():.5f}")
    print(f"Strategy B:     {strategy_B['returns'].mean():.5f}")
    print(f"Benchmark 100%: {benchmarks['bench_100'].mean():.5f}")
    print(f"Benchmark 50/50: {benchmarks['bench_50'].mean():.5f}")
    if benchmarks['bench_avg'] is not None:
        print(f"Benchmark avgW: {benchmarks['bench_avg'].mean():.5f}")

