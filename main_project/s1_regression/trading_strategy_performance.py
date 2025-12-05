"""
Performance evaluation functions for trading strategies.

This module handles calculation of returns, Sharpe ratios, and cumulative performance.
"""

import pandas as pd
import numpy as np


def geometric_return(r):
    """
    Calculate geometric average return.
    
    Parameters
    ----------
    r : pd.Series
        Return series
    
    Returns
    -------
    float
        Geometric mean return
    """
    r = r.dropna()
    gross = (1 + r).prod()
    return gross ** (1 / len(r)) - 1


def sharpe_annual(excess_ret):
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    excess_ret : pd.Series
        Excess return series (portfolio - risk-free)
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    excess_ret = excess_ret.dropna()
    if excess_ret.std() == 0:
        return np.nan
    return (excess_ret.mean() / excess_ret.std()) * np.sqrt(12)


def evaluate_stock_cash_strategy(strategy_df, cash_ret=0.0):
    """
    Evaluate stock-cash strategy performance.
    
    Parameters
    ----------
    strategy_df : pd.DataFrame
        DataFrame with w_stock, w_cash, sp500_ret, rf_ret columns
    cash_ret : float
        Cash return rate (default 0.0)
    
    Returns
    -------
    dict
        Dictionary with strategy returns and performance metrics
    """
    # Dynamic ERP timing strategy
    ret_dyn = (
        strategy_df["w_stock"] * strategy_df["sp500_ret"] +
        strategy_df["w_cash"] * cash_ret
    )
    
    # Benchmark: constant mix equal to the average risky load
    w_bar = strategy_df["w_stock"].mean()
    ret_bench = (
        w_bar * strategy_df["sp500_ret"] +
        (1 - w_bar) * cash_ret
    )
    
    # Geometric annualized returns
    g_dyn_m = geometric_return(ret_dyn)
    g_bench_m = geometric_return(ret_bench)
    
    g_dyn_a = (1 + g_dyn_m) ** 12 - 1
    g_bench_a = (1 + g_bench_m) ** 12 - 1
    
    # Sharpe ratios
    ex_dyn = ret_dyn - strategy_df["rf_ret"]
    ex_bench = ret_bench - strategy_df["rf_ret"]
    
    sharpe_dyn = sharpe_annual(ex_dyn)
    sharpe_bench = sharpe_annual(ex_bench)
    
    return {
        "ret_dyn": ret_dyn,
        "ret_bench": ret_bench,
        "g_dyn_annual": g_dyn_a,
        "g_bench_annual": g_bench_a,
        "sharpe_dyn": sharpe_dyn,
        "sharpe_bench": sharpe_bench,
    }


def evaluate_stock_bond_strategy(data_df, w_stock):
    """
    Evaluate stock-bond strategy performance.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with sp500_ret and rf_ret columns
    w_stock : pd.Series
        Stock weights series
    
    Returns
    -------
    dict
        Dictionary with strategy returns and performance metrics
    """
    strategy_sb = data_df[["sp500_ret", "rf_ret"]].copy()
    
    # Align weights with strategy_sb index
    w_stock_sb = w_stock.reindex(strategy_sb.index).fillna(0.5)
    
    strategy_sb["w_stock"] = w_stock_sb
    strategy_sb["w_bond"] = 1 - strategy_sb["w_stock"]
    
    # Bond return = 3M T-bill monthly return
    strategy_sb["ret_bond"] = strategy_sb["rf_ret"]
    
    # Dynamic stock–bond strategy
    ret_dyn_sb = (
        strategy_sb["w_stock"] * strategy_sb["sp500_ret"] +
        strategy_sb["w_bond"] * strategy_sb["ret_bond"]
    )
    
    # Benchmark: constant mix with same average risky weight
    w_bar_sb = strategy_sb["w_stock"].mean()
    ret_bench_sb = (
        w_bar_sb * strategy_sb["sp500_ret"] +
        (1 - w_bar_sb) * strategy_sb["ret_bond"]
    )
    
    # Geometric annualized returns
    g_dyn_sb_m = geometric_return(ret_dyn_sb)
    g_bench_sb_m = geometric_return(ret_bench_sb)
    
    g_dyn_sb_a = (1 + g_dyn_sb_m) ** 12 - 1
    g_bench_sb_a = (1 + g_bench_sb_m) ** 12 - 1
    
    # Sharpe ratios
    ex_dyn_sb = ret_dyn_sb - strategy_sb["rf_ret"]
    ex_bench_sb = ret_bench_sb - strategy_sb["rf_ret"]
    
    sharpe_dyn_sb = sharpe_annual(ex_dyn_sb)
    sharpe_bench_sb = sharpe_annual(ex_bench_sb)
    
    return {
        "ret_dyn": ret_dyn_sb,
        "ret_bench": ret_bench_sb,
        "g_dyn_annual": g_dyn_sb_a,
        "g_bench_annual": g_bench_sb_a,
        "sharpe_dyn": sharpe_dyn_sb,
        "sharpe_bench": sharpe_bench_sb,
    }

