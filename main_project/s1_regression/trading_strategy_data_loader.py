"""
Data loading functions for trading strategy with full sample.

This module handles loading S&P 500 and 3-month T-bill data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_base_path():
    """Get the base path for macro data."""
    return Path(__file__).parent.parent.parent / "data" / "macro_processed"


def load_market_data(start="1990-01-02", end=None):
    """
    Load S&P 500 and 3-month T-bill data.
    
    Parameters
    ----------
    start : str
        Start date for data
    end : str or None
        End date for data (None = most recent)
    
    Returns
    -------
    tuple
        (sp500, tbill) DataFrames
    """
    BASE = get_base_path()
    
    # S&P 500
    sp500_raw = pd.read_csv(
        BASE / "sp500_processed.csv",
        parse_dates=["date"]
    ).set_index("date")
    
    sp500 = sp500_raw[["value"]].rename(columns={"value": "SP500"})
    sp500 = sp500.loc[start:end]
    
    # 3M T-bill
    tbill_raw = pd.read_csv(
        BASE / "3m_yield_processed.csv",
        parse_dates=["date"]
    ).set_index("date")
    
    tbill = tbill_raw[["value"]].rename(columns={"value": "TB3MS"})
    tbill = tbill.loc[start:end]
    
    return sp500, tbill


def build_monthly_returns(sp500, tbill):
    """
    Build monthly returns from S&P 500 and T-bill data.
    
    Parameters
    ----------
    sp500 : pd.DataFrame
        S&P 500 price data
    tbill : pd.DataFrame
        T-bill yield data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sp500_ret, rf_ret, erp
    """
    # Build monthly S&P500 returns
    sp500_m = sp500.resample("ME").last()
    sp500_level = sp500_m["SP500"]
    sp500_ret = sp500_level.pct_change().dropna()
    sp500_ret.index = sp500_ret.index.to_period("M")
    
    # Build monthly T-bill returns from annualized rate
    tbill_m = tbill.resample("ME").last()
    rf_ann = tbill_m["TB3MS"] / 100.0  # convert % → decimal
    
    # annual → monthly simple return
    rf_monthly = (1 + rf_ann) ** (1/12) - 1
    rf_monthly = rf_monthly.dropna()
    rf_monthly.index = rf_monthly.index.to_period("M")
    
    # Align both series to same monthly index
    idx = sp500_ret.index.intersection(rf_monthly.index)
    
    sp500_ret = sp500_ret.loc[idx]
    rf_monthly = rf_monthly.loc[idx]
    
    # Build data frame
    data = pd.DataFrame({
        "sp500_ret": sp500_ret,
        "rf_ret": rf_monthly
    })
    data["erp"] = data["sp500_ret"] - data["rf_ret"]
    
    return data

