"""
Data loading functions for quarterly regression analysis.

This module handles loading and preprocessing quarterly EMRP and macro variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_base_path():
    """Get the base path for macro data."""
    return Path(__file__).parent.parent.parent / "data" / "macro_processed1"


def load_monthly(path, date_col="date", value_col="value", name=None):
    """
    Load monthly series and resample to month-end.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file
    date_col : str
        Name of date column
    value_col : str
        Name of value column
    name : str or None
        Name to assign to series (defaults to value_col)
    
    Returns
    -------
    pd.DataFrame
        Monthly series
    """
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    if name is None:
        name = value_col
    return df[[value_col]].rename(columns={value_col: name}).resample("ME").last()


def safe_pct_change(df, lag=1):
    """
    Calculate safe percentage change, handling zeros and infinities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    lag : int
        Lag for percentage change
    
    Returns
    -------
    pd.DataFrame
        Percentage change dataframe
    """
    out = df.replace(0, np.nan).pct_change(lag) * 100.0
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def load_quarterly_emrp():
    """
    Load and compute quarterly geometric EMRP.
    
    Returns
    -------
    pd.DataFrame
        Quarterly EMRP with EMRP_q and EMRP_next_q columns
    """
    BASE = get_base_path()
    
    # Load S&P 500
    sp = pd.read_csv(
        BASE / "other" / "sp500_processed.csv",
        parse_dates=["date"]
    ).set_index("date").sort_index()
    
    sp["sp500_geo_m"] = sp["pct_change_mom"] / 100.0
    sp_m = sp.resample("ME").last()[["sp500_geo_m"]]
    
    # Load 3M T-bill
    rf = pd.read_csv(
        BASE / "other" / "3m_yield_processed.csv",
        parse_dates=["date"]
    ).set_index("date").sort_index()
    
    rf["r_daily"] = (rf["value"] / 100.0) / 252.0
    rf_m = rf["r_daily"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    rf_m = rf_m.to_frame("rf_geo_m")
    
    # Combine and compute EMRP
    m = sp_m.join(rf_m, how="inner")
    m["EMRP_geo_m"] = m["sp500_geo_m"] - m["rf_geo_m"]
    
    # Aggregate to quarterly
    emrp_q = (1 + m["EMRP_geo_m"]).resample("QE").prod() - 1
    emrp_q = emrp_q.to_frame("EMRP_q")
    emrp_q["EMRP_next_q"] = emrp_q["EMRP_q"].shift(-1)
    
    return emrp_q


def load_quarterly_macro():
    """
    Load and process quarterly macro variables.
    
    Returns
    -------
    pd.DataFrame
        Quarterly macro variables with QoQ changes and levels
    """
    BASE = get_base_path()
    
    paths = {
        "cpi": BASE / "inflation" / "cpi_processed.csv",
        "pce": BASE / "inflation" / "PCE_price_index_processed.csv",
        "ppi": BASE / "inflation" / "PPI_inflation_processed.csv",
        "indprod": BASE / "ec_growth" / "industrial_production_processed.csv",
        "retail_sales": BASE / "ec_growth" / "retail_sales_processed.csv",
        "inventories": BASE / "ec_growth" / "tot_business_inventories_processed.csv",
        "export_px": BASE / "ec_growth" / "export_price_index_processed.csv",
        "import_px": BASE / "ec_growth" / "import_price_index_processed.csv",
        "unemp": BASE / "ec_growth" / "unemployment_processed.csv",
        "m2": BASE / "mon_policy" / "m2_real_money_supply_processed.csv",
        "fedfunds": BASE / "mon_policy" / "fedfunds_processed.csv",
        "discount_rate": BASE / "mon_policy" / "fed_reserve_discount_rate_processed.csv",
        "spread_10y_2y": BASE / "mkt_vol" / "10y_2y_spread_processed.csv",
        "nat_fin_cond": BASE / "mkt_vol" / "nat_fin_condition_indx_processed.csv",
        "nasdaq_vol": BASE / "mkt_vol" / "nasdaq_vol_indx_processed.csv",
        "hy_spread": BASE / "other" / "bofa_highyield_spread_processed.csv",
        "y2": BASE / "other" / "2y_yield_processed.csv",
        "y10": BASE / "other" / "10y_yield_processed.csv",
    }
    
    # Load all monthly macro series
    macro_list = []
    for name, path in paths.items():
        if path.exists():
            macro_list.append(load_monthly(path, name=name))
        else:
            print(f"Warning: {path} not found, skipping {name}")
    
    macro_levels_m = pd.concat(macro_list, axis=1)
    macro_q_levels = macro_levels_m.resample("QE").last()
    
    # Compute QoQ changes
    macro_q = safe_pct_change(macro_q_levels).add_suffix("_qoq")
    
    # Keep level variables as-is
    level_vars = [
        "unemp_rate", "fedfunds", "discount_rate", "spread_10y_2y",
        "nat_fin_cond", "nasdaq_vol", "hy_spread", "y2", "y10"
    ]
    
    for col in level_vars:
        if col in macro_q_levels.columns:
            macro_q[col] = macro_q_levels[col]
            macro_q.drop(columns=[col + "_qoq"], errors="ignore", inplace=True)
    
    # Lag all predictors by one quarter
    macro_q_lagged = macro_q.shift(1)
    
    return macro_q_lagged


def prepare_regression_data():
    """
    Prepare full quarterly regression dataset.
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with EMRP and macro variables
    """
    emrp_q = load_quarterly_emrp()
    macro_q_lagged = load_quarterly_macro()
    
    # Merge
    reg_q = emrp_q.join(macro_q_lagged, how="inner").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    
    return reg_q

