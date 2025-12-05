"""
Data loading functions for ARX(3) model.

This module handles loading and preprocessing of:
- S&P 500 returns
- 3-month T-bill returns
- Macroeconomic variables
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_base_path():
    """Get the base path for macro data."""
    return Path(__file__).parent.parent.parent / "data" / "macro_processed_full"


def load_monthly_series(path, name):
    """
    Load a macro series with a 'date' column and 'value' column,
    resample to monthly endpoints, and forward-fill.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file with 'date' and 'value' columns
    name : str
        Name to assign to the series
    
    Returns
    -------
    pd.DataFrame
        Monthly series with single column named `name`
    """
    df = (
        pd.read_csv(path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    # keep 'value' as level, monthly frequency
    df = df[["value"]].rename(columns={"value": name})
    df_m = df.resample("M").last().ffill()
    return df_m


def load_market_data():
    """
    Load S&P 500 and 3-month T-bill data, compute ERP.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sp500_ret, rf_ret, erp, erp_next, 
        sp500_ret_next, rf_ret_next
    """
    BASE = get_base_path()
    
    # Paths for price / yield data
    path_sp = BASE / "other" / "sp500_processed.csv"
    path_rf = BASE / "other" / "3m_yield_processed.csv"
    
    # --- S&P 500 ---
    sp = pd.read_csv(path_sp, parse_dates=["date"]).set_index("date").sort_index()
    
    # sp500_processed.csv already has pct_change_mom in percent
    # convert to simple monthly return in decimals
    sp["sp500_ret"] = sp["pct_change_mom"] / 100.0
    sp_m = sp[["sp500_ret"]].resample("M").last()
    
    # --- 3M T-bill ---
    rf = pd.read_csv(path_rf, parse_dates=["date"]).set_index("date").sort_index()
    # value is annualized yield in percent
    rf_ann = rf["value"] / 100.0
    rf_m = ((1 + rf_ann) ** (1/12) - 1).to_frame("rf_ret").resample("M").last()
    
    # align
    idx = sp_m.index.intersection(rf_m.index)
    sp_m = sp_m.loc[idx]
    rf_m = rf_m.loc[idx]
    
    # ERP_t = SP_t - RF_t  (current month)
    model_df = sp_m.join(rf_m, how="inner")
    model_df["erp"] = model_df["sp500_ret"] - model_df["rf_ret"]
    
    # ERP_next_t = ERP_{t+1}, sp500_ret_next & rf_ret_next
    model_df["erp_next"] = model_df["erp"].shift(-1)
    model_df["sp500_ret_next"] = model_df["sp500_ret"].shift(-1)
    model_df["rf_ret_next"] = model_df["rf_ret"].shift(-1)
    
    # keep sample where next-month data exists
    model_df = model_df.dropna()
    
    # convert index to PeriodIndex (monthly)
    model_df.index = model_df.index.to_period("M")
    
    return model_df


def load_all_macro_variables():
    """
    Load all macroeconomic variables used in ARX(3) model.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all macro variables as columns
    """
    BASE = get_base_path()
    
    paths_macro = {
        # Inflation indices
        "cpi":          BASE / "inflation" / "cpi_processed.csv",
        "pce":          BASE / "inflation" / "PCE_price_index_processed.csv",
        "ppi":          BASE / "inflation" / "PPI_inflation_processed.csv",
        
        # Real activity
        "indprod":      BASE / "ec_growth" / "industrial_production_processed.csv",
        "retail_sales": BASE / "ec_growth" / "retail_sales_processed.csv",
        "inventories":  BASE / "ec_growth" / "tot_business_inventories_processed.csv",
        "export_px":    BASE / "ec_growth" / "export_price_index_processed.csv",
        "import_px":    BASE / "ec_growth" / "import_price_index_processed.csv",
        "unemp":        BASE / "ec_growth" / "unemployment_processed.csv",
        
        # Money and policy
        "m2":           BASE / "mon_policy" / "m2_real_money_supply_processed.csv",
        "fedfunds":     BASE / "mon_policy" / "fedfunds_processed.csv",
        "discount_rate": BASE / "mon_policy" / "fed_reserve_discount_rate_processed.csv",
        
        # Yields, spreads, vol
        "spread_10y_2y": BASE / "mkt_vol" / "10y_2y_spread_processed.csv",
        "nat_fin_cond": BASE / "mkt_vol" / "nat_fin_condition_indx_processed.csv",
        "nasdaq_vol":   BASE / "mkt_vol" / "nasdaq_vol_indx_processed.csv",
        "hy_spread":    BASE / "other" / "bofa_highyield_spread_processed.csv",
        "y2":           BASE / "other" / "2y_yield_processed.csv",
        "y3m":          BASE / "other" / "3m_yield_processed.csv",
        "y10":          BASE / "other" / "10y_yield_processed.csv",
    }
    
    macro_m_list = []
    for name, path in paths_macro.items():
        if path.exists():
            series_m = load_monthly_series(path, name)
            macro_m_list.append(series_m)
        else:
            print(f"Warning: {path} not found, skipping {name}")
    
    if not macro_m_list:
        raise FileNotFoundError("No macro variables could be loaded")
    
    macro_m = pd.concat(macro_m_list, axis=1)
    
    # convert to PeriodIndex
    macro_m.index = macro_m.index.to_period("M")
    
    return macro_m


def prepare_model_data():
    """
    Prepare the full modeling dataset by combining market data and macro variables.
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with market returns, ERP, and macro variables
    """
    # Load market data
    model_df = load_market_data()
    
    # Load macro variables
    macro_m = load_all_macro_variables()
    
    # Lag everything by one month to mimic information delay:
    # at the end of month t we know macro_{t-1}, ERP_t, etc
    macro_lag = macro_m.shift(1)
    
    # Merge with target + returns
    df = model_df.join(macro_lag, how="inner").dropna()
    
    return df

