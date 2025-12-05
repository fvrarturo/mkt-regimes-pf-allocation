"""
Data loading functions for 3-month rolling regression.

This module handles loading monthly trading data and macro variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_base_path():
    """Get the base path for macro data."""
    return Path(__file__).parent.parent.parent / "data" / "macro_processed"


def get_macro_base_path():
    """Get the base path for full macro data."""
    return Path(__file__).parent.parent.parent / "data" / "macro_processed_full"


def load_market_data():
    """
    Load S&P 500 and 3-month T-bill data, compute log returns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sp500_ret, rf_ret, erp, erp_next columns
    """
    BASE = get_base_path()
    
    sp500_path = BASE / "sp500_processed.csv"
    tbill_path = BASE / "3m_yield_processed.csv"
    
    # Load S&P500 level
    sp500_raw = pd.read_csv(sp500_path, parse_dates=["date"]).set_index("date")
    sp500 = sp500_raw[["value"]].rename(columns={"value": "SP500"})
    
    # Load 3M T-bill (annualized %)
    tbill_raw = pd.read_csv(tbill_path, parse_dates=["date"]).set_index("date")
    tbill = tbill_raw[["value"]].rename(columns={"value": "TB3MS"})
    
    # Build monthly log returns
    sp500_m = sp500.resample("ME").last()
    sp500_ret = np.log(sp500_m["SP500"] / sp500_m["SP500"].shift(1)).dropna()
    sp500_ret.index = sp500_ret.index.to_period("M")
    
    # 3M T-bill monthly log return from annual yield
    tbill_m = tbill.resample("ME").last()
    rf_ann = tbill_m["TB3MS"] / 100.0
    rf_ret = np.log((1 + rf_ann) ** (1/12)).dropna()
    rf_ret.index = rf_ret.index.to_period("M")
    
    # Align
    idx = sp500_ret.index.intersection(rf_ret.index)
    sp500_ret = sp500_ret.loc[idx]
    rf_ret = rf_ret.loc[idx]
    
    # Build data frame
    data = pd.DataFrame({
        "sp500_ret": sp500_ret,
        "rf_ret": rf_ret
    })
    data["erp"] = data["sp500_ret"] - data["rf_ret"]
    data["erp_next"] = data["erp"].shift(-1)
    
    return data.dropna()


def load_monthly_series(path, name):
    """
    Load a macro series, resample to month-end, keep 'value' column.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file
    name : str
        Name to assign to series
    
    Returns
    -------
    pd.DataFrame
        Monthly series
    """
    df = (
        pd.read_csv(path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    df = df[["value"]].rename(columns={"value": name}).resample("ME").last()
    df = df.ffill()  # important: quarterly → monthly via ffill
    return df


def load_all_macro_variables():
    """
    Load all macroeconomic variables.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all macro variables
    """
    BASE = get_macro_base_path()
    
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
    
    macro_list = []
    for name, path in paths.items():
        if path.exists():
            macro_list.append(load_monthly_series(path, name))
        else:
            print(f"Warning: {path} not found, skipping {name}")
    
    macro_m = pd.concat(macro_list, axis=1)
    macro_m.index = macro_m.index.to_period("M")
    
    return macro_m


def prepare_model_data():
    """
    Prepare full modeling dataset.
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with returns and macro variables
    """
    # Load market data
    model_df = load_market_data()
    
    # Load macro variables
    macro_m = load_all_macro_variables()
    
    # Lag macro by one month
    macro_lag = macro_m.shift(1)
    
    # Merge
    df = model_df.join(macro_lag, how="inner").dropna()
    
    return df

