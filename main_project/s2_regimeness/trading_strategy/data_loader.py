"""
Data loading utilities for Step 3 trading evaluation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _get_base_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return Path(__file__).parent.parent


def load_market_data(base_dir: Optional[Path] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load monthly equity returns, bond returns, and ERP.
    """
    base_dir = _get_base_dir(base_dir)
    data_dir = base_dir / "data" / "macro_processed"

    sp500 = pd.read_csv(data_dir / "sp500_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    equity_returns = (sp500["pct_change_mom"] / 100.0).resample("M").last()

    tbill = pd.read_csv(data_dir / "3m_yield_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    bond_returns = (tbill["value"] / 100.0 / 12.0).resample("M").last()

    aligned = pd.DataFrame({"equity_return": equity_returns, "bond_return": bond_returns}).dropna()
    erp = aligned["equity_return"] - aligned["bond_return"]

    equity_returns = equity_returns.reindex(aligned.index)
    bond_returns = bond_returns.reindex(aligned.index)

    return equity_returns, bond_returns, erp


def load_all_macro_variables(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all macro variables from macro_processed_full directories.
    
    This includes variables from:
    - ec_growth/
    - inflation/
    - mkt_vol/
    - mon_policy/
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all macro variables, indexed by date
    """
    base_dir = _get_base_dir(base_dir)
    macro_data_dir = base_dir / "data" / "macro_processed_full"
    
    # Define directories and their expected files
    macro_dirs = {
        "ec_growth": [
            "export_price_index_processed.csv",
            "gdp_processed.csv",
            "import_price_index_processed.csv",
            "industrial_production_processed.csv",
            "real_gdp_processed.csv",
            "retail_sales_processed.csv",
            "tot_business_inventories_processed.csv",
            "unemployment_processed.csv",
        ],
        "inflation": [
            "PCE_price_index_processed.csv",
            "PPI_inflation_processed.csv",
            "cpi_processed.csv",
        ],
        "mkt_vol": [
            "10y_2y_spread_processed.csv",
            "3month_vol_index_sp500_processed.csv",
            "nasdaq_vol_indx_processed.csv",
            "nat_fin_condition_indx_processed.csv",
            "vix_processed.csv",
            # Also check for monthly versions
            "10y_2y_spread_processed_monthly.csv",
            "nat_fin_condition_indx_processed_monthly.csv",
            "vix_processed_monthly.csv",
        ],
        "mon_policy": [
            "10y_treasury_const_maturity_rate_processed.csv",
            "fed_reserve_discount_rate_processed.csv",
            "fedfunds_processed.csv",
            "m2_real_money_supply_processed.csv",
        ],
    }
    
    all_data = []
    
    for category, files in macro_dirs.items():
        category_dir = macro_data_dir / category
        
        for filename in files:
            file_path = category_dir / filename
            
            if not file_path.exists():
                continue
            
            try:
                df = pd.read_csv(file_path, parse_dates=["date"])
                
                # Use 'value' column if available, otherwise use first numeric column
                if "value" in df.columns:
                    value_col = "value"
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        continue
                    value_col = numeric_cols[0]
                
                # Create variable name from filename
                var_name = (
                    filename.replace("_processed 2.csv", "")
                    .replace("_processed_monthly.csv", "")
                    .replace("_processed.csv", "")
                    .replace(" ", "_")
                )
                
                # Select date and value
                df_subset = df[["date", value_col]].copy()
                df_subset.columns = ["date", var_name]
                
                # Convert to monthly if not already
                df_subset["date"] = pd.to_datetime(df_subset["date"])
                df_subset = df_subset.set_index("date").sort_index()
                df_subset = df_subset.resample("ME").last().ffill()
                
                all_data.append(df_subset)
                
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No macro variables loaded from macro_processed_full")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, axis=1)
    combined_df.index = combined_df.index.to_period("M").to_timestamp("M")
    
    # Also add base factors from final_macro.csv
    macro_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    base_macro = pd.read_csv(macro_path, parse_dates=["date"]).set_index("date").sort_index()
    base_macro.index = base_macro.index.to_period("M").to_timestamp("M")
    
    # Join base factors
    combined_df = combined_df.join(base_macro, how="inner")
    
    return combined_df.dropna(how="all")
