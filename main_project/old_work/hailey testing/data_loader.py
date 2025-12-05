"""
Data Loading Module
Loads and prepares data from various sources for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_macro_data(
    macro_path: Path,
    start_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load macro final data and filter by start date.
    
    Parameters:
    -----------
    macro_path : Path
        Path to final_macro.csv
    start_date : str, optional
        Start date to filter from (YYYY-MM-DD format)
    
    Returns:
    --------
    pd.DataFrame
        Loaded and filtered macro data
    """
    df = pd.read_csv(macro_path)
    df['date'] = pd.to_datetime(df['date'])
    
    if start_date:
        start_date_dt = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date_dt].reset_index(drop=True)
    
    return df


def load_market_data(
    stock_path: Path,
    bond_path: Path,
    start_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stock and bond data, aggregate bond data to monthly frequency.
    
    Parameters:
    -----------
    stock_path : Path
        Path to sp500_processed.csv
    bond_path : Path
        Path to 3m_yield_processed.csv
    start_date : str, optional
        Start date to filter from (YYYY-MM-DD format)
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Stock and bond DataFrames (bond aggregated to monthly)
    """
    stock_df = pd.read_csv(stock_path)
    bond_df = pd.read_csv(bond_path)
    
    # Convert date columns to datetime
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    bond_df['date'] = pd.to_datetime(bond_df['date'])
    
    # Filter by start date if provided
    if start_date:
        start_date_dt = pd.to_datetime(start_date)
        stock_df = stock_df[stock_df['date'] >= start_date_dt].reset_index(drop=True)
        bond_df = bond_df[bond_df['date'] >= start_date_dt].reset_index(drop=True)
    
    # Aggregate bond_df to monthly frequency
    bond_df = bond_df.set_index('date')
    bond_df_monthly = bond_df.resample('ME').last()
    bond_df_monthly = bond_df_monthly.reset_index()
    
    return stock_df, bond_df_monthly


def load_all_data(
    base_dir: Path,
    start_date: str = "1990-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data sources (macro, stock, bond).
    
    Parameters:
    -----------
    base_dir : Path
        Base directory containing data subdirectories
    start_date : str
        Start date to filter from
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Macro, stock, and bond DataFrames
    """
    macro_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    stock_path = base_dir / "data" / "macro_processed" / "sp500_processed.csv"
    bond_path = base_dir / "data" / "macro_processed" / "3m_yield_processed.csv"
    
    df = load_macro_data(macro_path, start_date)
    stock_df, bond_df = load_market_data(stock_path, bond_path, start_date)
    
    print("All dataframes aligned to start from", start_date)
    print(f"\ndf shape: {df.shape}, start date: {df['date'].min()}")
    print(f"stock_df shape: {stock_df.shape}, start date: {stock_df['date'].min()}")
    print(f"bond_df shape: {bond_df.shape}, start date: {bond_df['date'].min()}")
    
    return df, stock_df, bond_df

