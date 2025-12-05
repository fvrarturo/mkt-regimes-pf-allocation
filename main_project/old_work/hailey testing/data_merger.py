"""
Data Merging Module
Merges and aggregates data from multiple sources for forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_forecasting_data(
    yahoo_path: Path,
    macro_path: Path,
    tips_path: Path,
    start_date: str = "1980-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load forecasting data sources.
    
    Parameters:
    -----------
    yahoo_path : Path
        Path to Yahoo Finance data CSV
    macro_path : Path
        Path to merged macro dataset CSV
    tips_path : Path
        Path to TIPS breakeven CSV
    start_date : str
        Start date to filter from
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Yahoo, macro, and TIPS DataFrames
    """
    yfdf = pd.read_csv(yahoo_path)
    macrodf = pd.read_csv(macro_path)
    tipsdf = pd.read_csv(tips_path)
    
    # Standardize column names to lowercase
    yfdf.columns = yfdf.columns.str.lower()
    tipsdf.columns = tipsdf.columns.str.lower()
    macrodf.columns = macrodf.columns.str.lower()
    
    return yfdf, macrodf, tipsdf


def merge_yahoo_tips(
    yfdf: pd.DataFrame,
    tipsdf: pd.DataFrame,
    start_date: str = "1980-01-01"
) -> pd.DataFrame:
    """
    Merge Yahoo Finance and TIPS data.
    
    Parameters:
    -----------
    yfdf : pd.DataFrame
        Yahoo Finance DataFrame
    tipsdf : pd.DataFrame
        TIPS DataFrame
    start_date : str
        Start date to filter from
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame
    """
    # Merge on 'date'
    merged_df = yfdf.merge(tipsdf, on='date', how='outer')
    
    # Convert date to datetime
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Filter to start date
    start_date_dt = pd.to_datetime(start_date)
    merged_df = merged_df[merged_df['date'] >= start_date_dt].reset_index(drop=True)
    
    return merged_df


def aggregate_to_monthly(
    df: pd.DataFrame,
    method: str = "mean"
) -> pd.DataFrame:
    """
    Aggregate daily data to monthly frequency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date index
    method : str
        Aggregation method ('mean', 'last', 'first')
    
    Returns:
    --------
    pd.DataFrame
        Monthly aggregated DataFrame
    """
    df = df.copy()
    df['month'] = df.index.to_period('M')
    
    # Reset index to have date as column
    df_reset = df.reset_index()
    
    # Group by month and aggregate
    if method == "mean":
        df_monthly = df_reset.groupby('month').mean(numeric_only=True).reset_index()
    elif method == "last":
        df_monthly = df_reset.groupby('month').last().reset_index()
    elif method == "first":
        df_monthly = df_reset.groupby('month').first().reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Convert month back to timestamp
    df_monthly['date'] = df_monthly['month'].dt.to_timestamp()
    
    return df_monthly


def combine_datasets(
    merged_df: pd.DataFrame,
    macrodf: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine merged Yahoo/TIPS data with macro final data.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged Yahoo/TIPS DataFrame (daily)
    macrodf : pd.DataFrame
        Macro final DataFrame (monthly)
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame aligned on date
    """
    # Set date as index for both
    macrodf = macrodf.copy()
    merged_df = merged_df.copy()
    
    # Handle macrodf: if already indexed by date, reset first
    if isinstance(macrodf.index, pd.DatetimeIndex):
        macrodf = macrodf.reset_index()
        # If reset_index created 'index' column, rename it to 'date'
        if 'index' in macrodf.columns:
            macrodf = macrodf.rename(columns={'index': 'date'})
    
    # Ensure date column exists and is datetime
    if 'date' not in macrodf.columns:
        raise ValueError("macrodf must have a 'date' column or be indexed by date")
    macrodf['date'] = pd.to_datetime(macrodf['date'])
    macrodf = macrodf.set_index('date')
    
    # Handle merged_df: if already indexed by date, reset first
    if isinstance(merged_df.index, pd.DatetimeIndex):
        merged_df = merged_df.reset_index()
        # If reset_index created 'index' column, rename it to 'date'
        if 'index' in merged_df.columns:
            merged_df = merged_df.rename(columns={'index': 'date'})
    
    # Ensure date column exists and is datetime
    if 'date' not in merged_df.columns:
        raise ValueError("merged_df must have a 'date' column or be indexed by date")
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.set_index('date')
    
    # Aggregate merged_df from daily to monthly frequency (taking average)
    merged_df_monthly = aggregate_to_monthly(merged_df, method="mean")
    
    # Now align both dataframes on date
    macrodf_reset = macrodf.reset_index()
    combined_df = merged_df_monthly.merge(macrodf_reset, on='date', how='inner')
    
    return combined_df


def prepare_forecasting_data(
    yahoo_path: Path,
    macro_merged_path: Path,
    tips_path: Path,
    macro_final_path: Path,
    start_date: str = "1980-01-01"
) -> pd.DataFrame:
    """
    Complete pipeline to prepare forecasting data.
    
    Parameters:
    -----------
    yahoo_path : Path
        Path to Yahoo Finance CSV
    macro_merged_path : Path
        Path to merged macro dataset CSV
    tips_path : Path
        Path to TIPS CSV
    macro_final_path : Path
        Path to final macro CSV
    start_date : str
        Start date to filter from
    
    Returns:
    --------
    pd.DataFrame
        Combined and prepared DataFrame
    """
    # Load data
    yfdf, macrodf_merged, tipsdf = load_forecasting_data(
        yahoo_path, macro_merged_path, tips_path, start_date
    )
    
    # Merge Yahoo and TIPS
    merged_df = merge_yahoo_tips(yfdf, tipsdf, start_date)
    
    # Load macro final
    macrodf_final = pd.read_csv(macro_final_path)
    macrodf_final['date'] = pd.to_datetime(macrodf_final['date'])
    macrodf_final = macrodf_final.set_index('date')
    
    # Combine datasets
    combined_df = combine_datasets(merged_df, macrodf_final)
    
    print(f"Combined df shape: {combined_df.shape}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df

