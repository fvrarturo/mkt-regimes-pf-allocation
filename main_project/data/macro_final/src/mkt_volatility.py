#!/usr/bin/env python3
"""
Market Volatility / Financial Conditions Module

Extracts the NFCI (National Financial Conditions Index) as the market volatility indicator.

Methodology:
- Use NFCI (Chicago Fed Financial Conditions Index)
- This aggregates credit, liquidity, leverage, funding stress, vol, spreads, etc.
- Alternative: VIX (commented option)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_market_volatility_factor(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Create market volatility/financial conditions factor using NFCI.
    
    Args:
        data_dir: Path to data directory
        start_date: Start date for filtering (default: '1989-01-01')
        end_date: End date for filtering (default: '2025-12-31')
    
    Returns:
        DataFrame with columns: date, market_volatility_factor
    """
    # Load NFCI data
    nfci_file = data_dir / 'macro_processed' / 'mkt_vol' / 'nat_fin_condition_indx_processed.csv'
    
    df = pd.read_csv(nfci_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Convert to monthly frequency (take first value of each month)
    df['year_month'] = df['date'].dt.to_period('M')
    df_monthly = df.groupby('year_month').first().reset_index()
    df_monthly['date'] = df_monthly['year_month'].dt.to_timestamp()
    
    # Use the NFCI value as the market volatility factor
    result_df = pd.DataFrame({
        'date': df_monthly['date'],
        'market_volatility_factor': df_monthly['value'].values
    })
    
    # Store metadata
    result_df.attrs = {
        'n_observations': len(result_df),
        'date_range': (result_df['date'].min(), result_df['date'].max()),
        'method': 'NFCI (National Financial Conditions Index)'
    }
    
    return result_df


def create_market_volatility_factor_vix(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Alternative: Create market volatility factor using VIX.
    
    This is an alternative implementation if you prefer VIX.
    """
    vix_file = data_dir / 'macro_processed' / 'mkt_vol' / 'vix_processed.csv'
    
    df = pd.read_csv(vix_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Convert to monthly frequency
    df['year_month'] = df['date'].dt.to_period('M')
    df_monthly = df.groupby('year_month').first().reset_index()
    df_monthly['date'] = df_monthly['year_month'].dt.to_timestamp()
    
    result_df = pd.DataFrame({
        'date': df_monthly['date'],
        'market_volatility_factor': df_monthly['value'].values
    })
    
    result_df.attrs = {
        'n_observations': len(result_df),
        'date_range': (result_df['date'].min(), result_df['date'].max()),
        'method': 'VIX'
    }
    
    return result_df


if __name__ == "__main__":
    # Test the module
    data_dir = Path(__file__).parent.parent.parent.parent
    result = create_market_volatility_factor(data_dir)
    print(f"Market volatility factor created: {len(result)} observations")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"Method: {result.attrs.get('method', 'N/A')}")
    print(f"\nFirst few rows:")
    print(result.head())

