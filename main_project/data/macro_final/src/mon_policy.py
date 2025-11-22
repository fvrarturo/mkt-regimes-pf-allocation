#!/usr/bin/env python3
"""
Monetary Policy Module

Extracts the 10y-2y yield curve slope as the monetary policy indicator.

Methodology:
- Use the 10y-2y spread (already computed in processed data)
- This captures policy stance and expectations
- Alternative: Fed Funds Rate (commented option)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_monetary_policy_factor(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Create monetary policy factor using 10y-2y yield curve slope.
    
    Args:
        data_dir: Path to data directory
        start_date: Start date for filtering (default: '1989-01-01')
        end_date: End date for filtering (default: '2025-12-31')
    
    Returns:
        DataFrame with columns: date, monetary_policy_factor
    """
    # Load 10y-2y spread data
    spread_file = data_dir / 'macro_processed' / 'mkt_vol' / '10y_2y_spread_processed.csv'
    
    df = pd.read_csv(spread_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Convert to monthly frequency (take first value of each month)
    df['year_month'] = df['date'].dt.to_period('M')
    df_monthly = df.groupby('year_month').first().reset_index()
    df_monthly['date'] = df_monthly['year_month'].dt.to_timestamp()
    
    # Use the spread value as the monetary policy factor
    # The spread is already computed as 10y - 2y
    result_df = pd.DataFrame({
        'date': df_monthly['date'],
        'monetary_policy_factor': df_monthly['value'].values
    })
    
    # Store metadata
    result_df.attrs = {
        'n_observations': len(result_df),
        'date_range': (result_df['date'].min(), result_df['date'].max()),
        'method': '10y-2y yield curve slope'
    }
    
    return result_df


def create_monetary_policy_factor_fedfunds(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Alternative: Create monetary policy factor using Fed Funds Rate.
    
    This is an alternative implementation if you prefer the policy rate directly.
    """
    fedfunds_file = data_dir / 'macro_processed' / 'mon_policy' / 'fedfunds_processed.csv'
    
    df = pd.read_csv(fedfunds_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Convert to monthly frequency
    df['year_month'] = df['date'].dt.to_period('M')
    df_monthly = df.groupby('year_month').first().reset_index()
    df_monthly['date'] = df_monthly['year_month'].dt.to_timestamp()
    
    result_df = pd.DataFrame({
        'date': df_monthly['date'],
        'monetary_policy_factor': df_monthly['value'].values
    })
    
    result_df.attrs = {
        'n_observations': len(result_df),
        'date_range': (result_df['date'].min(), result_df['date'].max()),
        'method': 'Fed Funds Rate'
    }
    
    return result_df


if __name__ == "__main__":
    # Test the module
    data_dir = Path(__file__).parent.parent.parent.parent
    result = create_monetary_policy_factor(data_dir)
    print(f"Monetary policy factor created: {len(result)} observations")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"Method: {result.attrs.get('method', 'N/A')}")
    print(f"\nFirst few rows:")
    print(result.head())

