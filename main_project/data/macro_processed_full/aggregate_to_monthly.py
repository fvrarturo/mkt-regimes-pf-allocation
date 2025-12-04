#!/usr/bin/env python3
"""
Aggregate high-frequency market volatility data to monthly frequency.

This script aggregates:
- vix_processed 2.csv (daily)
- nat_fin_condition_indx_processed 2.csv (weekly)
- 10y_2y_spread_processed 2.csv (daily)

to monthly frequency using end-of-month values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Get script directory
SCRIPT_DIR = Path(__file__).resolve().parent
MKT_VOL_DIR = SCRIPT_DIR / 'mkt_vol'

def aggregate_to_monthly(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'value') -> pd.DataFrame:
    """
    Aggregate dataframe to monthly frequency using end-of-month values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with date column
    date_col : str
        Name of date column
    value_col : str
        Name of value column to aggregate
    
    Returns:
    --------
    pd.DataFrame with monthly aggregated data
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Resample to monthly end-of-month
    monthly = df.resample('ME').last()
    
    # Reset index to have date as column
    monthly = monthly.reset_index()
    monthly.columns.name = None
    
    return monthly

def main():
    """Main function to aggregate all three files."""
    print("="*80)
    print("AGGREGATING MARKET VOLATILITY DATA TO MONTHLY FREQUENCY")
    print("="*80)
    
    files_to_process = {
        'vix': 'vix_processed 2.csv',
        'nat_fin_condition_indx': 'nat_fin_condition_indx_processed 2.csv',
        '10y_2y_spread': '10y_2y_spread_processed 2.csv'
    }
    
    results = {}
    
    for name, filename in files_to_process.items():
        file_path = MKT_VOL_DIR / filename
        
        if not file_path.exists():
            print(f"\n⚠️  Warning: File not found: {file_path}")
            continue
        
        print(f"\nProcessing {name}...")
        print(f"  File: {filename}")
        
        # Read CSV
        df = pd.read_csv(file_path)
        print(f"  Original shape: {df.shape}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Determine value column (could be 'value' or other)
        value_cols = [col for col in df.columns if col != 'date']
        if 'value' in df.columns:
            value_col = 'value'
        elif len(value_cols) == 1:
            value_col = value_cols[0]
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                value_col = numeric_cols[0]
            else:
                print(f"  ⚠️  Warning: Could not determine value column for {name}")
                continue
        
        # Aggregate to monthly
        monthly_df = aggregate_to_monthly(df, date_col='date', value_col=value_col)
        
        # Keep all columns (not just value)
        # Re-read and aggregate all columns properly
        df_full = pd.read_csv(file_path)
        df_full['date'] = pd.to_datetime(df_full['date'])
        df_full = df_full.set_index('date').sort_index()
        
        # Resample all columns to monthly (last value of month)
        monthly_full = df_full.resample('ME').last()
        monthly_full = monthly_full.reset_index()
        
        print(f"  Monthly shape: {monthly_full.shape}")
        print(f"  Monthly date range: {monthly_full['date'].min()} to {monthly_full['date'].max()}")
        
        # Save aggregated file
        output_filename = filename.replace(' 2.csv', '_monthly.csv')
        output_path = MKT_VOL_DIR / output_filename
        monthly_full.to_csv(output_path, index=False)
        print(f"  ✓ Saved to: {output_filename}")
        
        results[name] = {
            'original_shape': df.shape,
            'monthly_shape': monthly_full.shape,
            'output_file': output_filename
        }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, info in results.items():
        print(f"\n{name}:")
        print(f"  Original: {info['original_shape'][0]} observations")
        print(f"  Monthly: {info['monthly_shape'][0]} observations")
        print(f"  Output: {info['output_file']}")
    
    print("\n✓ Aggregation complete!")

if __name__ == "__main__":
    main()

