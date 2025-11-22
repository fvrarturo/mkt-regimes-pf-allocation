#!/usr/bin/env python3
"""
Main script to create final macro.csv file

This script:
1. Creates inflation factor (PC1 from CPI, PCE, PPI)
2. Creates growth factor (PC1 from IP, Retail Sales, Unemployment, GDP)
3. Extracts monetary policy factor (10y-2y spread)
4. Extracts market volatility factor (NFCI)
5. Merges all factors into a single monthly dataset (1989-2025)
6. Saves to final_macro.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inflation import create_inflation_factor
from ec_growth import create_growth_factor
from mon_policy import create_monetary_policy_factor
from mkt_volatility import create_market_volatility_factor


def create_final_macro(
    data_dir: Path,
    output_file: Path,
    start_date: str = '1989-01-01',
    end_date: str = '2025-12-31'
) -> pd.DataFrame:
    """
    Create final macro.csv file with all four factors.
    
    Args:
        data_dir: Path to data directory (parent of macro_processed)
        output_file: Path to output CSV file
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        DataFrame with final macro factors
    """
    print("=" * 80)
    print("Creating Final Macro Dataset")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}\n")
    
    # 1. Create inflation factor
    print("1. Creating inflation factor (PC1 from CPI, PCE, PPI)...")
    try:
        inflation_df = create_inflation_factor(data_dir, start_date, end_date)
        print(f"   ✓ Created {len(inflation_df)} observations")
        print(f"   ✓ PCA explained variance: {inflation_df.attrs.get('pca_explained_variance_ratio', 0):.4f}")
        print(f"   ✓ Date range: {inflation_df['date'].min()} to {inflation_df['date'].max()}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        raise
    
    # 2. Create growth factor
    print("2. Creating growth factor (PC1 from IP, Retail Sales, Unemployment, GDP)...")
    try:
        growth_df = create_growth_factor(data_dir, start_date, end_date)
        print(f"   ✓ Created {len(growth_df)} observations")
        print(f"   ✓ PCA explained variance: {growth_df.attrs.get('pca_explained_variance_ratio', 0):.4f}")
        print(f"   ✓ Date range: {growth_df['date'].min()} to {growth_df['date'].max()}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        raise
    
    # 3. Create monetary policy factor
    print("3. Extracting monetary policy factor (10y-2y yield curve slope)...")
    try:
        monetary_df = create_monetary_policy_factor(data_dir, start_date, end_date)
        print(f"   ✓ Created {len(monetary_df)} observations")
        print(f"   ✓ Method: {monetary_df.attrs.get('method', 'N/A')}")
        print(f"   ✓ Date range: {monetary_df['date'].min()} to {monetary_df['date'].max()}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        raise
    
    # 4. Create market volatility factor
    print("4. Extracting market volatility factor (NFCI)...")
    try:
        volatility_df = create_market_volatility_factor(data_dir, start_date, end_date)
        print(f"   ✓ Created {len(volatility_df)} observations")
        print(f"   ✓ Method: {volatility_df.attrs.get('method', 'N/A')}")
        print(f"   ✓ Date range: {volatility_df['date'].min()} to {volatility_df['date'].max()}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        raise
    
    # 5. Merge all factors
    print("5. Merging all factors...")
    final_df = inflation_df[['date', 'inflation_factor']].copy()
    final_df = final_df.merge(growth_df[['date', 'growth_factor']], on='date', how='outer')
    final_df = final_df.merge(monetary_df[['date', 'monetary_policy_factor']], on='date', how='outer')
    final_df = final_df.merge(volatility_df[['date', 'market_volatility_factor']], on='date', how='outer')
    
    # Sort by date
    final_df = final_df.sort_values('date').reset_index(drop=True)
    
    # Filter to date range and ensure monthly frequency
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df[(final_df['date'] >= start_date) & (final_df['date'] <= end_date)].copy()
    
    # Create monthly date range and forward fill
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    monthly_df = pd.DataFrame({'date': monthly_dates})
    final_df = monthly_df.merge(final_df, on='date', how='left')
    
    # Forward fill missing values
    final_df = final_df.sort_values('date').fillna(method='ffill').fillna(method='bfill')
    
    print(f"   ✓ Final dataset: {len(final_df)} monthly observations")
    print(f"   ✓ Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"   ✓ Columns: {', '.join(final_df.columns)}\n")
    
    # 6. Save to CSV
    print(f"6. Saving to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved successfully!\n")
    
    # Print summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(final_df.describe())
    print("\n")
    
    # Store metadata
    final_df.attrs = {
        'inflation_pca_explained_variance': inflation_df.attrs.get('pca_explained_variance_ratio', 0),
        'growth_pca_explained_variance': growth_df.attrs.get('pca_explained_variance_ratio', 0),
        'monetary_policy_method': monetary_df.attrs.get('method', 'N/A'),
        'market_volatility_method': volatility_df.attrs.get('method', 'N/A'),
        'n_observations': len(final_df),
        'date_range': (final_df['date'].min(), final_df['date'].max()),
        'created_at': datetime.now().isoformat()
    }
    
    return final_df


if __name__ == "__main__":
    # Set paths
    script_dir = Path(__file__).parent
    # data_dir should point to main_project/data (parent of macro_processed)
    data_dir = script_dir.parent  # main_project/data
    output_file = script_dir / 'final_macro.csv'
    
    # Create final macro dataset
    final_df = create_final_macro(
        data_dir=data_dir,
        output_file=output_file,
        start_date='1989-01-01',
        end_date='2025-12-31'
    )
    
    print("=" * 80)
    print("✓ Process completed successfully!")
    print(f"✓ Output file: {output_file}")
    print("=" * 80)

