#!/usr/bin/env python3
"""
Economic Growth Factor Module

Creates an economic growth factor using Real GDP (interpolated to monthly frequency).
Industrial Production and Retail Sales are not used in the final factor.

Methodology:
- Interpolate quarterly Real GDP growth (pct_change_mom) to monthly frequency
- Use interpolated GDP as the "growth factor" for each month
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_interpolated_gdp(data_dir: Path) -> pd.DataFrame:
    """
    Load Real GDP growth data and interpolate to monthly frequency.

    Returns:
        DataFrame with columns: date, gdp_mom
    """
    gdp_file = data_dir / 'macro_processed' / 'ec_growth' / 'real_gdp_processed.csv'
    
    gdp = pd.read_csv(gdp_file)
    gdp['date'] = pd.to_datetime(gdp['date'])
    
    min_date = gdp['date'].min()
    max_date = gdp['date'].max()
    monthly_dates = pd.date_range(start=min_date, end=max_date, freq='MS')

    gdp_monthly = gdp.set_index('date')['pct_change_mom'].reindex(monthly_dates)
    gdp_monthly = gdp_monthly.interpolate(method='linear')
    gdp_monthly_df = pd.DataFrame({'date': monthly_dates, 'gdp_mom': gdp_monthly.values})

    return gdp_monthly_df


def create_growth_factor(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Create economic growth factor using interpolated monthly GDP only.

    Args:
        data_dir: Path to data directory
        start_date: Start date for filtering (default: '1989-01-01')
        end_date: End date for filtering (default: '2025-12-31')

    Returns:
        DataFrame with columns: date, growth_factor (==gdp_mom), gdp_mom
    """
    # Load interpolated GDP
    df = load_interpolated_gdp(data_dir)

    # Filter date range
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    # The "growth factor" is the interpolated monthly GDP growth
    df = df.sort_values('date').reset_index(drop=True)
    df['growth_factor'] = df['gdp_mom']

    # Add simple attributes for reporting
    df.attrs = {
        'n_observations': len(df),
        'date_range': (df['date'].min(), df['date'].max())
    }

    return df


if __name__ == "__main__":
    # Test the module
    data_dir = Path(__file__).parent.parent.parent.parent
    result = create_growth_factor(data_dir)
    print(f"Growth factor created: {len(result)} observations")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"\nFirst few rows:")
    print(result.head())

