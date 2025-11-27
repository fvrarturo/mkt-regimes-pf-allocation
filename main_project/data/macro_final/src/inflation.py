#!/usr/bin/env python3
"""
Inflation Factor Module

Creates an inflation factor from PCE's month-over-month percentage change (no PCA).
The inflation factor is simply the PCE MoM % change series.

Methodology:
- Extract pct_change_mom for PCE
- Use this as the inflation factor
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_pce_inflation_data(data_dir: Path) -> pd.DataFrame:
    """
    Load PCE processed data.

    Returns:
        DataFrame with columns: date, pce_mom
    """
    pce_file = data_dir / 'macro_processed' / 'inflation' / 'PCE_price_index_processed.csv'

    # Load data
    pce = pd.read_csv(pce_file)
    pce['date'] = pd.to_datetime(pce['date'])

    # Only keep date and pct_change_mom
    pce_monthly = pce[['date', 'pct_change_mom']].rename(columns={'pct_change_mom': 'pce_mom'})

    return pce_monthly


def create_inflation_factor(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Create inflation factor using PCE MoM % change only.

    Args:
        data_dir: Path to data directory
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        DataFrame with columns: date, inflation_factor (=pce_mom), pce_mom
    """
    df = load_pce_inflation_data(data_dir)

    # Filter date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    # Drop rows where pce_mom is missing
    df = df.dropna(subset=['pce_mom'])

    if len(df) == 0:
        raise ValueError("No valid data after filtering for inflation factor (PCE MoM)")

    # Use PCE MoM as the inflation factor (no transformation)
    result_df = df[['date', 'pce_mom']].copy()
    result_df['inflation_factor'] = result_df['pce_mom']

    # Rearrange columns: date, inflation_factor, pce_mom
    result_df = result_df[['date', 'inflation_factor', 'pce_mom']]

    # Store stats for reporting (mock keys for compatibility)
    result_df.attrs = {
        'method': 'pce_mom_only',
        'n_observations': len(result_df),
        'date_range': (result_df['date'].min(), result_df['date'].max())
    }

    return result_df


if __name__ == "__main__":
    # Test the module
    data_dir = Path(__file__).parent.parent.parent.parent
    result = create_inflation_factor(data_dir)
    print(f"Inflation factor created: {len(result)} observations")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"\nFirst few rows:")
    print(result.head())

