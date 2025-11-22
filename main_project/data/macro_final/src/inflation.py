#!/usr/bin/env python3
"""
Inflation Factor Module

Creates a composite inflation factor using Principal Component Analysis (PCA)
on standardized CPI, PCE, and PPI month-over-month percentage changes.

Methodology:
- Standardize (z-score) pct_change_mom for CPI, PCE, PPI
- Run PCA and extract PC1 (first principal component)
- PC1 represents the common inflation trend across all three indices
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_inflation_data(data_dir: Path) -> pd.DataFrame:
    """
    Load CPI, PCE, and PPI processed data.
    
    Returns:
        DataFrame with columns: date, cpi_mom, pce_mom, ppi_mom
    """
    cpi_file = data_dir / 'macro_processed' / 'inflation' / 'cpi_processed.csv'
    pce_file = data_dir / 'macro_processed' / 'inflation' / 'PCE_price_index_processed.csv'
    ppi_file = data_dir / 'macro_processed' / 'inflation' / 'PPI_inflation_processed.csv'
    
    # Load data
    cpi = pd.read_csv(cpi_file)
    pce = pd.read_csv(pce_file)
    ppi = pd.read_csv(ppi_file)
    
    # Convert dates
    cpi['date'] = pd.to_datetime(cpi['date'])
    pce['date'] = pd.to_datetime(pce['date'])
    ppi['date'] = pd.to_datetime(ppi['date'])
    
    # Extract month-over-month changes
    cpi_monthly = cpi[['date', 'pct_change_mom']].rename(columns={'pct_change_mom': 'cpi_mom'})
    pce_monthly = pce[['date', 'pct_change_mom']].rename(columns={'pct_change_mom': 'pce_mom'})
    ppi_monthly = ppi[['date', 'pct_change_mom']].rename(columns={'pct_change_mom': 'ppi_mom'})
    
    # Merge all on date (outer join to get all dates)
    inflation_df = cpi_monthly.merge(pce_monthly, on='date', how='outer')
    inflation_df = inflation_df.merge(ppi_monthly, on='date', how='outer')
    
    # Sort by date
    inflation_df = inflation_df.sort_values('date').reset_index(drop=True)
    
    return inflation_df


def create_inflation_factor(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Create inflation composite factor using PCA.
    
    Args:
        data_dir: Path to data directory
        start_date: Start date for filtering (default: '1989-01-01')
        end_date: End date for filtering (default: '2025-12-31')
    
    Returns:
        DataFrame with columns: date, inflation_factor, cpi_mom, pce_mom, ppi_mom
    """
    # Load data
    df = load_inflation_data(data_dir)
    
    # Filter date range
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Prepare data for PCA
    inflation_vars = ['cpi_mom', 'pce_mom', 'ppi_mom']
    
    # Remove rows where all variables are NaN
    df_clean = df[['date'] + inflation_vars].copy()
    df_clean = df_clean.dropna(subset=inflation_vars, how='all')
    
    # For rows with some NaN, forward fill then backward fill
    df_clean[inflation_vars] = df_clean[inflation_vars].fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining rows with NaN
    df_clean = df_clean.dropna(subset=inflation_vars)
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after cleaning for inflation factor")
    
    # Standardize the variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[inflation_vars])
    
    # Run PCA
    pca = PCA(n_components=1)
    inflation_factor = pca.fit_transform(X_scaled)
    
    # Store results
    result_df = df_clean[['date']].copy()
    result_df['inflation_factor'] = inflation_factor.flatten()
    result_df['cpi_mom'] = df_clean['cpi_mom'].values
    result_df['pce_mom'] = df_clean['pce_mom'].values
    result_df['ppi_mom'] = df_clean['ppi_mom'].values
    
    # Store PCA statistics for reporting
    result_df.attrs = {
        'pca_explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
        'pca_components': pca.components_[0].tolist(),
        'n_observations': len(df_clean),
        'date_range': (df_clean['date'].min(), df_clean['date'].max())
    }
    
    return result_df


if __name__ == "__main__":
    # Test the module
    data_dir = Path(__file__).parent.parent.parent.parent
    result = create_inflation_factor(data_dir)
    print(f"Inflation factor created: {len(result)} observations")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"PCA explained variance: {result.attrs.get('pca_explained_variance_ratio', 'N/A'):.4f}")
    print(f"\nFirst few rows:")
    print(result.head())

