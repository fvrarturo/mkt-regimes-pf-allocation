#!/usr/bin/env python3
"""
Economic Growth Factor Module

Creates a composite economic growth factor using Principal Component Analysis (PCA)
on standardized real-side indicators: Industrial Production, Retail Sales, 
Unemployment (inverted), and Real GDP (interpolated to monthly).

Methodology:
- Standardize (z-score) pct_change_mom for IP, Retail Sales
- Invert unemployment (so higher = better growth)
- Interpolate GDP to monthly frequency
- Run PCA and extract PC1 (first principal component)
- PC1 represents the common real activity trend
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_growth_data(data_dir: Path) -> pd.DataFrame:
    """
    Load economic growth indicators: IP, Retail Sales, Unemployment, GDP.
    
    Returns:
        DataFrame with columns: date, ip_mom, retail_sales_mom, unemployment, gdp
    """
    ip_file = data_dir / 'macro_processed' / 'ec_growth' / 'industrial_production_processed.csv'
    retail_file = data_dir / 'macro_processed' / 'ec_growth' / 'retail_sales_processed.csv'
    unemp_file = data_dir / 'macro_processed' / 'ec_growth' / 'unemployment_processed.csv'
    gdp_file = data_dir / 'macro_processed' / 'ec_growth' / 'real_gdp_processed.csv'
    
    # Load data
    ip = pd.read_csv(ip_file)
    retail = pd.read_csv(retail_file)
    unemp = pd.read_csv(unemp_file)
    gdp = pd.read_csv(gdp_file)
    
    # Convert dates
    ip['date'] = pd.to_datetime(ip['date'])
    retail['date'] = pd.to_datetime(retail['date'])
    unemp['date'] = pd.to_datetime(unemp['date'])
    gdp['date'] = pd.to_datetime(gdp['date'])
    
    # Extract month-over-month changes for IP and Retail Sales
    growth_df = pd.DataFrame({'date': ip['date']})
    growth_df['ip_mom'] = ip['pct_change_mom']
    growth_df['retail_sales_mom'] = retail['pct_change_mom']
    
    # For unemployment, use the level (we'll invert it)
    unemp_monthly = unemp[['date', 'value']].rename(columns={'value': 'unemployment'})
    growth_df = growth_df.merge(unemp_monthly, on='date', how='outer')
    
    # For GDP, we need to interpolate quarterly to monthly
    # First, create monthly date range
    min_date = min(ip['date'].min(), retail['date'].min(), unemp['date'].min(), gdp['date'].min())
    max_date = max(ip['date'].max(), retail['date'].max(), unemp['date'].max(), gdp['date'].max())
    monthly_dates = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    # Interpolate GDP to monthly
    gdp['date'] = pd.to_datetime(gdp['date'])
    gdp_monthly = gdp.set_index('date')['pct_change_mom'].reindex(monthly_dates)
    gdp_monthly = gdp_monthly.interpolate(method='linear')
    gdp_monthly_df = pd.DataFrame({'date': monthly_dates, 'gdp_mom': gdp_monthly.values})
    
    # Merge GDP
    growth_df = growth_df.merge(gdp_monthly_df, on='date', how='outer')
    
    # Sort by date
    growth_df = growth_df.sort_values('date').reset_index(drop=True)
    
    return growth_df


def create_growth_factor(data_dir: Path, start_date: str = '1989-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    """
    Create economic growth composite factor using PCA.
    
    Args:
        data_dir: Path to data directory
        start_date: Start date for filtering (default: '1989-01-01')
        end_date: End date for filtering (default: '2025-12-31')
    
    Returns:
        DataFrame with columns: date, growth_factor, ip_mom, retail_sales_mom, unemployment_inv, gdp_mom
    """
    # Load data
    df = load_growth_data(data_dir)
    
    # Filter date range
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Prepare variables for PCA
    # For unemployment, calculate month-over-month change (negative change = positive growth)
    # First forward/backward fill unemployment
    df['unemployment'] = df['unemployment'].fillna(method='ffill').fillna(method='bfill')
    # Calculate change in unemployment (inverted, so lower unemployment = growth)
    df['unemployment_change'] = -df['unemployment'].diff()
    
    # Prepare variables - use available ones
    # Check which variables are available
    available_vars = []
    if 'ip_mom' in df.columns:
        available_vars.append('ip_mom')
    if 'retail_sales_mom' in df.columns:
        available_vars.append('retail_sales_mom')
    if 'unemployment_change' in df.columns:
        available_vars.append('unemployment_change')
    if 'gdp_mom' in df.columns:
        available_vars.append('gdp_mom')
    
    if len(available_vars) < 2:
        raise ValueError(f"Need at least 2 growth variables, but only found: {available_vars}")
    
    # Create clean dataframe
    df_clean = df[['date'] + available_vars].copy()
    
    # Forward fill then backward fill for each variable
    for var in available_vars:
        df_clean[var] = df_clean[var].fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows where we still don't have at least 2 non-NaN values
    df_clean = df_clean.dropna(subset=available_vars, thresh=2)
    
    # For remaining NaN, fill with column median (more robust than mean)
    for var in available_vars:
        if df_clean[var].isna().any():
            df_clean[var] = df_clean[var].fillna(df_clean[var].median())
    
    # Final check - ensure we have data
    if len(df_clean) == 0:
        raise ValueError(f"No valid data after cleaning for growth factor. Available vars: {available_vars}")
    
    # Ensure no NaN remain
    if df_clean[available_vars].isna().any().any():
        # Last resort: fill with 0
        df_clean[available_vars] = df_clean[available_vars].fillna(0)
    
    # Standardize the variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[available_vars])
    
    # Run PCA
    pca = PCA(n_components=1)
    growth_factor = pca.fit_transform(X_scaled)
    
    # Store results
    result_df = df_clean[['date']].copy()
    result_df['growth_factor'] = growth_factor.flatten()
    
    # Store individual variables if available
    for var in available_vars:
        if var in df_clean.columns:
            result_df[var] = df_clean[var].values
    
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
    result = create_growth_factor(data_dir)
    print(f"Growth factor created: {len(result)} observations")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"PCA explained variance: {result.attrs.get('pca_explained_variance_ratio', 'N/A'):.4f}")
    print(f"\nFirst few rows:")
    print(result.head())

