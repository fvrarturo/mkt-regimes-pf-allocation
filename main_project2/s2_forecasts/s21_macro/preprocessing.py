"""
Data preprocessing module for TVP-VAR forecasting.

Functions:
- load_macro_data: Load macro factors from final_macro.csv
- prepare_forecast_data: Prepare data for forecasting with train/test split
- select_lag_order: Select optimal lag order using AIC/BIC
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from statsmodels.tsa.vector_ar.var_model import VAR


def load_macro_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load macro factors data.
    
    Parameters:
    -----------
    data_dir : Path, optional
        Base data directory. If None, uses relative path from this file.
    
    Returns:
    --------
    pd.DataFrame
        Macro data with date index and columns:
        - growth_factor
        - inflation_factor
        - monetary_policy_factor
        - market_volatility_factor
    """
    if data_dir is None:
        # From s21_macro/preprocessing.py: go up to main_project2
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "data" / "macro_final"
    else:
        data_dir = Path(data_dir)
    
    macro_path = data_dir / "final_macro.csv"
    
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro data file not found: {macro_path}")
    
    # Load data
    df = pd.read_csv(macro_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    
    # Ensure we have the required columns
    required_cols = [
        'growth_factor',
        'inflation_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select only the required columns
    df = df[required_cols].copy()
    
    # Drop any rows with NaN
    df = df.dropna()
    
    print(f"Loaded macro data: {len(df)} observations")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Variables: {list(df.columns)}")
    
    return df


def prepare_forecast_data(
    df: pd.DataFrame,
    train_split: float = 0.65,
    horizons: list = [1, 3, 6]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """
    Prepare data for forecasting with train/test split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Macro data with date index
    train_split : float
        Fraction of data to use for training (default 0.65 = 65%)
    horizons : list
        List of forecast horizons in months
    
    Returns:
    --------
    tuple
        (train_data, test_data, full_data, test_dates)
        - train_data: Training period data
        - test_data: Test period data
        - full_data: Full dataset (for reference)
        - test_dates: DatetimeIndex of test period dates
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate split point
    n_total = len(df)
    n_train = int(n_total * train_split)
    
    train_data = df.iloc[:n_train].copy()
    test_data = df.iloc[n_train:].copy()
    
    # Get test period dates (excluding last max(horizons) dates for forecast evaluation)
    max_horizon = max(horizons)
    test_dates = test_data.index[:-max_horizon] if len(test_data) > max_horizon else test_data.index
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} observations ({train_data.index.min()} to {train_data.index.max()})")
    print(f"  Test: {len(test_data)} observations ({test_data.index.min()} to {test_data.index.max()})")
    print(f"  Test forecast origins: {len(test_dates)} dates")
    
    return train_data, test_data, df, test_dates


def select_lag_order(
    data: pd.DataFrame,
    max_lags: int = 3,
    ic: str = 'bic'
) -> int:
    """
    Select optimal lag order using information criteria.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (variables as columns)
    max_lags : int
        Maximum number of lags to consider
    ic : str
        Information criterion: 'aic', 'bic', 'hqic', or 'fpe'
    
    Returns:
    --------
    int
        Optimal lag order
    """
    # Fit VAR models with different lag orders
    model = VAR(data)
    
    # Get IC values for different lag orders
    lag_results = model.select_order(maxlags=max_lags)
    
    # Extract IC values from ics dictionary
    ic_key = ic.lower()
    if ic_key not in ['aic', 'bic', 'hqic', 'fpe']:
        raise ValueError(f"Unknown information criterion: {ic}. Choose from 'aic', 'bic', 'hqic', 'fpe'")
    
    ic_values = lag_results.ics[ic_key]
    
    # Find optimal lag (minimum IC)
    # Note: ic_values[0] corresponds to lag 1, ic_values[1] to lag 2, etc.
    optimal_lag_idx = int(np.argmin(ic_values))
    optimal_lag = optimal_lag_idx + 1  # Convert to actual lag order (1-indexed)
    
    print(f"\nLag order selection ({ic.upper()}):")
    for lag in range(1, max_lags + 1):
        ic_val = ic_values[lag - 1]
        marker = " <--" if lag == optimal_lag else ""
        print(f"  Lag {lag}: {ic_val:.4f}{marker}")
    print(f"\nSelected lag order: {optimal_lag}")
    
    return optimal_lag

