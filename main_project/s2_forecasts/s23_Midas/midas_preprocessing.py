"""
Data preprocessing module for MIDAS TVP-VAR forecasting.

Functions:
- load_macro_data: Load 4 macro factors from final_macro.csv
- load_daily_oil_data: Load daily oil prices
- prepare_forecast_data: Prepare train/test split with MIDAS factors
- select_lag_order: Select optimal lag order using AIC/BIC
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from statsmodels.tsa.vector_ar.var_model import VAR
from midas_tvpvar_model import create_midas_oil_factor, prepare_midas_data


def load_macro_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load 4 macro factors.
    
    Returns:
    --------
    pd.DataFrame
        Macro data with columns: growth_factor, inflation_factor, 
        monetary_policy_factor, market_volatility_factor
    """
    if data_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "data" / "macro_final"
    else:
        data_dir = Path(data_dir)
    
    macro_path = data_dir / "final_macro.csv"
    
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro data file not found: {macro_path}")
    
    df = pd.read_csv(macro_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    
    required_cols = [
        'growth_factor',
        'inflation_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[required_cols].copy()
    df = df.dropna()
    
    print(f"Loaded macro data: {len(df)} observations")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def load_daily_oil_data(data_dir: Optional[Path] = None) -> pd.Series:
    """
    Load daily oil price data.
    
    Returns:
    --------
    pd.Series
        Daily oil prices with date index
    """
    if data_dir is None:
        # Oil data is in main_project, not main_project2
        base_dir = Path(__file__).parent.parent.parent.parent
        data_dir = base_dir / "main_project" / "data"
    else:
        data_dir = Path(data_dir)
    
    oil_path = data_dir / "macro_processed" / "daily_factors" / "daily_wti.csv"
    
    if not oil_path.exists():
        raise FileNotFoundError(f"Daily oil file not found: {oil_path}")
    
    df = pd.read_csv(oil_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    oil_prices = df['Close'].copy()
    oil_prices = oil_prices.ffill()  # Forward fill for missing values
    
    print(f"Loaded daily oil data: {len(oil_prices)} observations")
    print(f"Date range: {oil_prices.index.min()} to {oil_prices.index.max()}")
    
    return oil_prices


def prepare_midas_forecast_data(
    macro_df: pd.DataFrame,
    oil_prices: pd.Series,
    train_split: float = 0.65,
    horizons: list = [1, 3, 6],
    theta: float = 0.03,
    K: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """
    Prepare data for MIDAS TVP-VAR forecasting.
    
    Steps:
    1. Create monthly MIDAS oil factor from daily prices
    2. Merge with macro factors
    3. Split into train and test
    4. Exclude last max_horizon dates from test_dates (for forecast evaluation)
    
    Parameters:
    -----------
    macro_df : pd.DataFrame
        4 macro factors
    oil_prices : pd.Series
        Daily oil prices
    train_split : float
        Fraction for training (0-1)
    horizons : list
        Forecast horizons (for reference)
    theta : float
        MIDAS decay parameter
    K : int
        Number of daily lags in MIDAS aggregation
    
    Returns:
    --------
    tuple
        (train_data, test_data, full_data, test_dates)
    """
    print("\n" + "="*80)
    print("Preparing MIDAS forecast data")
    print("="*80)
    
    # Step 1: Create oil MIDAS factor
    print(f"\nStep 1: Creating MIDAS oil factor")
    print(f"  Decay parameter (theta): {theta}")
    print(f"  Number of daily lags (K): {K}")
    oil_midas = create_midas_oil_factor(oil_prices, theta=theta, K=K)
    
    # Step 2: Merge
    print(f"\nStep 2: Merging macro factors with oil MIDAS")
    full_data = prepare_midas_data(macro_df, oil_midas)
    
    # Step 3: Split
    print(f"\nStep 3: Splitting into train ({int(train_split*100)}%) and test")
    n_total = len(full_data)
    n_train = int(n_total * train_split)
    
    train_data = full_data.iloc[:n_train].copy()
    test_data = full_data.iloc[n_train:].copy()
    
    # Step 4: Get test_dates, excluding last max_horizon for forecast evaluation
    max_horizon = max(horizons)
    test_dates = test_data.index[:-max_horizon] if len(test_data) > max_horizon else test_data.index
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} observations ({train_data.index.min()} to {train_data.index.max()})")
    print(f"  Test: {len(test_data)} observations ({test_data.index.min()} to {test_data.index.max()})")
    print(f"  Test forecast origins: {len(test_dates)} dates (excluding last {max_horizon} for evaluation)")
    
    return train_data, test_data, full_data, test_dates


def select_lag_order(
    df: pd.DataFrame,
    max_lags: int = 3,
    ic: str = 'bic'
) -> int:
    """
    Select optimal VAR lag order using information criteria.
    
    Uses VAR.select_order() to efficiently evaluate all lag orders at once.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    max_lags : int
        Maximum lags to consider
    ic : str
        Information criterion: 'bic' or 'aic'
    
    Returns:
    --------
    int
        Optimal lag order
    """
    print(f"\nSelecting optimal lag order (using {ic.upper()})...")
    
    var_model = VAR(df)
    lag_results = var_model.select_order(maxlags=max_lags)
    
    # Extract IC values
    ic_key = ic.lower()
    ic_values = lag_results.ics[ic_key]
    
    # Find optimal lag (minimum IC value)
    optimal_lag = int(np.argmin(ic_values)) + 1  # +1 because ic_values[0] is lag 1
    
    print(f"  Lag order selection ({ic.upper()}):")
    for lag in range(1, max_lags + 1):
        ic_val = ic_values[lag - 1]
        marker = " <--" if lag == optimal_lag else ""
        print(f"    Lag {lag}: {ic_val:.4f}{marker}")
    
    print(f"  Selected optimal lag order: {optimal_lag}")
    
    return optimal_lag
