"""
Data preprocessing module for XGBoost forecasting.

Functions:
- load_data: Load macro and sentiment data
- create_lag_features: Create lagged features for macro variables
- create_sentiment_features: Create sentiment features (with optional lags)
- prepare_features: Combine all features for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler


def load_data(
    data_dir: Optional[Path] = None,
    include_sentiment: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load macro factors and sentiment data.
    
    Parameters:
    -----------
    data_dir : Path, optional
        Base data directory. If None, uses relative path from this file.
    include_sentiment : bool
        Whether to load sentiment data
    
    Returns:
    --------
    tuple
        (macro_df, sentiment_df)
        - macro_df: Macro factors with date index
        - sentiment_df: Sentiment scores (None if not available or not requested)
    """
    if data_dir is None:
        # From s22_ml_based/preprocessing.py: go up to main_project2
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir
    else:
        data_dir = Path(data_dir)
    
    # Load macro factors
    macro_path = data_dir / "data" / "macro_final" / "final_macro.csv"
    
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro data file not found: {macro_path}")
    
    macro_df = pd.read_csv(macro_path, parse_dates=["date"])
    macro_df = macro_df.set_index("date").sort_index()
    
    # Ensure we have the required columns
    required_cols = [
        'growth_factor',
        'inflation_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    missing_cols = [col for col in required_cols if col not in macro_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    macro_df = macro_df[required_cols].copy()
    macro_df = macro_df.dropna()
    
    print(f"Loaded macro data: {len(macro_df)} observations")
    print(f"Date range: {macro_df.index.min()} to {macro_df.index.max()}")
    
    # Load sentiment data if requested
    sentiment_df = None
    if include_sentiment:
        sentiment_path = data_dir / "data" / "news_data" / "sentiment_scores.csv"
        
        if sentiment_path.exists():
            try:
                sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
                sentiment_df = sentiment_df.set_index("date").sort_index()
                
                # Resample to monthly (take last value of month)
                sentiment_df = sentiment_df.resample("ME").last()
                
                # Select sentiment columns
                sentiment_cols = [
                    'inflation_sentiment',
                    'ec_growth_sentiment',
                    'monetary_policy_sentiment',
                    'market_vol_sentiment'
                ]
                
                available_cols = [col for col in sentiment_cols if col in sentiment_df.columns]
                if len(available_cols) < len(sentiment_cols):
                    missing = [col for col in sentiment_cols if col not in available_cols]
                    print(f"Warning: Missing sentiment columns: {missing}")
                
                sentiment_df = sentiment_df[available_cols].copy()
                sentiment_df = sentiment_df.dropna()
                
                print(f"Loaded sentiment data: {len(sentiment_df)} observations")
                print(f"Date range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")
            except Exception as e:
                print(f"Warning: Could not load sentiment data: {e}")
                sentiment_df = None
        else:
            print(f"Warning: Sentiment file not found at {sentiment_path}")
    
    return macro_df, sentiment_df


def create_lag_features(
    data: pd.DataFrame,
    variables: List[str],
    max_lags: int = 12
) -> pd.DataFrame:
    """
    Create lagged features for specified variables.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with variables as columns
    variables : list
        List of variable names to create lags for
    max_lags : int
        Maximum number of lags to create
    
    Returns:
    --------
    pd.DataFrame
        Data with lagged features added
    """
    lagged_data = data.copy()
    
    for var in variables:
        if var not in data.columns:
            continue
        
        for lag in range(1, max_lags + 1):
            lagged_data[f'{var}_lag{lag}'] = data[var].shift(lag)
    
    return lagged_data


def create_arima_features(
    data: pd.DataFrame,
    target_vars: List[str],
    ar_lags: int = 3,
    diff_order: int = 1,
    ma_window: int = 3
) -> pd.DataFrame:
    """
    Create ARIMA-like features: AR terms, differencing, and MA-like features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    target_vars : list
        Target variable names (e.g., ['growth_factor', 'inflation_factor'])
    ar_lags : int
        Number of AR (autoregressive) lags to include
    diff_order : int
        Order of differencing (1 for first differences, 2 for second)
    ma_window : int
        Window size for moving average features
    
    Returns:
    --------
    pd.DataFrame
        Data with ARIMA features added
    """
    arima_data = data.copy()
    
    for var in target_vars:
        if var not in data.columns:
            continue
        
        # AR terms: explicit lagged target values
        for lag in range(1, ar_lags + 1):
            arima_data[f'{var}_ar{lag}'] = data[var].shift(lag)
        
        # Differencing: first differences (I component)
        if diff_order >= 1:
            arima_data[f'{var}_diff1'] = data[var].diff(1)
            # Second differences if requested
            if diff_order >= 2:
                arima_data[f'{var}_diff2'] = arima_data[f'{var}_diff1'].diff(1)
        
        # MA-like features: rolling mean of the target
        arima_data[f'{var}_ma{ma_window}'] = data[var].rolling(window=ma_window).mean()
        
        # Trend features: cumulative sum of differences (integrated component)
        arima_data[f'{var}_trend'] = data[var].diff(1).cumsum()
        
        # Volatility features: rolling std of differences
        arima_data[f'{var}_vol'] = data[var].diff(1).rolling(window=ma_window).std()
    
    return arima_data


def create_sentiment_features(
    sentiment_df: pd.DataFrame,
    max_lags: int = 3
) -> pd.DataFrame:
    """
    Create sentiment features with optional lags.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        Sentiment data
    max_lags : int
        Maximum number of lags for sentiment (default: 3 for short lags)
    
    Returns:
    --------
    pd.DataFrame
        Sentiment features with lags
    """
    sentiment_features = sentiment_df.copy()
    
    # Create lags for each sentiment variable
    for col in sentiment_df.columns:
        for lag in range(1, max_lags + 1):
            sentiment_features[f'{col}_lag{lag}'] = sentiment_df[col].shift(lag)
    
    return sentiment_features


def prepare_features(
    macro_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    macro_lags: int = 12,
    sentiment_lags: int = 3,
    include_sentiment: bool = True,
    include_arima: bool = True,
    ar_lags: int = 3
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for model training with ARIMA-like features.
    
    Parameters:
    -----------
    macro_df : pd.DataFrame
        Macro factors data
    sentiment_df : pd.DataFrame, optional
        Sentiment data
    macro_lags : int
        Number of lags for macro variables
    sentiment_lags : int
        Number of lags for sentiment variables
    include_sentiment : bool
        Whether to include sentiment features
    include_arima : bool
        Whether to include ARIMA-like features (AR, differencing, MA)
    ar_lags : int
        Number of AR lags for target variables
    
    Returns:
    --------
    tuple
        (feature_df, feature_names)
        - feature_df: DataFrame with all features
        - feature_names: List of feature names
    """
    # Create macro lag features
    macro_vars = [
        'growth_factor',
        'inflation_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    feature_df = create_lag_features(macro_df, macro_vars, max_lags=macro_lags)
    
    # Add ARIMA-like features for target variables
    if include_arima:
        target_vars = ['growth_factor', 'inflation_factor']
        arima_features = create_arima_features(
            macro_df,
            target_vars,
            ar_lags=ar_lags,
            diff_order=1,
            ma_window=3
        )
        # Add only the ARIMA features (not the original targets)
        arima_cols = [col for col in arima_features.columns 
                     if col not in macro_df.columns or col.endswith(('_ar1', '_ar2', '_ar3', '_diff1', '_ma3', '_trend', '_vol'))]
        for col in arima_cols:
            if col in arima_features.columns:
                feature_df[col] = arima_features[col]
    
    # Add sentiment features if available
    if include_sentiment and sentiment_df is not None:
        # Align sentiment data with macro data dates first
        # Resample sentiment to match macro frequency if needed
        sentiment_aligned = sentiment_df.reindex(macro_df.index, method='ffill')
        
        # Create sentiment features with lags
        sentiment_features = create_sentiment_features(sentiment_aligned, max_lags=sentiment_lags)
        
        # Merge sentiment features (use inner join to keep only overlapping dates)
        # This ensures we have both macro and sentiment data
        feature_df = feature_df.join(sentiment_features, how='inner')
    else:
        # For macro-only, just use macro features
        pass
    
    # Drop rows with any NaN (from lagging)
    feature_df = feature_df.dropna()
    
    # Get feature names (exclude target variables)
    feature_names = [col for col in feature_df.columns 
                    if col not in macro_vars and not col.startswith('date')]
    
    print(f"  Feature matrix shape: {feature_df.shape}")
    print(f"  Date range: {feature_df.index.min()} to {feature_df.index.max()}")
    if include_arima:
        print(f"  ARIMA features included: AR lags={ar_lags}, differencing, MA features")
    
    return feature_df, feature_names


def create_targets(
    data: pd.DataFrame,
    target_var: str,
    horizons: List[int] = [1, 3, 6]
) -> pd.DataFrame:
    """
    Create target variables for different forecast horizons.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with target variable
    target_var : str
        Name of target variable (e.g., 'growth_factor', 'inflation_factor')
    horizons : list
        Forecast horizons in months
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with target columns for each horizon
    """
    if target_var not in data.columns:
        raise ValueError(f"Target variable {target_var} not found in data")
    
    targets = pd.DataFrame(index=data.index)
    
    for h in horizons:
        targets[f'target_h{h}'] = data[target_var].shift(-h)
    
    return targets


def prepare_train_test_split(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    train_split: float = 0.65
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Parameters:
    -----------
    feature_df : pd.DataFrame
        Feature matrix
    target_df : pd.DataFrame
        Target variables
    train_split : float
        Fraction of data for training
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Align indices
    common_idx = feature_df.index.intersection(target_df.index)
    feature_df = feature_df.loc[common_idx]
    target_df = target_df.loc[common_idx]
    
    # Sort by date
    feature_df = feature_df.sort_index()
    target_df = target_df.sort_index()
    
    # Split
    n_total = len(feature_df)
    n_train = int(n_total * train_split)
    
    X_train = feature_df.iloc[:n_train]
    X_test = feature_df.iloc[n_train:]
    y_train = target_df.iloc[:n_train]
    y_test = target_df.iloc[n_train:]
    
    print(f"\nTrain/test split:")
    print(f"  Training: {len(X_train)} observations ({X_train.index.min()} to {X_train.index.max()})")
    print(f"  Test: {len(X_test)} observations ({X_test.index.min()} to {X_test.index.max()})")
    
    return X_train, X_test, y_train, y_test

