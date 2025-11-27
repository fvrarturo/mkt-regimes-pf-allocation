"""
Data preprocessing module for LSTM forecasting.

Functions:
- create_sequences: Create sequences for LSTM input
- prepare_lstm_data: Prepare data for LSTM training
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler


def create_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    target_cols: List[str],
    horizons: List[int] = [1, 3, 6]
) -> Tuple[np.ndarray, dict]:
    """
    Create sequences for LSTM input.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with features and targets
    sequence_length : int
        Length of input sequences (e.g., 12 or 24 months)
    target_cols : list
        List of target column names (e.g., ['growth_factor', 'inflation_factor'])
    horizons : list
        Forecast horizons in months
    
    Returns:
    --------
    tuple
        (X, y_dict)
        - X: Input sequences (n_samples, sequence_length, n_features)
        - y_dict: Dictionary mapping horizon to target sequences
    """
    # Separate features and targets
    feature_cols = [col for col in data.columns if col not in target_cols]
    feature_data = data[feature_cols].values
    target_data = {col: data[col].values for col in target_cols}
    
    n_samples = len(data) - sequence_length - max(horizons) + 1
    
    if n_samples <= 0:
        raise ValueError(f"Not enough data for sequence_length={sequence_length} and max_horizon={max(horizons)}")
    
    # Create input sequences
    X = np.zeros((n_samples, sequence_length, len(feature_cols)))
    
    for i in range(n_samples):
        X[i] = feature_data[i:i + sequence_length]
    
    # Create target sequences for each horizon
    y_dict = {}
    for h in horizons:
        y_horizon = {}
        for col in target_cols:
            # Target is the value h steps ahead
            y_horizon[col] = target_data[col][sequence_length + h - 1:sequence_length + h - 1 + n_samples]
        y_dict[h] = y_horizon
    
    return X, y_dict


def prepare_lstm_data(
    macro_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    sequence_length: int = 12,
    horizons: List[int] = [1, 3, 6],
    include_sentiment: bool = True,
    train_split: float = 0.65,
    include_arima: bool = True,
    ar_lags: int = 3
) -> dict:
    """
    Prepare data for LSTM training.
    
    Parameters:
    -----------
    macro_df : pd.DataFrame
        Macro factors data
    sentiment_df : pd.DataFrame, optional
        Sentiment data
    sequence_length : int
        Length of input sequences
    horizons : list
        Forecast horizons
    include_sentiment : bool
        Whether to include sentiment features
    train_split : float
        Fraction of data for training
    
    Returns:
    --------
    dict
        Dictionary with prepared data:
        - X_train, X_test: Input sequences
        - y_train, y_test: Target sequences (dicts by horizon)
        - scaler: Fitted scaler for features
        - target_scalers: Fitted scalers for targets
        - feature_names: List of feature names
    """
    # Combine macro and sentiment
    if include_sentiment and sentiment_df is not None:
        # Align sentiment with macro dates
        sentiment_aligned = sentiment_df.reindex(macro_df.index, method='ffill')
        combined_df = macro_df.join(sentiment_aligned, how='inner')
    else:
        combined_df = macro_df.copy()
    
    # Drop NaN
    combined_df = combined_df.dropna()
    
    # Define targets
    target_cols = ['growth_factor', 'inflation_factor']
    
    # Add ARIMA-like features for LSTM
    if include_arima:
        # Create ARIMA features inline to avoid circular import
        for var in target_cols:
            if var not in combined_df.columns:
                continue
            
            # AR terms: explicit lagged target values (most important for dynamics)
            for lag in range(1, ar_lags + 1):
                combined_df[f'{var}_ar{lag}'] = combined_df[var].shift(lag)
            
            # Differencing: first differences (I component) - helps capture changes
            combined_df[f'{var}_diff1'] = combined_df[var].diff(1)
            # Also add second differences for stronger stationarity
            combined_df[f'{var}_diff2'] = combined_df[f'{var}_diff1'].diff(1)
            
            # MA-like features: rolling mean of the target
            combined_df[f'{var}_ma3'] = combined_df[var].rolling(window=3).mean()
            combined_df[f'{var}_ma6'] = combined_df[var].rolling(window=6).mean()
            
            # Trend features: cumulative sum of differences (integrated component)
            combined_df[f'{var}_trend'] = combined_df[var].diff(1).cumsum()
            
            # Volatility features: rolling std of differences
            combined_df[f'{var}_vol'] = combined_df[var].diff(1).rolling(window=3).std()
            
            # Momentum features: rate of change (handle division by zero)
            momentum = combined_df[var].pct_change(periods=1)
            combined_df[f'{var}_momentum'] = momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Acceleration: change in momentum
            accel = combined_df[f'{var}_momentum'].diff(1)
            combined_df[f'{var}_accel'] = accel.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Drop NaN again after differencing
    combined_df = combined_df.dropna()
    
    # Replace any remaining inf values with NaN, then fill
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Separate features and targets
    feature_cols = [col for col in combined_df.columns if col not in target_cols]
    feature_data = combined_df[feature_cols]
    target_data = combined_df[target_cols]
    
    # Scale features (handle any remaining issues)
    scaler = StandardScaler()
    feature_scaled = pd.DataFrame(
        scaler.fit_transform(feature_data),
        index=feature_data.index,
        columns=feature_data.columns
    )
    
    # Final check: replace any inf/nan in scaled features
    feature_scaled = feature_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale targets
    target_scalers = {}
    target_scaled_df = pd.DataFrame(index=target_data.index)
    for col in target_cols:
        target_scaler = StandardScaler()
        target_scaled_df[col] = target_scaler.fit_transform(target_data[[col]]).ravel()
        target_scalers[col] = target_scaler
    
    # Combine scaled features and targets
    data_scaled = feature_scaled.join(target_scaled_df, how='inner')
    
    # Split into train/test
    n_total = len(data_scaled)
    n_train = int(n_total * train_split)
    
    train_data = data_scaled.iloc[:n_train]
    test_data = data_scaled.iloc[n_train:]
    
    print(f"\nLSTM data preparation:")
    print(f"  Sequence length: {sequence_length} months")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training: {len(train_data)} observations ({train_data.index.min()} to {train_data.index.max()})")
    print(f"  Test: {len(test_data)} observations ({test_data.index.min()} to {test_data.index.max()})")
    
    # Create sequences for training
    X_train, y_train_dict = create_sequences(
        train_data,
        sequence_length,
        target_cols,
        horizons=horizons
    )
    
    # Create sequences for testing
    X_test, y_test_dict = create_sequences(
        test_data,
        sequence_length,
        target_cols,
        horizons=horizons
    )
    
    print(f"  Training sequences: {X_train.shape}")
    print(f"  Test sequences: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_dict,
        'y_test': y_test_dict,
        'scaler': scaler,
        'target_scalers': target_scalers,
        'feature_names': feature_cols,
        'target_names': target_cols,
        'sequence_length': sequence_length
    }

