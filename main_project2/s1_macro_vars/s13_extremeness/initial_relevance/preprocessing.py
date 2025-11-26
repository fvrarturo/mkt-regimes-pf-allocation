"""
Data preprocessing for extremeness models.

Loads and prepares macro, sentiment, and ERP data for extremeness analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load ERP, macro factors, and sentiment data.
    
    Returns:
    --------
    tuple
        (erp_df, macro_df, sentiment_df)
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    
    # Load ERP
    erp_path = base_dir / "data" / "macro_processed" / "equity_risk_pr.csv"
    erp_df = pd.read_csv(erp_path, parse_dates=["date"])
    erp_df = erp_df.set_index("date").sort_index()
    
    # Load macro factors
    macro_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    macro_df = pd.read_csv(macro_path, parse_dates=["date"])
    macro_df = macro_df.set_index("date").sort_index()
    
    # Load sentiment (optional)
    sentiment_path = base_dir / "data" / "news_data" / "sentiment_scores.csv"
    sentiment_df = None
    try:
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
        sentiment_df = sentiment_df.set_index("date").sort_index()
        # Resample sentiment to monthly (take last value)
        sentiment_df = sentiment_df.resample("ME").last()
    except FileNotFoundError:
        print("Warning: Sentiment data not found")
    
    return erp_df, macro_df, sentiment_df


def prepare_data(erp_df, macro_df, sentiment_df=None, include_sentiment=False):
    """
    Prepare and align data for extremeness models.
    
    Parameters:
    -----------
    erp_df : pd.DataFrame
        ERP data with 'ERP' column
    macro_df : pd.DataFrame
        Macro factors data
    sentiment_df : pd.DataFrame, optional
        Sentiment data
    include_sentiment : bool
        Whether to include sentiment variables
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with ERP and predictors
    """
    # Normalize macro dates to month-end to match ERP
    macro_df_normalized = macro_df.copy()
    if isinstance(macro_df_normalized.index, pd.DatetimeIndex):
        # Convert month-start to month-end
        macro_df_normalized.index = macro_df_normalized.index.to_period('M').to_timestamp('M')
    else:
        # If date is a column, convert it
        if 'date' in macro_df_normalized.columns:
            macro_df_normalized['date'] = pd.to_datetime(macro_df_normalized['date'])
            macro_df_normalized['date'] = macro_df_normalized['date'].dt.to_period('M').dt.to_timestamp('M')
            macro_df_normalized = macro_df_normalized.set_index('date')
    
    # Merge ERP and macro
    df = erp_df[["ERP"]].join(macro_df_normalized, how="inner")
    
    # Add sentiment if requested and available
    if include_sentiment and sentiment_df is not None:
        # Normalize sentiment dates to month-end
        sentiment_normalized = sentiment_df.copy()
        if isinstance(sentiment_normalized.index, pd.DatetimeIndex):
            sentiment_normalized.index = sentiment_normalized.index.to_period('M').to_timestamp('M')
        else:
            if 'date' in sentiment_normalized.columns:
                sentiment_normalized['date'] = pd.to_datetime(sentiment_normalized['date'])
                sentiment_normalized['date'] = sentiment_normalized['date'].dt.to_period('M').dt.to_timestamp('M')
                sentiment_normalized = sentiment_normalized.set_index('date')
        
        # Rename sentiment columns
        sentiment_cols = {
            'inflation_sentiment': 'sentiment_inflation',
            'ec_growth_sentiment': 'sentiment_growth',
            'monetary_policy_sentiment': 'sentiment_policy',
            'market_vol_sentiment': 'sentiment_volatility'
        }
        sentiment_renamed = sentiment_normalized.rename(columns=sentiment_cols)
        df = df.join(sentiment_renamed[list(sentiment_cols.values())], how="left")
    
    # Drop rows with any NaN
    df = df.dropna()
    
    return df


def get_feature_columns(include_sentiment=False):
    """
    Get list of feature column names.
    
    Parameters:
    -----------
    include_sentiment : bool
        Whether to include sentiment columns
    
    Returns:
    --------
    list
        List of feature column names
    """
    macro_vars = [
        'inflation_factor',
        'growth_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    if include_sentiment:
        sentiment_vars = [
            'sentiment_inflation',
            'sentiment_growth',
            'sentiment_policy',
            'sentiment_volatility'
        ]
        return macro_vars + sentiment_vars
    
    return macro_vars


def standardize_features(df, feature_cols):
    """
    Standardize features using z-scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with features
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    tuple
        (scaler, X_scaled) where X_scaled is numpy array
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return scaler, X_scaled

