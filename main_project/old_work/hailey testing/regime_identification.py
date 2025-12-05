"""
Regime Identification Module
Identifies economic regimes based on growth and inflation factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def identify_regimes(
    df: pd.DataFrame,
    growth_col: str = 'growth_factor',
    inflation_col: str = 'inflation_factor',
    method: str = 'mode'
) -> Tuple[pd.DataFrame, float, float]:
    """
    Identify regimes based on growth and inflation thresholds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with growth and inflation factors
    growth_col : str
        Name of growth factor column
    inflation_col : str
        Name of inflation factor column
    method : str
        Method to determine thresholds ('mode', 'median', 'mean')
    
    Returns:
    --------
    Tuple of (DataFrame with regime column, growth_threshold, inflation_threshold)
    """
    df = df.copy()
    
    # Determine thresholds
    if method == 'mode':
        growth_mode = df[growth_col].mode()
        inflation_mode = df[inflation_col].mode()
        growth_threshold = growth_mode.iloc[0] if len(growth_mode) > 0 else df[growth_col].median()
        inflation_threshold = inflation_mode.iloc[0] if len(inflation_mode) > 0 else df[inflation_col].median()
    elif method == 'median':
        growth_threshold = df[growth_col].median()
        inflation_threshold = df[inflation_col].median()
    elif method == 'mean':
        growth_threshold = df[growth_col].mean()
        inflation_threshold = df[inflation_col].mean()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Growth Threshold ({method}): {growth_threshold:.4f}")
    print(f"Inflation Threshold ({method}): {inflation_threshold:.4f}")
    
    # Classify into 4 regimes
    def classify_regime(row):
        high_growth = row[growth_col] > growth_threshold
        high_inflation = row[inflation_col] > inflation_threshold
        
        if high_growth and high_inflation:
            return 'HG_HI'
        elif high_growth and not high_inflation:
            return 'HG_LI'
        elif not high_growth and high_inflation:
            return 'LG_HI'
        else:
            return 'LG_LI'
    
    df['regime'] = df.apply(classify_regime, axis=1)
    
    # Lag regime by 2 periods to remove forward bias
    df['regime_lagged'] = df['regime'].shift(2)
    
    print("\nRegime Classification Summary:")
    print(df['regime'].value_counts())
    print("\nRegime Lagged Classification Summary:")
    print(df['regime_lagged'].value_counts())
    
    return df, growth_threshold, inflation_threshold

