"""
Strategy Construction Module
Constructs regime-based portfolio trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict


# Default allocation strategy based on regimes
DEFAULT_ALLOCATIONS = {
    'HG_HI': {'stocks': 0.60, 'bonds': 0.40},
    'HG_LI': {'stocks': 0.70, 'bonds': 0.30},
    'LG_HI': {'stocks': 0.35, 'bonds': 0.65},
    'LG_LI': {'stocks': 0.45, 'bonds': 0.55}
}


def construct_strategy(
    df: pd.DataFrame,
    stock_df: pd.DataFrame,
    bond_df: pd.DataFrame,
    allocations: Dict[str, Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Construct regime-based portfolio trading strategy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with regime assignments
    stock_df : pd.DataFrame
        Stock returns DataFrame
    bond_df : pd.DataFrame
        Bond returns DataFrame
    allocations : Dict, optional
        Allocation dictionary by regime
    
    Returns:
    --------
    pd.DataFrame
        Strategy DataFrame with returns
    """
    if allocations is None:
        allocations = DEFAULT_ALLOCATIONS
    
    df_strategy = df.copy()
    
    # Assign allocations based on LAGGED regime (T+2)
    df_strategy['stock_allocation'] = df_strategy['regime_lagged'].map(
        lambda x: allocations[x]['stocks'] if pd.notna(x) else np.nan
    )
    df_strategy['bond_allocation'] = df_strategy['regime_lagged'].map(
        lambda x: allocations[x]['bonds'] if pd.notna(x) else np.nan
    )
    
    # Remove rows with NaN allocations
    df_strategy = df_strategy.dropna(subset=['stock_allocation', 'bond_allocation']).copy()
    
    # For each date in df (start of month), find the end of month date
    df_strategy['end_of_month'] = df_strategy['date'] + pd.offsets.MonthEnd(0)
    
    # Merge returns into strategy dataframe using end_of_month dates
    df_strategy = df_strategy.merge(
        stock_df[['date', 'pct_change_mom']].rename(
            columns={'date': 'end_of_month', 'pct_change_mom': 'stock_return'}
        ),
        on='end_of_month',
        how='left'
    )
    df_strategy = df_strategy.merge(
        bond_df[['date', 'first_diff']].rename(
            columns={'date': 'end_of_month', 'first_diff': 'bond_return'}
        ),
        on='end_of_month',
        how='left'
    )
    
    # Remove any rows where returns are missing
    df_strategy = df_strategy.dropna(subset=['stock_return', 'bond_return'])
    
    # Calculate regime-based portfolio returns
    df_strategy['regime_portfolio_return'] = (
        df_strategy['stock_allocation'] * df_strategy['stock_return'] +
        df_strategy['bond_allocation'] * df_strategy['bond_return']
    )
    
    # Calculate 50/50 benchmark portfolio returns
    df_strategy['benchmark_return'] = (
        0.50 * df_strategy['stock_return'] + 0.50 * df_strategy['bond_return']
    )
    
    return df_strategy

