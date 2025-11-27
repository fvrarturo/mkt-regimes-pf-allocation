"""
Preprocessing for conditional extremeness analysis.

Loads regime assignments and extremeness scores, combines them for regression analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util

# Add parent directories to path
initial_relevance_dir = Path(__file__).parent.parent / "initial_relevance"
sys.path.insert(0, str(initial_relevance_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from initial_relevance preprocessing module
spec = importlib.util.spec_from_file_location(
    "initial_preprocessing", 
    initial_relevance_dir / "preprocessing.py"
)
initial_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initial_preprocessing)

from isolation_forest import run_isolation_forest_analysis
from pca_distance import run_pca_distance_analysis


def load_regime_data(regime_type='hmm_optimal'):
    """
    Load regime assignments from s12_regimeness.
    
    Parameters:
    -----------
    regime_type : str
        'hmm_optimal' or '2x2'
    
    Returns:
    --------
    pd.DataFrame
        Regime assignments with date index
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    
    if regime_type == 'hmm_optimal':
        # Use the optimal HMM model (Growth + Policy, 4 regimes)
        regime_path = base_dir / "s1_macro_vars" / "s12_regimeness" / "regimes" / "HMM_regimes" / "results_2vars_optimal" / "regime_assignments.csv"
    elif regime_type == '2x2':
        # Use simple 2x2 regimes
        regime_path = base_dir / "s1_macro_vars" / "s12_regimeness" / "regimes" / "2x2_regimes" / "results" / "regime_assignments.csv"
    else:
        raise ValueError(f"Unknown regime_type: {regime_type}")
    
    if not regime_path.exists():
        raise FileNotFoundError(f"Regime file not found: {regime_path}")
    
    regime_df = pd.read_csv(regime_path, parse_dates=["date"])
    regime_df = regime_df.set_index("date").sort_index()
    
    return regime_df


def load_extremeness_data():
    """
    Load or compute extremeness scores.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with extremeness scores and flags
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    
    # Try to load pre-computed extremeness results
    results_dir = base_dir / "s1_macro_vars" / "s13_extremeness" / "results"
    
    # Load data to compute extremeness if needed
    erp_df, macro_df, _ = initial_preprocessing.load_data()
    df = initial_preprocessing.prepare_data(erp_df, macro_df, None, include_sentiment=False)
    
    # Get feature columns
    macro_cols = initial_preprocessing.get_feature_columns(include_sentiment=False)
    
    # Run Isolation Forest to get extremeness scores
    print("Computing extremeness scores using Isolation Forest...")
    if_results = run_isolation_forest_analysis(
        df, macro_cols, contamination=0.1, threshold_percentile=90
    )
    
    # Extract extremeness data
    extremeness_df = if_results['results_df'][['extremeness', 'is_extreme', 'is_extreme_p90']].copy()
    
    return extremeness_df


def combine_regime_extremeness(regime_df, extremeness_df):
    """
    Combine regime assignments with extremeness data.
    
    Parameters:
    -----------
    regime_df : pd.DataFrame
        Regime assignments
    extremeness_df : pd.DataFrame
        Extremeness scores
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with regime and extremeness data
    """
    # Normalize dates to month-end for both dataframes
    regime_df_norm = regime_df.copy()
    if isinstance(regime_df_norm.index, pd.DatetimeIndex):
        regime_df_norm.index = regime_df_norm.index.to_period('M').to_timestamp('M')
    
    extremeness_df_norm = extremeness_df.copy()
    if isinstance(extremeness_df_norm.index, pd.DatetimeIndex):
        extremeness_df_norm.index = extremeness_df_norm.index.to_period('M').to_timestamp('M')
    
    # Merge on date index
    combined = regime_df_norm.join(extremeness_df_norm, how='inner')
    
    # Drop rows with missing data
    combined = combined.dropna()
    
    return combined


def create_extremeness_variables(df, method='binary'):
    """
    Create extremeness variables for regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe with extremeness column
    method : str
        'binary' (0/1) or 'continuous' (use extremeness score)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with extremeness variables added
    """
    df = df.copy()
    
    if method == 'binary':
        # Binary: 1 if extreme (90th percentile), 0 otherwise
        df['extreme'] = df['is_extreme_p90'].astype(int)
    elif method == 'continuous':
        # Continuous: use extremeness score directly
        df['extreme'] = df['extremeness']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def create_per_variable_extremeness(df):
    """
    Create per-variable extremeness flags (Task 2 from goals.md).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with macro variables
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with per-variable extremeness flags added
    """
    df = df.copy()
    
    # Macro variable columns
    macro_vars = ['growth_factor', 'inflation_factor', 'monetary_policy_factor', 'market_volatility_factor']
    
    for var in macro_vars:
        if var in df.columns:
            # Flag if variable > 90th percentile
            threshold = df[var].quantile(0.90)
            df[f'extreme_{var}'] = (df[var] > threshold).astype(int)
            
            # Also create z-score version (z > 1.5)
            z_score = (df[var] - df[var].mean()) / df[var].std()
            df[f'extreme_{var}_z'] = (z_score > 1.5).astype(int)
    
    return df

