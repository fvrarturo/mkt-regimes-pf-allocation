"""
Data loading module for strategy evaluation.

Loads:
- Market returns (equity, bond, ERP)
- Regime probabilities
- Extremeness scores
- Forecast outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_market_data(base_dir: Optional[Path] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load market return data from S&P 500 and 3M yield files.
    
    Parameters:
    -----------
    base_dir : Path, optional
        Base directory. If None, uses main_project2.
    
    Returns:
    --------
    tuple
        (equity_returns, bond_returns, erp)
        - equity_returns: S&P 500 monthly returns
        - bond_returns: Risk-free (3M Treasury) monthly returns
        - erp: Equity Risk Premium (equity - bond)
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Load S&P 500 returns
    sp500_path = base_dir / "data" / "macro_processed" / "sp500_processed.csv"
    if not sp500_path.exists():
        raise FileNotFoundError(f"S&P 500 file not found: {sp500_path}")
    
    sp500 = pd.read_csv(sp500_path, parse_dates=["date"])
    sp500 = sp500.set_index("date").sort_index()
    
    # Extract monthly returns (convert percentage to decimal)
    equity_returns = sp500['pct_change_mom'] / 100.0
    equity_returns = equity_returns.resample('ME').last()
    equity_returns = equity_returns.dropna()
    
    # Load 3M Treasury yield
    yield_path = base_dir / "data" / "macro_processed" / "3m_yield_processed.csv"
    if not yield_path.exists():
        raise FileNotFoundError(f"3M yield file not found: {yield_path}")
    
    yield_3m = pd.read_csv(yield_path, parse_dates=["date"])
    yield_3m = yield_3m.set_index("date").sort_index()
    
    # Convert annual yield to monthly return
    # Annual yield / 100 / 12 = monthly return
    bond_returns = (yield_3m['value'] / 100.0) / 12.0
    bond_returns = bond_returns.resample('ME').last()
    bond_returns = bond_returns.dropna()
    
    # Compute ERP (align dates first)
    aligned = pd.DataFrame({
        'equity_return': equity_returns,
        'bond_return': bond_returns
    }).dropna()
    
    erp = aligned['equity_return'] - aligned['bond_return']
    
    # Align all series to common dates
    common_dates = aligned.index
    equity_returns = equity_returns.reindex(common_dates)
    bond_returns = bond_returns.reindex(common_dates)
    
    return equity_returns, bond_returns, erp


def load_regime_probabilities(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load HMM regime probabilities.
    
    Parameters:
    -----------
    base_dir : Path, optional
        Base directory
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with regime probabilities (columns: prob_R0, prob_R1, prob_R2, prob_R3)
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Search for regime files
    possible_paths = [
        base_dir / "s1_macro_vars" / "s12_regimeness" / "results" / "regime_assignments.csv",
        base_dir / "s1_macro_vars" / "s12_regimeness" / "regime_assignments.csv",
        base_dir / "data" / "regime_assignments.csv"
    ]
    
    for path in possible_paths:
        if path.exists():
            regime_df = pd.read_csv(path, parse_dates=["date"])
            regime_df = regime_df.set_index("date").sort_index()
            
            # Extract probability columns
            prob_cols = [col for col in regime_df.columns if col.startswith('prob_R')]
            if len(prob_cols) > 0:
                return regime_df[prob_cols + ['regime']] if 'regime' in regime_df.columns else regime_df[prob_cols]
    
    # If not found, return empty DataFrame
    print("Warning: Regime probabilities not found. Returning empty DataFrame.")
    return pd.DataFrame()


def load_extremeness_scores(base_dir: Optional[Path] = None) -> pd.Series:
    """
    Load extremeness scores.
    
    Parameters:
    -----------
    base_dir : Path, optional
        Base directory
    
    Returns:
    --------
    pd.Series
        Extremeness scores indexed by date
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Search for extremeness results
    possible_paths = [
        base_dir / "s1_macro_vars" / "s13_extremeness" / "results" / "*extremeness*.csv",
        base_dir / "s1_macro_vars" / "s13_extremeness" / "initial_relevance" / "results" / "*extremeness*.csv"
    ]
    
    # Try to find extremeness files
    extremeness_dir = base_dir / "s1_macro_vars" / "s13_extremeness" / "results"
    if extremeness_dir.exists():
        import glob
        files = glob.glob(str(extremeness_dir / "*extremeness*.csv"))
        if len(files) > 0:
            # Load the first available file
            df = pd.read_csv(files[0], parse_dates=["date"])
            df = df.set_index("date").sort_index()
            if 'extremeness' in df.columns:
                return df['extremeness']
    
    print("Warning: Extremeness scores not found. Returning empty Series.")
    return pd.Series(dtype=float)


def load_forecasts(base_dir: Optional[Path] = None) -> Dict[str, Dict[str, pd.Series]]:
    """
    Load forecast outputs from all models.
    
    Parameters:
    -----------
    base_dir : Path, optional
        Base directory
    
    Returns:
    --------
    dict
        Nested dictionary: forecasts[model_name][variable] = forecast_series
        Models: 'tvpvar', 'xgboost_macro', 'xgboost_sentiment', 'lstm'
        Variables: 'growth', 'inflation'
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    forecasts = {}
    
    # Note: Forecasts would need to be saved as time series
    # For now, return empty structure - forecasts would need to be loaded
    # from saved CSV files or regenerated
    
    return forecasts

