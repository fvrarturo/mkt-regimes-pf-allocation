"""
Ensemble strategy implementation (Section 6.4).

Combines forecast, regime, and extremeness signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from data_loader import load_market_data
from performance import compute_performance_metrics, compute_turnover


def ensemble_strategy(
    forecast_erp: pd.Series,
    regime_weights: pd.Series,
    extremeness: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    extremeness_threshold_percentile: float = 90.0,
    low_weight: float = 0.2,
    high_weight: float = 0.7,
    neutral_weight: float = 0.5
) -> Dict:
    """
    Implement ensemble strategy combining forecast, regime, and extremeness.
    
    Rule:
    - Low weight if: forecast ERP < 0 OR extremeness > threshold
    - High weight if: forecast ERP > 0 AND regime = expansion
    - Neutral otherwise
    
    Parameters:
    -----------
    forecast_erp : pd.Series
        Forecasted ERP
    regime_weights : pd.Series
        Base weights from regime strategy
    extremeness : pd.Series
        Extremeness scores
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    extremeness_threshold_percentile : float
        Extremeness threshold percentile
    low_weight : float
        Weight in low-risk state
    high_weight : float
        Weight in high-risk state
    neutral_weight : float
        Weight in neutral state
    
    Returns:
    --------
    dict
        Strategy results
    """
    # Compute extremeness threshold
    threshold = extremeness.quantile(extremeness_threshold_percentile / 100.0)
    
    # Determine expansion regime (regime with highest average weight)
    # Assume regime 0 is expansion (can be adjusted)
    is_expansion = regime_weights > 0.6  # High weight indicates expansion
    
    # Apply ensemble rule
    equity_weights = pd.Series(index=forecast_erp.index, dtype=float)
    
    for idx in forecast_erp.index:
        erp_forecast = forecast_erp.loc[idx] if idx in forecast_erp.index else 0
        is_extreme = extremeness.loc[idx] > threshold if idx in extremeness.index else False
        is_exp = is_expansion.loc[idx] if idx in is_expansion.index else False
        
        if erp_forecast < 0 or is_extreme:
            equity_weights.loc[idx] = low_weight
        elif erp_forecast > 0 and is_exp:
            equity_weights.loc[idx] = high_weight
        else:
            equity_weights.loc[idx] = neutral_weight
    
    # Compute strategy returns
    aligned = pd.DataFrame({
        'weight': equity_weights,
        'equity_return': equity_returns,
        'bond_return': bond_returns
    }).dropna()
    
    strategy_returns = (
        aligned['weight'] * aligned['equity_return'] +
        (1 - aligned['weight']) * aligned['bond_return']
    )
    
    # Compute metrics
    metrics = compute_performance_metrics(strategy_returns)
    metrics['turnover'] = compute_turnover(equity_weights)
    
    return {
        'weights': equity_weights,
        'returns': strategy_returns,
        'metrics': metrics
    }

