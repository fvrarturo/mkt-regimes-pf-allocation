"""
Extremeness-based strategy implementation (Section 6.2).

Uses extremeness indicators to cut risk under extreme macro stress.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from data_loader import load_market_data, load_extremeness_scores
from performance import compute_performance_metrics, compute_turnover, compute_crash_avoidance


def extremeness_binary_strategy(
    extremeness: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    threshold_percentile: float = 90.0,
    risk_on_weight: float = 0.6,
    risk_off_weight: float = 0.2
) -> Dict:
    """
    Implement binary extremeness-based strategy (Option A).
    
    Parameters:
    -----------
    extremeness : pd.Series
        Extremeness scores
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    threshold_percentile : float
        Percentile threshold for extreme states (default: 90)
    risk_on_weight : float
        Equity weight in normal states (default: 0.6)
    risk_off_weight : float
        Equity weight in extreme states (default: 0.2)
    
    Returns:
    --------
    dict
        Strategy results
    """
    # Compute threshold
    threshold = extremeness.quantile(threshold_percentile / 100.0)
    
    # Binary weights
    equity_weights = pd.Series(
        np.where(extremeness > threshold, risk_off_weight, risk_on_weight),
        index=extremeness.index
    )
    
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
    
    # Crash avoidance
    crash_metrics = compute_crash_avoidance(equity_weights, strategy_returns)
    metrics.update(crash_metrics)
    
    return {
        'weights': equity_weights,
        'returns': strategy_returns,
        'metrics': metrics,
        'threshold': threshold,
        'threshold_percentile': threshold_percentile
    }


def extremeness_regime_combined_strategy(
    extremeness: pd.Series,
    regime_weights: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    threshold_percentile: float = 90.0,
    low_risk_weight: float = 0.2
) -> Dict:
    """
    Implement extremeness + regime combined strategy (Option B).
    
    Parameters:
    -----------
    extremeness : pd.Series
        Extremeness scores
    regime_weights : pd.Series
        Base weights from regime strategy
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    threshold_percentile : float
        Percentile threshold for extreme states
    low_risk_weight : float
        Equity weight when extremeness is high
    
    Returns:
    --------
    dict
        Strategy results
    """
    # Compute threshold
    threshold = extremeness.quantile(threshold_percentile / 100.0)
    
    # Combine: use regime weights normally, override with low risk when extreme
    extreme_flag = extremeness > threshold
    
    equity_weights = pd.Series(index=regime_weights.index, dtype=float)
    for idx in regime_weights.index:
        if idx in extreme_flag.index and extreme_flag.loc[idx]:
            equity_weights.loc[idx] = low_risk_weight
        else:
            equity_weights.loc[idx] = regime_weights.loc[idx] if idx in regime_weights.index else 0.5
    
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
    
    # Crash avoidance
    crash_metrics = compute_crash_avoidance(equity_weights, strategy_returns)
    metrics.update(crash_metrics)
    
    return {
        'weights': equity_weights,
        'returns': strategy_returns,
        'metrics': metrics,
        'threshold': threshold,
        'threshold_percentile': threshold_percentile
    }

