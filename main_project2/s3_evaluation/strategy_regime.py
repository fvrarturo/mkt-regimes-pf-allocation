"""
Regime-based strategy implementation (Section 6.1).

Uses HMM regime probabilities to determine equity allocation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from data_loader import load_market_data, load_regime_probabilities
from performance import compute_performance_metrics, compute_turnover


def compute_regime_weights(
    regime_probs: pd.DataFrame,
    regime_weights: Dict[int, float]
) -> pd.Series:
    """
    Compute probability-weighted equity weights.
    
    Parameters:
    -----------
    regime_probs : pd.DataFrame
        Regime probabilities (columns: prob_R0, prob_R1, prob_R2, prob_R3)
    regime_weights : dict
        Equity weight for each regime {0: 0.7, 1: 0.5, 2: 0.3, 3: 0.2}
    
    Returns:
    --------
    pd.Series
        Equity weights over time
    """
    weights = pd.Series(index=regime_probs.index, dtype=float)
    
    # Get probability columns
    prob_cols = [col for col in regime_probs.columns if col.startswith('prob_R')]
    
    for idx in regime_probs.index:
        weight = 0.0
        for regime_num, regime_weight in regime_weights.items():
            # Find probability column for this regime
            prob_col = None
            for col in prob_cols:
                if f'R{regime_num}' in col:
                    prob_col = col
                    break
            
            if prob_col and prob_col in regime_probs.columns:
                prob = regime_probs.loc[idx, prob_col]
                weight += prob * regime_weight
        
        weights.loc[idx] = weight
    
    return weights


def compute_strategy_returns(
    equity_weights: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series
) -> pd.Series:
    """
    Compute strategy returns from weights and asset returns.
    
    Parameters:
    -----------
    equity_weights : pd.Series
        Equity weights over time
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    
    Returns:
    --------
    pd.Series
        Strategy returns
    """
    # Align all series
    aligned = pd.DataFrame({
        'weight': equity_weights,
        'equity_return': equity_returns,
        'bond_return': bond_returns
    }).dropna()
    
    # Compute portfolio return: w * r_equity + (1-w) * r_bond
    strategy_returns = (
        aligned['weight'] * aligned['equity_return'] +
        (1 - aligned['weight']) * aligned['bond_return']
    )
    
    return strategy_returns


def regime_strategy(
    regime_probs: pd.DataFrame,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    regime_weights: Optional[Dict[int, float]] = None
) -> Dict:
    """
    Implement regime-based strategy.
    
    Parameters:
    -----------
    regime_probs : pd.DataFrame
        Regime probabilities
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    regime_weights : dict, optional
        Equity weights by regime. Default: {0: 0.7, 1: 0.5, 2: 0.3, 3: 0.2}
    
    Returns:
    --------
    dict
        Dictionary with strategy results:
        - weights: Equity weights over time
        - returns: Strategy returns
        - metrics: Performance metrics
    """
    # Default regime weights (expansion, neutral, risk-off, crisis)
    if regime_weights is None:
        regime_weights = {
            0: 0.7,  # Expansion
            1: 0.5,  # Neutral
            2: 0.3,  # Risk-off
            3: 0.2   # Crisis
        }
    
    # Compute equity weights
    equity_weights = compute_regime_weights(regime_probs, regime_weights)
    
    # Compute strategy returns
    strategy_returns = compute_strategy_returns(equity_weights, equity_returns, bond_returns)
    
    # Compute performance metrics
    metrics = compute_performance_metrics(strategy_returns)
    metrics['turnover'] = compute_turnover(equity_weights)
    
    return {
        'weights': equity_weights,
        'returns': strategy_returns,
        'metrics': metrics,
        'regime_weights': regime_weights
    }

