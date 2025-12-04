"""
Performance metrics for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute performance metrics for a strategy.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns indexed by date
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of performance metrics
    """
    if returns.empty or len(returns) == 0:
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'total_return': 1.0,
            'n_periods': 0
        }
    
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'total_return': 1.0,
            'n_periods': 0
        }
    
    # Annualized return
    n_periods = len(returns)
    total_return = (1 + returns).prod() - 1
    periods_per_year = 12  # Monthly data
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'total_return': total_return,
        'n_periods': n_periods
    }

