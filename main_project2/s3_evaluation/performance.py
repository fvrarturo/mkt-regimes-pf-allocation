"""
Performance metrics computation for strategy evaluation.

Functions:
- compute_performance_metrics: Calculate Sharpe, volatility, drawdown, etc.
- compute_turnover: Calculate portfolio turnover
- compute_hit_rates: Calculate prediction hit rates
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_performance_metrics(
    returns: pd.Series,
    risk_free_rate: Optional[pd.Series] = None,
    periods_per_year: int = 12
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    risk_free_rate : pd.Series, optional
        Risk-free rate (for excess returns)
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {
            'annualized_return': np.nan,
            'annualized_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'calmar_ratio': np.nan,
            'total_return': np.nan,
            'n_periods': 0
        }
    
    # Excess returns if risk-free rate provided
    if risk_free_rate is not None:
        excess_returns = returns - risk_free_rate.reindex(returns.index, method='ffill').fillna(0)
    else:
        excess_returns = returns
    
    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    
    # Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    excess_mean = excess_returns.mean() * periods_per_year
    sharpe_ratio = excess_mean / annualized_volatility if annualized_volatility > 0 else np.nan
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else np.nan
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'total_return': total_return,
        'n_periods': len(returns)
    }


def compute_turnover(weights: pd.Series) -> float:
    """
    Compute portfolio turnover.
    
    Parameters:
    -----------
    weights : pd.Series
        Equity weights over time
    
    Returns:
    --------
    float
        Average absolute change in weights
    """
    if len(weights) < 2:
        return 0.0
    
    weight_changes = weights.diff().abs()
    return weight_changes.mean()


def compute_hit_rate(
    forecasts: pd.Series,
    actuals: pd.Series,
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Compute hit rate for sign prediction.
    
    Parameters:
    -----------
    forecasts : pd.Series
        Forecasted values
    actuals : pd.Series
        Actual values
    threshold : float
        Threshold for "positive" prediction
    
    Returns:
    --------
    dict
        Dictionary with hit rate statistics
    """
    # Align series
    aligned = pd.DataFrame({
        'forecast': forecasts,
        'actual': actuals
    }).dropna()
    
    if len(aligned) == 0:
        return {
            'hit_rate': np.nan,
            'n_observations': 0,
            'n_correct': 0
        }
    
    # Sign prediction hit rate
    forecast_positive = aligned['forecast'] > threshold
    actual_positive = aligned['actual'] > threshold
    sign_correct = (forecast_positive == actual_positive)
    
    hit_rate = sign_correct.mean()
    
    return {
        'hit_rate': hit_rate,
        'n_observations': len(aligned),
        'n_correct': sign_correct.sum()
    }


def compute_crash_avoidance(
    weights: pd.Series,
    returns: pd.Series,
    n_worst: int = 20
) -> Dict[str, float]:
    """
    Compute crash avoidance metrics.
    
    Parameters:
    -----------
    weights : pd.Series
        Equity weights over time
    returns : pd.Series
        Returns (to identify worst periods)
    n_worst : int
        Number of worst periods to analyze
    
    Returns:
    --------
    dict
        Dictionary with crash avoidance statistics
    """
    aligned = pd.DataFrame({
        'weight': weights,
        'return': returns
    }).dropna()
    
    if len(aligned) < n_worst:
        return {
            'avg_weight_during_crashes': np.nan,
            'avg_weight_during_normal': np.nan,
            'crash_avoidance_score': np.nan
        }
    
    # Identify worst periods
    worst_periods = aligned.nsmallest(n_worst, 'return')
    normal_periods = aligned.drop(worst_periods.index)
    
    avg_weight_crashes = worst_periods['weight'].mean()
    avg_weight_normal = normal_periods['weight'].mean()
    
    # Crash avoidance score: lower weight during crashes = better
    crash_avoidance_score = (avg_weight_normal - avg_weight_crashes) / avg_weight_normal if avg_weight_normal > 0 else np.nan
    
    return {
        'avg_weight_during_crashes': avg_weight_crashes,
        'avg_weight_during_normal': avg_weight_normal,
        'crash_avoidance_score': crash_avoidance_score
    }

