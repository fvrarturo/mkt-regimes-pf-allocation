"""
Forecast-based strategy implementation (Section 6.3).

Uses forecasting outputs to predict ERP and determine equity allocation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal
from pathlib import Path

from data_loader import load_market_data
from performance import compute_performance_metrics, compute_turnover, compute_hit_rate, compute_crash_avoidance


def forecast_to_weights_threshold(
    forecast_erp: pd.Series,
    risk_on_weight: float = 0.7,
    risk_off_weight: float = 0.3,
    threshold: float = 0.0
) -> pd.Series:
    """
    Convert forecast ERP to weights using threshold rule (Version A).
    
    Parameters:
    -----------
    forecast_erp : pd.Series
        Forecasted ERP
    risk_on_weight : float
        Weight when forecast > threshold
    risk_off_weight : float
        Weight when forecast <= threshold
    threshold : float
        Threshold for positive prediction
    
    Returns:
    --------
    pd.Series
        Equity weights
    """
    weights = pd.Series(
        np.where(forecast_erp > threshold, risk_on_weight, risk_off_weight),
        index=forecast_erp.index
    )
    return weights


def forecast_to_weights_linear(
    forecast_erp: pd.Series,
    w_min: float = 0.2,
    w_max: float = 0.8
) -> pd.Series:
    """
    Convert forecast ERP to weights using linear mapping (Version B).
    
    Parameters:
    -----------
    forecast_erp : pd.Series
        Forecasted ERP
    w_min : float
        Minimum equity weight
    w_max : float
        Maximum equity weight
    
    Returns:
    --------
    pd.Series
        Equity weights
    """
    # Compute percentiles for normalization
    erp_5th = forecast_erp.quantile(0.05)
    erp_95th = forecast_erp.quantile(0.95)
    
    # Linear mapping
    normalized = (forecast_erp - erp_5th) / (erp_95th - erp_5th) if erp_95th != erp_5th else 0
    weights = w_min + normalized * (w_max - w_min)
    
    # Clip to bounds
    weights = weights.clip(w_min, w_max)
    
    return weights


def forecast_to_weights_regime_conditioned(
    forecast_erp: pd.Series,
    regime_probs: pd.DataFrame,
    expansion_regime: int = 0,
    risk_off_regime: int = 2,
    risk_on_weight: float = 0.7,
    risk_off_weight: float = 0.3,
    neutral_weight: float = 0.5
) -> pd.Series:
    """
    Convert forecast ERP to weights using regime-conditioned rule (Version C).
    
    Parameters:
    -----------
    forecast_erp : pd.Series
        Forecasted ERP
    regime_probs : pd.DataFrame
        Regime probabilities
    expansion_regime : int
        Expansion regime number
    risk_off_regime : int
        Risk-off regime number
    risk_on_weight : float
        Weight when forecast > 0 AND expansion
    risk_off_weight : float
        Weight when forecast < 0 AND risk-off
    neutral_weight : float
        Weight otherwise
    
    Returns:
    --------
    pd.Series
        Equity weights
    """
    weights = pd.Series(index=forecast_erp.index, dtype=float)
    
    # Get regime probability columns
    prob_cols = {r: None for r in [expansion_regime, risk_off_regime]}
    for col in regime_probs.columns:
        if col.startswith('prob_R'):
            for r in prob_cols.keys():
                if f'R{r}' in col:
                    prob_cols[r] = col
    
    for idx in forecast_erp.index:
        erp_forecast = forecast_erp.loc[idx]
        
        # Get regime probabilities
        exp_prob = regime_probs.loc[idx, prob_cols[expansion_regime]] if prob_cols[expansion_regime] else 0
        risk_off_prob = regime_probs.loc[idx, prob_cols[risk_off_regime]] if prob_cols[risk_off_regime] else 0
        
        # Apply rule
        if erp_forecast > 0 and exp_prob > 0.5:
            weights.loc[idx] = risk_on_weight
        elif erp_forecast < 0 and risk_off_prob > 0.5:
            weights.loc[idx] = risk_off_weight
        else:
            weights.loc[idx] = neutral_weight
    
    return weights


def forecast_strategy(
    forecast_erp: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    actual_erp: Optional[pd.Series] = None,
    method: Literal['threshold', 'linear', 'regime_conditioned'] = 'threshold',
    regime_probs: Optional[pd.DataFrame] = None,
    **kwargs
) -> Dict:
    """
    Implement forecast-based strategy.
    
    Parameters:
    -----------
    forecast_erp : pd.Series
        Forecasted ERP (1-month ahead)
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    actual_erp : pd.Series, optional
        Actual ERP (for hit rate calculation)
    method : str
        Weight conversion method ('threshold', 'linear', 'regime_conditioned')
    regime_probs : pd.DataFrame, optional
        Regime probabilities (required for 'regime_conditioned' method)
    **kwargs
        Additional parameters for weight conversion
    
    Returns:
    --------
    dict
        Strategy results
    """
    # Convert forecast to weights
    if method == 'threshold':
        equity_weights = forecast_to_weights_threshold(forecast_erp, **kwargs)
    elif method == 'linear':
        equity_weights = forecast_to_weights_linear(forecast_erp, **kwargs)
    elif method == 'regime_conditioned':
        if regime_probs is None:
            raise ValueError("regime_probs required for regime_conditioned method")
        equity_weights = forecast_to_weights_regime_conditioned(
            forecast_erp, regime_probs, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
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
    
    # Hit rate if actual ERP provided
    if actual_erp is not None:
        hit_rate_metrics = compute_hit_rate(forecast_erp, actual_erp)
        metrics.update({f'hit_rate_{k}': v for k, v in hit_rate_metrics.items()})
    
    # Crash avoidance
    crash_metrics = compute_crash_avoidance(equity_weights, strategy_returns)
    metrics.update(crash_metrics)
    
    return {
        'weights': equity_weights,
        'returns': strategy_returns,
        'metrics': metrics,
        'method': method
    }

