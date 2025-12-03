"""
Trading utilities: translate ERP forecasts into allocations and compute returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    name: str
    returns: pd.Series
    weights: pd.Series
    forecast: pd.Series


def forecast_to_weights(
    forecasts: pd.Series,
    min_weight: float = 0.1,
    max_weight: float = 0.9
) -> pd.Series:
    """
    Map ERP forecasts into equity weights using a scaled sigmoid rule.
    """
    forecasts = forecasts.dropna()
    if forecasts.empty:
        return forecasts

    std = forecasts.std()
    if std == 0 or np.isnan(std):
        scaled = forecasts * 0.0
    else:
        scaled = forecasts / std

    weights = 0.5 + 0.25 * scaled
    return weights.clip(min_weight, max_weight)


def run_trading_strategy(
    name: str,
    forecasts: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    min_weight: float = 0.1,
    max_weight: float = 0.9
) -> StrategyResult:
    """
    Compute strategy returns based on ERP forecasts and a simple weight rule.
    """
    weights = forecast_to_weights(forecasts, min_weight, max_weight)
    aligned = pd.DataFrame({
        "weight": weights,
        "equity": equity_returns,
        "bond": bond_returns
    }).dropna()

    if aligned.empty:
        return StrategyResult(name=name, returns=pd.Series(dtype=float), weights=weights, forecast=forecasts)

    aligned["strategy_return"] = aligned["weight"] * aligned["equity"] + (1 - aligned["weight"]) * aligned["bond"]
    return StrategyResult(
        name=name,
        returns=aligned["strategy_return"],
        weights=aligned["weight"],
        forecast=forecasts.reindex(aligned.index)
    )
