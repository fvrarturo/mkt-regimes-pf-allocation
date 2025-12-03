"""
Data loading utilities for Step 3 trading evaluation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _get_base_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return Path(__file__).parent.parent


def load_market_data(base_dir: Optional[Path] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load monthly equity returns, bond returns, and ERP.
    """
    base_dir = _get_base_dir(base_dir)
    data_dir = base_dir / "data" / "macro_processed"

    sp500 = pd.read_csv(data_dir / "sp500_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    equity_returns = (sp500["pct_change_mom"] / 100.0).resample("M").last()

    tbill = pd.read_csv(data_dir / "3m_yield_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    bond_returns = (tbill["value"] / 100.0 / 12.0).resample("M").last()

    aligned = pd.DataFrame({"equity_return": equity_returns, "bond_return": bond_returns}).dropna()
    erp = aligned["equity_return"] - aligned["bond_return"]

    equity_returns = equity_returns.reindex(aligned.index)
    bond_returns = bond_returns.reindex(aligned.index)

    return equity_returns, bond_returns, erp


def load_macro_features(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load macro factor panel (base factors + selected processed series).
    """
    base_dir = _get_base_dir(base_dir)
    macro_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    macro_df = pd.read_csv(macro_path, parse_dates=["date"]).set_index("date").sort_index()

    cols = ["growth_factor", "inflation_factor", "monetary_policy_factor", "market_volatility_factor"]
    missing = [c for c in cols if c not in macro_df.columns]
    if missing:
        raise ValueError(f"Missing macro columns in final_macro.csv: {missing}")

    macro_df = macro_df[cols].dropna()
    macro_df.index = macro_df.index.to_period("M").to_timestamp("M")

    selection_dir = base_dir / "data" / "macro_processed_full" / "selection"
    if selection_dir.exists():
        selection_features: Dict[str, pd.Series] = {}
        for csv_path in selection_dir.glob("*.csv"):
            df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
            if "value" not in df.columns:
                continue
            series = df["value"].resample("M").last().ffill()
            name = csv_path.stem.split("_processed")[0].replace(" ", "").lower()
            selection_features[name] = series

        if selection_features:
            selection_df = pd.DataFrame(selection_features).dropna(how="all")
            selection_df.index = selection_df.index.to_period("M").to_timestamp("M")
            macro_df = macro_df.join(selection_df, how="inner")

    return macro_df.dropna()


def load_regime_probabilities(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load regime probabilities from the Growth+Policy HMM run.
    """
    base_dir = _get_base_dir(base_dir)
    regime_path = (
        base_dir
        / "s1_macro_vars"
        / "s12_regimeness"
        / "regimes"
        / "HMM_regimes"
        / "results_2vars_optimal"
        / "regime_assignments.csv"
    )
    if not regime_path.exists():
        raise FileNotFoundError(f"Regime file not found: {regime_path}")

    regime_df = pd.read_csv(regime_path, parse_dates=["date"]).set_index("date").sort_index()
    regime_df.index = regime_df.index.to_period("M").to_timestamp("M")
    prob_cols = [c for c in regime_df.columns if c.startswith("prob_R")]
    if not prob_cols:
        raise ValueError("No probability columns found in regime assignments.")

    return regime_df[prob_cols + ["regime"]]


def load_2x2_regimes(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load 2×2 Growth × Inflation regime assignments.
    """
    base_dir = _get_base_dir(base_dir)
    regime_path = (
        base_dir
        / "s1_macro_vars"
        / "s12_regimeness"
        / "regimes"
        / "2x2_regimes"
        / "results"
        / "regime_assignments.csv"
    )
    if not regime_path.exists():
        raise FileNotFoundError(f"2x2 regime file not found: {regime_path}")

    regime_df = pd.read_csv(regime_path, parse_dates=["date"]).set_index("date").sort_index()
    regime_df.index = regime_df.index.to_period("M").to_timestamp("M")

    if "regime" not in regime_df.columns:
        raise ValueError("2x2 regime assignments must contain a 'regime' column.")

    return regime_df[["regime"]]
