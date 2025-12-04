"""
2x2 Regime-based ERP forecasting using hard thresholds.

This module implements strategies:
1. Forecast-based: Use forecasts at T to determine hard regime at T+1, apply to macro vars at T
2. Actual-based: Use actual values at T to determine hard regime, apply to macro vars at T
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import 2x2 regime definitions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "regimes" / "2x2_regimes"))
from regime_definitions import RegimeDefinitions


def load_2x2_regime_definitions_and_coefficients(
    base_dir: Optional[Path] = None,
    combination: str = "2vars_growth_inflation",
    k: int = 4
) -> Tuple[RegimeDefinitions, Dict[int, Dict[str, float]], pd.DataFrame]:
    """
    Load the 2x2 regime definitions and regression coefficients.
    
    Note: For now, we use HMM K=4 coefficients mapped to 2x2 regimes.
    In the future, we should run separate regressions for 2x2 regimes.
    
    Returns:
    --------
    regime_def : RegimeDefinitions
        2x2 regime classifier
    coefficients : Dict[int, Dict[str, float]]
        Dictionary mapping regime (0-3) -> variable -> coefficient
    macro_data : pd.DataFrame
        Full macro dataset used for fitting
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # If base_dir is s2_regimeness, go up to main_project
    if base_dir.name == "s2_regimeness":
        main_project_dir = base_dir.parent
    else:
        main_project_dir = base_dir
    
    # Load macro data
    macro_path = main_project_dir / "data" / "macro_final" / "final_macro.csv"
    macro_df = pd.read_csv(macro_path, parse_dates=["date"]).set_index("date").sort_index()
    
    # Initialize 2x2 regime definitions
    regime_def = RegimeDefinitions(threshold_method='median')
    
    # Determine thresholds from historical data
    growth_data = macro_df["growth_factor"].dropna()
    inflation_data = macro_df["inflation_factor"].dropna()
    regime_def.determine_thresholds(growth_data, inflation_data)
    
    # Load regression coefficients for 2x2 regimes
    reg_path = (
        base_dir / "regressions" / "results" /
        "conditional_regression_results_all.csv"
    )
    
    if not reg_path.exists():
        # Try alternative path
        reg_path = (
            main_project_dir / "s1_macro_vars" / "s12_regimeness" / "regressions" / "results" /
            "conditional_regression_results_all.csv"
        )
    
    if not reg_path.exists():
        raise FileNotFoundError(f"Regression results not found. Tried: {reg_path}")
    
    reg_df = pd.read_csv(reg_path)
    
    # Filter for 2x2_growth_inflation combination, K=4
    reg_subset = reg_df[
        (reg_df["combination"] == "2x2_growth_inflation") & (reg_df["K"] == 4)
    ].copy()
    
    if reg_subset.empty:
        raise ValueError(f"No regression results found for 2x2_growth_inflation, K=4")
    
    # Build coefficients dictionary: regime -> variable -> coefficient
    coefficients: Dict[int, Dict[str, float]] = {}
    for regime in sorted(reg_subset["regime"].unique()):
        regime_data = reg_subset[reg_subset["regime"] == regime]
        coefficients[int(regime)] = dict(zip(regime_data["variable"], regime_data["coefficient"]))
    
    return regime_def, coefficients, macro_df


def get_hard_regime_assignment(
    regime_def: RegimeDefinitions,
    growth_values: pd.Series,
    inflation_values: pd.Series
) -> pd.Series:
    """
    Get hard regime assignments from 2x2 classifier given growth and inflation values.
    
    Parameters:
    -----------
    regime_def : RegimeDefinitions
        2x2 regime classifier
    growth_values : pd.Series
        Growth factor values (indexed by date)
    inflation_values : pd.Series
        Inflation factor values (indexed by date)
    
    Returns:
    --------
    pd.Series
        Regime assignments (0-3) indexed by date
    """
    # Align series
    aligned = pd.DataFrame({
        "growth_factor": growth_values,
        "inflation_factor": inflation_values
    }).dropna()
    
    if aligned.empty:
        return pd.Series(dtype=int)
    
    # Classify each row into a hard regime
    regimes = regime_def.classify_dataframe(
        aligned,
        growth_col="growth_factor",
        inflation_col="inflation_factor"
    )
    
    return regimes


def compute_erp_forecast_hard_regime(
    regime_assignments: pd.Series,
    coefficients: Dict[int, Dict[str, float]],
    macro_vars: pd.DataFrame,
    exclude_vars: Optional[list] = None
) -> pd.Series:
    """
    Compute ERP forecast using hard regime coefficients.
    
    Note: Coefficients are for standardized variables, so we standardize macro_vars first.
    
    Parameters:
    -----------
    regime_assignments : pd.Series
        Hard regime assignments (0-3) indexed by date
    coefficients : Dict[int, Dict[str, float]]
        Dictionary mapping regime -> variable -> coefficient (for standardized variables)
    macro_vars : pd.DataFrame
        Macro variables at time T (excluding growth and inflation)
    exclude_vars : list, optional
        Variables to exclude from the forecast (default: ['growth_factor', 'inflation_factor'])
    
    Returns:
    --------
    pd.Series
        ERP forecasts indexed by date
    """
    if exclude_vars is None:
        exclude_vars = ["growth_factor", "inflation_factor"]
    
    # Align data
    common_index = regime_assignments.index.intersection(macro_vars.index)
    if len(common_index) == 0:
        return pd.Series(dtype=float, index=common_index)
    
    regime_assignments_aligned = regime_assignments.reindex(common_index)
    macro_vars_aligned = macro_vars.reindex(common_index)
    
    # Get all variables that appear in coefficients
    all_vars = set()
    for regime_coefs in coefficients.values():
        all_vars.update(regime_coefs.keys())
    
    # Remove excluded variables
    forecast_vars = [v for v in all_vars if v not in exclude_vars]
    
    # Ensure all forecast variables exist in macro_vars
    missing_vars = [v for v in forecast_vars if v not in macro_vars_aligned.columns]
    if missing_vars:
        print(f"Warning: Missing variables in macro data: {missing_vars}")
        forecast_vars = [v for v in forecast_vars if v in macro_vars_aligned.columns]
    
    if len(forecast_vars) == 0:
        return pd.Series(dtype=float, index=common_index)
    
    # Standardize macro variables (coefficients are for standardized variables)
    macro_subset = macro_vars_aligned[forecast_vars].dropna()
    if len(macro_subset) == 0:
        return pd.Series(dtype=float, index=common_index)
    
    means = macro_subset.mean()
    stds = macro_subset.std().replace(0, 1.0)  # Avoid division by zero
    
    # Standardize the aligned macro variables
    macro_standardized = (macro_vars_aligned[forecast_vars] - means) / stds
    
    forecasts = []
    
    for date in common_index:
        regime_id = int(regime_assignments_aligned.loc[date])
        macro_row_std = macro_standardized.loc[date]
        
        # Get coefficients for this regime
        regime_coefs = coefficients.get(regime_id, {})
        
        # Compute forecast: sum(coef * standardized_macro_value)
        forecast = 0.0
        for var, coef in regime_coefs.items():
            if var in forecast_vars and var in macro_row_std.index:
                try:
                    value = macro_row_std.loc[var]
                    if pd.notna(value) and np.isfinite(value):
                        forecast += coef * float(value)
                except (KeyError, ValueError, TypeError):
                    continue
        
        forecasts.append(forecast)
    
    return pd.Series(forecasts, index=common_index, name="erp_forecast")


def strategy_forecast_based(
    regime_def: RegimeDefinitions,
    coefficients: Dict[int, Dict[str, float]],
    macro_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> pd.Series:
    """
    Strategy 1: Use forecasts at T to determine hard regime at T+1, apply to macro vars at T.
    """
    # Ensure forecast_df has date column
    if "date" not in forecast_df.columns:
        raise ValueError("forecast_df must have a 'date' column")
    
    # Convert date to datetime and normalize to month-end for alignment
    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    
    # Convert to month-end to match macro_df index
    forecast_df["date"] = forecast_df["date"].dt.to_period("M").dt.to_timestamp("M")
    forecast_df = forecast_df.set_index("date").sort_index()
    
    # Get forecasted growth and inflation
    growth_forecast = forecast_df["growth_prediction"]
    inflation_forecast = forecast_df["inflation_prediction"]
    
    # Get hard regime assignments from forecasts
    regime_assignments = get_hard_regime_assignment(
        regime_def, growth_forecast, inflation_forecast
    )
    
    # Compute ERP forecast using macro vars at T
    erp_forecast = compute_erp_forecast_hard_regime(
        regime_assignments,
        coefficients,
        macro_df,
        exclude_vars=["growth_factor", "inflation_factor"]
    )
    
    return erp_forecast


def strategy_actual_based(
    regime_def: RegimeDefinitions,
    coefficients: Dict[int, Dict[str, float]],
    macro_df: pd.DataFrame
) -> pd.Series:
    """
    Strategy 2: Use actual values at T to determine hard regime, apply to macro vars at T.
    """
    # Get actual growth and inflation
    growth_actual = macro_df["growth_factor"]
    inflation_actual = macro_df["inflation_factor"]
    
    # Get hard regime assignments from actuals
    regime_assignments = get_hard_regime_assignment(
        regime_def, growth_actual, inflation_actual
    )
    
    # Compute ERP forecast using macro vars at T
    erp_forecast = compute_erp_forecast_hard_regime(
        regime_assignments,
        coefficients,
        macro_df,
        exclude_vars=["growth_factor", "inflation_factor"]
    )
    
    return erp_forecast


def generate_all_2x2_strategies(
    macro_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    base_dir: Optional[Path] = None
) -> Dict[str, pd.Series]:
    """
    Generate all 2x2 regime-based strategies.
    
    Parameters:
    -----------
    macro_df : pd.DataFrame
        Full macro dataset with all variables (from load_all_macro_variables)
    forecast_df : pd.DataFrame
        Forecast dataframe with columns: date, inflation_prediction, growth_prediction
    base_dir : Optional[Path]
        Base directory for loading data
    
    Returns:
    --------
    Dict[str, pd.Series]
        Dictionary mapping strategy name -> ERP forecast series
    """
    # Load 2x2 regime definitions and coefficients
    regime_def, coefficients, _ = load_2x2_regime_definitions_and_coefficients(base_dir=base_dir)
    
    strategies = {}
    
    # Strategy 1: Forecast-based
    try:
        strategies["2x2_forecast_based"] = strategy_forecast_based(
            regime_def, coefficients, macro_df, forecast_df
        )
    except Exception as e:
        print(f"Error in 2x2 forecast-based strategy: {e}")
        import traceback
        traceback.print_exc()
        strategies["2x2_forecast_based"] = pd.Series(dtype=float)
    
    # Strategy 2: Actual-based
    try:
        strategies["2x2_actual_based"] = strategy_actual_based(
            regime_def, coefficients, macro_df
        )
    except Exception as e:
        print(f"Error in 2x2 actual-based strategy: {e}")
        import traceback
        traceback.print_exc()
        strategies["2x2_actual_based"] = pd.Series(dtype=float)
    
    return strategies

