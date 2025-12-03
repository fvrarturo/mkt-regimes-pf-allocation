"""
HMM-based ERP forecasting using growth + inflation regimes (K=4).

This module implements 4 strategies:
1. Forecast-based: Use forecasts at T to determine regime mix at T+1, apply to macro vars at T
2. Actual-based: Use actual values at T to determine regime mix, apply to macro vars at T
3. Fixed benchmark: 50/50 allocation
4. Pure prediction: Use actual T+1 values for regimes (lookahead), apply to macro vars at T
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import HMM model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "s1_macro_vars" / "s12_regimeness" / "regimes" / "HMM_regimes"))
from hmm_model import HMMRegimeModel


def load_hmm_model_and_coefficients(
    base_dir: Optional[Path] = None,
    combination: str = "2vars_growth_inflation",
    k: int = 4
) -> Tuple[HMMRegimeModel, Dict[int, Dict[str, float]], pd.DataFrame]:
    """
    Load the fitted HMM model and regression coefficients.
    
    Returns:
    --------
    hmm_model : HMMRegimeModel
        Fitted HMM model for growth + inflation, K=4
    coefficients : Dict[int, Dict[str, float]]
        Dictionary mapping regime -> variable -> coefficient
    macro_data : pd.DataFrame
        Full macro dataset used for fitting
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Load macro data (base factors for HMM fitting)
    macro_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    macro_df = pd.read_csv(macro_path, parse_dates=["date"]).set_index("date").sort_index()
    
    # Load regression coefficients
    reg_path = (
        base_dir / "s1_macro_vars" / "s12_regimeness" / "regressions" / "results" /
        "conditional_regression_results_all.csv"
    )
    reg_df = pd.read_csv(reg_path)
    
    # Filter for the specific combination and K
    reg_subset = reg_df[
        (reg_df["combination"] == combination) & (reg_df["K"] == k)
    ].copy()
    
    if reg_subset.empty:
        raise ValueError(f"No regression results found for {combination}, K={k}")
    
    # Build coefficients dictionary: regime -> variable -> coefficient
    coefficients: Dict[int, Dict[str, float]] = {}
    for regime in sorted(reg_subset["regime"].unique()):
        regime_data = reg_subset[reg_subset["regime"] == regime]
        coefficients[int(regime)] = dict(zip(regime_data["variable"], regime_data["coefficient"]))
    
    # Load and fit HMM model
    # We need to fit it on the full historical data
    variables = ["growth_factor", "inflation_factor"]
    feature_data = macro_df[variables].dropna()
    
    scaler = StandardScaler()
    features = scaler.fit_transform(feature_data.values)
    
    hmm_model = HMMRegimeModel(
        n_regimes=k,
        variables=variables,
        random_state=42
    )
    hmm_model.scaler = scaler
    hmm_model.fit(features, n_init=5)
    
    return hmm_model, coefficients, macro_df


def get_regime_probabilities(
    hmm_model: HMMRegimeModel,
    growth_values: pd.Series,
    inflation_values: pd.Series
) -> pd.DataFrame:
    """
    Get regime probabilities from HMM model given growth and inflation values.
    
    Parameters:
    -----------
    hmm_model : HMMRegimeModel
        Fitted HMM model
    growth_values : pd.Series
        Growth factor values (indexed by date)
    inflation_values : pd.Series
        Inflation factor values (indexed by date)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns prob_R0, prob_R1, ..., prob_R{k-1}, indexed by date
    """
    # Align series
    aligned = pd.DataFrame({
        "growth_factor": growth_values,
        "inflation_factor": inflation_values
    }).dropna()
    
    if aligned.empty:
        return pd.DataFrame()
    
    # Prepare features using the model's scaler
    features = hmm_model.prepare_features(aligned, fit_scaler=False)
    
    # Get probabilities
    probs = hmm_model.predict_proba(features)
    
    # Create DataFrame
    prob_df = pd.DataFrame(
        probs,
        index=aligned.index,
        columns=[f"prob_R{i}" for i in range(hmm_model.n_regimes)]
    )
    
    return prob_df


def compute_weighted_erp_forecast(
    regime_probs: pd.DataFrame,
    coefficients: Dict[int, Dict[str, float]],
    macro_vars: pd.DataFrame,
    exclude_vars: Optional[list] = None
) -> pd.Series:
    """
    Compute ERP forecast using weighted regime coefficients.
    
    Note: Coefficients are for standardized variables, so we standardize macro_vars first.
    
    Parameters:
    -----------
    regime_probs : pd.DataFrame
        Regime probabilities with columns prob_R0, prob_R1, etc.
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
    common_index = regime_probs.index.intersection(macro_vars.index)
    if len(common_index) == 0:
        return pd.Series(dtype=float, index=common_index)
    
    regime_probs_aligned = regime_probs.reindex(common_index)
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
    # Use the full available data to compute mean/std
    macro_subset = macro_vars_aligned[forecast_vars].dropna()
    if len(macro_subset) == 0:
        return pd.Series(dtype=float, index=common_index)
    
    means = macro_subset.mean()
    stds = macro_subset.std().replace(0, 1.0)  # Avoid division by zero
    
    # Standardize the aligned macro variables
    macro_standardized = (macro_vars_aligned[forecast_vars] - means) / stds
    
    forecasts = []
    
    for date in common_index:
        probs_row = regime_probs_aligned.loc[date]
        macro_row_std = macro_standardized.loc[date]
        
        # Compute weighted coefficients
        weighted_coefs = {}
        for var in forecast_vars:
            weighted_coef = 0.0
            for regime_id in range(len(probs_row)):
                prob = probs_row[f"prob_R{regime_id}"]
                coef = coefficients.get(regime_id, {}).get(var, 0.0)
                weighted_coef += prob * coef
            weighted_coefs[var] = weighted_coef
        
        # Compute forecast: sum(weighted_coef * standardized_macro_value)
        forecast = 0.0
        for var, coef in weighted_coefs.items():
            if var in macro_row_std.index:
                try:
                    value = macro_row_std.loc[var]
                    if pd.notna(value) and np.isfinite(value):
                        forecast += coef * float(value)
                except (KeyError, ValueError, TypeError):
                    continue
        
        forecasts.append(forecast)
    
    return pd.Series(forecasts, index=common_index, name="erp_forecast")


def strategy_forecast_based(
    hmm_model: HMMRegimeModel,
    coefficients: Dict[int, Dict[str, float]],
    macro_df: pd.DataFrame,
    forecast_df: pd.DataFrame
) -> pd.Series:
    """
    Strategy 1: Use forecasts at T to determine regime mix at T+1, apply to macro vars at T.
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
    
    # Get regime probabilities from forecasts
    regime_probs = get_regime_probabilities(hmm_model, growth_forecast, inflation_forecast)
    
    # Compute ERP forecast using macro vars at T
    erp_forecast = compute_weighted_erp_forecast(
        regime_probs,
        coefficients,
        macro_df,
        exclude_vars=["growth_factor", "inflation_factor"]
    )
    
    return erp_forecast


def strategy_actual_based(
    hmm_model: HMMRegimeModel,
    coefficients: Dict[int, Dict[str, float]],
    macro_df: pd.DataFrame
) -> pd.Series:
    """
    Strategy 2: Use actual values at T to determine regime mix, apply to macro vars at T.
    """
    # Get actual growth and inflation
    growth_actual = macro_df["growth_factor"]
    inflation_actual = macro_df["inflation_factor"]
    
    # Get regime probabilities from actuals
    regime_probs = get_regime_probabilities(hmm_model, growth_actual, inflation_actual)
    
    # Compute ERP forecast using macro vars at T
    erp_forecast = compute_weighted_erp_forecast(
        regime_probs,
        coefficients,
        macro_df,
        exclude_vars=["growth_factor", "inflation_factor"]
    )
    
    return erp_forecast


def strategy_pure_prediction(
    hmm_model: HMMRegimeModel,
    coefficients: Dict[int, Dict[str, float]],
    macro_df: pd.DataFrame
) -> pd.Series:
    """
    Strategy 4: Use actual T+1 values for regimes (lookahead), apply to macro vars at T.
    
    Note: This is a "pure prediction" strategy that uses future information
    to determine regimes, but still uses macro vars at T for the forecast.
    """
    # Shift growth and inflation forward by 1 period to get T+1 values
    growth_t1 = macro_df["growth_factor"].shift(-1)
    inflation_t1 = macro_df["inflation_factor"].shift(-1)
    
    # Get regime probabilities from T+1 actuals
    regime_probs = get_regime_probabilities(hmm_model, growth_t1, inflation_t1)
    
    # Align regime_probs index back to T (remove the last row which has no T+1 data)
    regime_probs = regime_probs.iloc[:-1] if len(regime_probs) > 0 else regime_probs
    
    # Compute ERP forecast using macro vars at T
    erp_forecast = compute_weighted_erp_forecast(
        regime_probs,
        coefficients,
        macro_df,
        exclude_vars=["growth_factor", "inflation_factor"]
    )
    
    return erp_forecast


def generate_all_hmm_strategies(
    macro_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    base_dir: Optional[Path] = None
) -> Dict[str, pd.Series]:
    """
    Generate all 4 HMM-based strategies.
    
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
    # Load HMM model and coefficients
    hmm_model, coefficients, _ = load_hmm_model_and_coefficients(base_dir=base_dir)
    
    strategies = {}
    
    # Strategy 1: Forecast-based
    try:
        strategies["hmm_forecast_based"] = strategy_forecast_based(
            hmm_model, coefficients, macro_df, forecast_df
        )
    except Exception as e:
        print(f"Error in forecast-based strategy: {e}")
        import traceback
        traceback.print_exc()
        strategies["hmm_forecast_based"] = pd.Series(dtype=float)
    
    # Strategy 2: Actual-based
    try:
        strategies["hmm_actual_based"] = strategy_actual_based(
            hmm_model, coefficients, macro_df
        )
    except Exception as e:
        print(f"Error in actual-based strategy: {e}")
        import traceback
        traceback.print_exc()
        strategies["hmm_actual_based"] = pd.Series(dtype=float)
    
    # Strategy 3: Fixed 50/50 benchmark (no forecast needed, handled in main.py)
    # We'll return None for this and handle it separately
    
    # Strategy 4: Pure prediction
    try:
        strategies["hmm_pure_prediction"] = strategy_pure_prediction(
            hmm_model, coefficients, macro_df
        )
    except Exception as e:
        print(f"Error in pure prediction strategy: {e}")
        import traceback
        traceback.print_exc()
        strategies["hmm_pure_prediction"] = pd.Series(dtype=float)
    
    return strategies

