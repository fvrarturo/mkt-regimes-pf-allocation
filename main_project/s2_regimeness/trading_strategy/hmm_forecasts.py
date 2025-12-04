"""
HMM-based ERP forecasting using growth + inflation regimes (K=4).

This module implements strategies:
1. Forecast-based: Use forecasts at T to determine regime mix at T+1, apply to macro vars at T
2. Actual-based: Use actual values at T to determine regime mix, apply to macro vars at T
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import HMM model
import sys
HMM_REGIMES_DIR = Path(__file__).parent.parent / "regimes" / "HMM_regimes"
if str(HMM_REGIMES_DIR) not in sys.path:
    sys.path.insert(0, str(HMM_REGIMES_DIR))
from hmm_model import HMMRegimeModel


def get_variables_from_combination(combination: str) -> List[str]:
    """
    Extract variable names from combination name.
    
    Parameters:
    -----------
    combination : str
        Combination name like 'all_4vars' or '2vars_growth_inflation'
    
    Returns:
    --------
    List[str]
        List of variable factor names
    """
    ALL_VARIABLES = [
        'growth_factor',
        'inflation_factor',
        'monetary_policy_factor',
        'market_volatility_factor'
    ]
    
    if combination == 'all_4vars':
        return ALL_VARIABLES
    
    # Extract from 2vars_* format
    if combination.startswith('2vars_'):
        var_names_str = combination.replace('2vars_', '')
        
        # Handle multi-word variable names by checking for known patterns
        # Known variable name patterns (in order of specificity)
        known_patterns = [
            ('market_volatility', 'market_volatility_factor'),
            ('monetary_policy', 'monetary_policy_factor'),
            ('growth', 'growth_factor'),
            ('inflation', 'inflation_factor'),
        ]
        
        variables = []
        remaining = var_names_str
        
        # Try to match patterns from most specific to least specific
        for pattern, factor_name in known_patterns:
            if pattern in remaining:
                variables.append(factor_name)
                # Remove matched pattern (with surrounding underscores if present)
                remaining = remaining.replace(pattern, '').strip('_')
        
        # If we didn't match both variables, try splitting by underscore
        if len(variables) < 2:
            parts = var_names_str.split('_')
            # Try to reconstruct variable names
            var_map = {
                'growth': 'growth_factor',
                'inflation': 'inflation_factor',
                'monetary': 'monetary_policy_factor',
                'policy': 'monetary_policy_factor',
                'market': 'market_volatility_factor',
                'volatility': 'market_volatility_factor'
            }
            
            # Check for two-word combinations first
            if 'monetary' in parts and 'policy' in parts:
                variables.append('monetary_policy_factor')
                parts = [p for p in parts if p not in ['monetary', 'policy']]
            if 'market' in parts and 'volatility' in parts:
                variables.append('market_volatility_factor')
                parts = [p for p in parts if p not in ['market', 'volatility']]
            
            # Handle remaining single-word parts
            for part in parts:
                if part in var_map:
                    var_name = var_map[part]
                    if var_name not in variables:
                        variables.append(var_name)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for v in variables:
            if v not in seen:
                seen.add(v)
                result.append(v)
        
        if len(result) != 2:
            raise ValueError(f"Could not extract exactly 2 variables from {combination}. Got: {result}")
        
        return result
    
    raise ValueError(f"Unknown combination format: {combination}")


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
    
    # If base_dir is s2_regimeness, go up to main_project
    if base_dir.name == "s2_regimeness":
        main_project_dir = base_dir.parent
    else:
        main_project_dir = base_dir
    
    # Load macro data (base factors for HMM fitting)
    macro_path = main_project_dir / "data" / "macro_final" / "final_macro.csv"
    macro_df = pd.read_csv(macro_path, parse_dates=["date"]).set_index("date").sort_index()
    
    # Load regression coefficients
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
    
    # Extract variables from combination name
    variables = get_variables_from_combination(combination)
    
    # Load and fit HMM model
    # We need to fit it on the full historical data
    feature_data = macro_df[variables].dropna()
    
    if len(feature_data) == 0:
        raise ValueError(f"No data available for variables: {variables}")
    
    scaler = StandardScaler()
    features = scaler.fit_transform(feature_data.values)
    
    hmm_model = HMMRegimeModel(
        n_regimes=k,
        variables=variables,
        random_state=42,
        covar_reg=0.5,  # Strong regularization to encourage separation (shrink large covariances by 50%)
        min_covar=0.1   # Minimum covariance to prevent regimes from being too narrow
    )
    hmm_model.scaler = scaler
    hmm_model.fit(features, n_init=5)
    
    return hmm_model, coefficients, macro_df


def get_regime_probabilities(
    hmm_model: HMMRegimeModel,
    macro_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Get regime probabilities from HMM model given macro variables.
    
    Parameters:
    -----------
    hmm_model : HMMRegimeModel
        Fitted HMM model
    macro_df : pd.DataFrame
        DataFrame with macro variables (must include all variables used by the model)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns prob_R0, prob_R1, ..., prob_R{k-1}, indexed by date
    """
    # Extract variables used by the model
    variables = hmm_model.variables
    
    # Check if all variables are present
    missing_vars = [v for v in variables if v not in macro_df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables in macro_df: {missing_vars}")
    
    # Extract and align data
    aligned = macro_df[variables].dropna()
    
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
    
    # Get variables used by the model
    variables = hmm_model.variables
    
    # Create a temporary macro_df with forecasted values for regime variables
    # For variables that are forecasted (growth, inflation), use forecasts
    # For other variables, use actual values
    temp_macro_df = macro_df.copy()
    
    if "growth_factor" in variables and "growth_prediction" in forecast_df.columns:
        temp_macro_df["growth_factor"] = forecast_df["growth_prediction"].reindex(temp_macro_df.index, method='ffill')
    if "inflation_factor" in variables and "inflation_prediction" in forecast_df.columns:
        temp_macro_df["inflation_factor"] = forecast_df["inflation_prediction"].reindex(temp_macro_df.index, method='ffill')
    
    # Get regime probabilities from forecasts
    regime_probs = get_regime_probabilities(hmm_model, temp_macro_df)
    
    # Compute ERP forecast using macro vars at T (actual values, not forecasts)
    exclude_vars = variables  # Exclude all variables used for regime detection
    erp_forecast = compute_weighted_erp_forecast(
        regime_probs,
        coefficients,
        macro_df,
        exclude_vars=exclude_vars
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
    # Get regime probabilities from actuals
    regime_probs = get_regime_probabilities(hmm_model, macro_df)
    
    # Compute ERP forecast using macro vars at T
    # Exclude variables used for regime detection
    exclude_vars = hmm_model.variables
    erp_forecast = compute_weighted_erp_forecast(
        regime_probs,
        coefficients,
        macro_df,
        exclude_vars=exclude_vars
    )
    
    return erp_forecast


def generate_all_hmm_strategies(
    macro_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    base_dir: Optional[Path] = None,
    k: int = 4
) -> Dict[str, pd.Series]:
    """
    Generate all HMM-based strategies for all combinations with K=4.
    
    Parameters:
    -----------
    macro_df : pd.DataFrame
        Full macro dataset with all variables (from load_all_macro_variables)
    forecast_df : pd.DataFrame
        Forecast dataframe with columns: date, inflation_prediction, growth_prediction
    base_dir : Optional[Path]
        Base directory for loading data
    k : int
        Number of regimes (default: 4)
    
    Returns:
    --------
    Dict[str, pd.Series]
        Dictionary mapping strategy name -> ERP forecast series
    """
    # All combinations to test
    combinations = [
        'all_4vars',
        '2vars_growth_inflation',
        '2vars_growth_monetary_policy',
        '2vars_growth_market_volatility',
        '2vars_inflation_monetary_policy',
        '2vars_inflation_market_volatility',
        '2vars_market_volatility_monetary_policy'
    ]
    
    strategies = {}
    
    for combination in combinations:
    try:
            # Load HMM model and coefficients for this combination
            hmm_model, coefficients, _ = load_hmm_model_and_coefficients(
                base_dir=base_dir,
                combination=combination,
                k=k
            )
    
            # Generate actual-based strategy
            strategy_name = f"hmm_{combination}_k{k}_actual_based"
            strategies[strategy_name] = strategy_actual_based(
            hmm_model, coefficients, macro_df
        )
            print(f"✓ Generated {strategy_name}")
            
    except Exception as e:
            print(f"✗ Error with {combination}, K={k}: {e}")
        import traceback
        traceback.print_exc()
            strategy_name = f"hmm_{combination}_k{k}_actual_based"
            strategies[strategy_name] = pd.Series(dtype=float)
    
    return strategies

