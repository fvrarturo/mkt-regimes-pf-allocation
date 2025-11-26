"""
Statistical comparison functions for cross-model evaluation.

Functions:
- run_dm_tests: Run Diebold-Mariano tests between model pairs
- compute_relative_improvement: Compute percentage improvement metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

# Import DM test from s22_ml_based
import sys
from pathlib import Path
base_path = Path(__file__).parent.parent / "s22_ml_based"
sys.path.insert(0, str(base_path))
import importlib.util
spec = importlib.util.spec_from_file_location("ml_stats", base_path / "stats.py")
ml_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_stats)
diebold_mariano_test = ml_stats.diebold_mariano_test


def run_dm_tests(
    forecasts_dict: Dict[str, pd.Series],
    actuals: pd.Series,
    horizons: list = [1, 3, 6],
    variable_name: str = "variable"
) -> pd.DataFrame:
    """
    Run Diebold-Mariano tests between model pairs.
    
    Parameters:
    -----------
    forecasts_dict : dict
        Dictionary mapping model names to forecast Series
    actuals : pd.Series
        Actual values
    horizons : list
        Forecast horizons
    variable_name : str
        Name of variable
    
    Returns:
    --------
    pd.DataFrame
        DM test results
    """
    results = []
    model_names = list(forecasts_dict.keys())
    
    # Define key comparisons
    comparisons = [
        ('XGBoost (Macro)', 'TVP-VAR'),
        ('XGBoost (Macro+Sent)', 'XGBoost (Macro)'),
        ('LSTM', 'XGBoost (Macro+Sent)'),
        ('LSTM', 'TVP-VAR'),
        ('XGBoost (Macro+Sent)', 'TVP-VAR')
    ]
    
    for model1_name, model2_name in comparisons:
        if model1_name not in forecasts_dict or model2_name not in forecasts_dict:
            continue
        
        forecast1 = forecasts_dict[model1_name]
        forecast2 = forecasts_dict[model2_name]
        
        for h in horizons:
            # Align forecasts with actuals
            errors1 = []
            errors2 = []
            
            for forecast_date in forecast1.index:
                target_date = forecast_date + pd.DateOffset(months=h)
                if target_date in actuals.index:
                    errors1.append(actuals.loc[target_date] - forecast1.loc[forecast_date])
            
            for forecast_date in forecast2.index:
                target_date = forecast_date + pd.DateOffset(months=h)
                if target_date in actuals.index:
                    errors2.append(actuals.loc[target_date] - forecast2.loc[forecast_date])
            
            # Align error arrays
            min_len = min(len(errors1), len(errors2))
            if min_len > 10:  # Need sufficient observations
                errors1 = np.array(errors1[:min_len])
                errors2 = np.array(errors2[:min_len])
                
                # Run DM test
                dm_result = diebold_mariano_test(errors1, errors2, h=h, power=2)
                
                results.append({
                    'model1': model1_name,
                    'model2': model2_name,
                    'variable': variable_name,
                    'horizon': h,
                    'dm_statistic': dm_result['dm_statistic'],
                    'p_value': dm_result['p_value'],
                    'mean_loss_diff': dm_result['mean_loss_diff'],
                    'n_obs': dm_result['n_obs']
                })
    
    return pd.DataFrame(results)


def compute_relative_improvement(
    performance_df: pd.DataFrame,
    baseline_model: str = 'TVP-VAR'
) -> pd.DataFrame:
    """
    Compute relative improvement over baseline model.
    
    Parameters:
    -----------
    performance_df : pd.DataFrame
        Performance table
    baseline_model : str
        Baseline model for comparison
    
    Returns:
    --------
    pd.DataFrame
        Relative improvement table
    """
    improvement_rows = []
    
    for variable in performance_df['variable'].unique():
        var_data = performance_df[performance_df['variable'] == variable]
        baseline_data = var_data[var_data['model'] == baseline_model]
        
        if len(baseline_data) == 0:
            continue
        
        for horizon in var_data['horizon'].unique():
            baseline_h = baseline_data[baseline_data['horizon'] == horizon]
            if len(baseline_h) == 0:
                continue
            
            baseline_rmse = baseline_h.iloc[0]['rmse']
            baseline_mae = baseline_h.iloc[0]['mae']
            
            for model in var_data['model'].unique():
                if model == baseline_model:
                    continue
                
                model_h = var_data[(var_data['model'] == model) & 
                                  (var_data['horizon'] == horizon)]
                if len(model_h) == 0:
                    continue
                
                model_rmse = model_h.iloc[0]['rmse']
                model_mae = model_h.iloc[0]['mae']
                
                improvement_rows.append({
                    'model': model,
                    'variable': variable,
                    'horizon': horizon,
                    'rmse_improvement_pct': ((baseline_rmse - model_rmse) / baseline_rmse) * 100,
                    'mae_improvement_pct': ((baseline_mae - model_mae) / baseline_mae) * 100
                })
    
    return pd.DataFrame(improvement_rows)

