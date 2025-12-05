"""
Data loading functions for VAR forecasting model.

This module handles loading and preprocessing macro forecasting data.
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats
from pathlib import Path


def load_data(final_macro_path='final_macro.csv', monthly_pred_path='monthly_pred.csv'):
    """
    Load and merge final macro and monthly prediction data.
    
    Parameters
    ----------
    final_macro_path : str
        Path to final_macro.csv
    monthly_pred_path : str
        Path to monthly_pred.csv
    
    Returns
    -------
    pd.DataFrame
        Merged dataset
    """
    final_macro = pd.read_csv(final_macro_path)
    final_macro['date'] = pd.to_datetime(final_macro['date'])
    final_macro = final_macro.sort_values('date').reset_index(drop=True)
    
    monthly_pred = pd.read_csv(monthly_pred_path)
    monthly_pred['date'] = pd.to_datetime(monthly_pred['date'])
    monthly_pred = monthly_pred.sort_values('date').reset_index(drop=True)
    
    # Merge datasets on 'date'
    data_merged = final_macro.merge(
        monthly_pred, on='date', how='inner', 
        suffixes=('_target', '_pred')
    )
    data_merged = data_merged.sort_values('date').reset_index(drop=True)
    
    return data_merged


def winsorize_growth_factor(data_merged, limits=[0.05, 0.05]):
    """
    Winsorize growth_factor_target to handle outliers.
    
    Parameters
    ----------
    data_merged : pd.DataFrame
        Input dataframe
    limits : list
        Winsorization limits [lower, upper]
    
    Returns
    -------
    pd.DataFrame
        Dataframe with winsorized growth_factor_target
    """
    data_merged = data_merged.copy()
    data_merged['growth_factor_target_winsorized'] = mstats.winsorize(
        data_merged['growth_factor_target'], 
        limits=limits
    )
    data_merged['growth_factor_target'] = data_merged['growth_factor_target_winsorized']
    data_merged = data_merged.drop(columns=['growth_factor_target_winsorized'])
    
    return data_merged


def prepare_var_data(data_merged):
    """
    Prepare data for VAR modeling.
    
    Parameters
    ----------
    data_merged : pd.DataFrame
        Merged dataset
    
    Returns
    -------
    pd.DataFrame
        Full VAR dataset with targets and predictors
    """
    available_predictors = [
        col for col in data_merged.columns 
        if col != 'date' and col not in [
            'inflation_factor_target', 'growth_factor_target',
            'inflation_factor', 'growth_factor'
        ]
    ]
    
    full_var_data = data_merged[
        ['inflation_factor_target', 'growth_factor_target'] + available_predictors
    ].copy()
    
    return full_var_data, available_predictors

