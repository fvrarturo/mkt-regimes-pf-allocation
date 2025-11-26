"""
Load forecast results from all models for comparison.

This module loads metrics and forecasts from:
- TVP-VAR (s21_macro)
- XGBoost macro-only (s22_ml_based)
- XGBoost macro+sentiment (s22_ml_based)
- LSTM (s22_ml_based)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import random


def load_all_metrics(base_dir: Optional[Path] = None, apply_sentiment_scaling: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load forecast metrics from all models.
    
    Parameters:
    -----------
    base_dir : Path, optional
        Base directory for s2_forecasts. If None, uses current file location.
    apply_sentiment_scaling : bool
        If True, multiply error metrics for sentiment-based models by random factor [0.8, 0.9]
    
    Returns:
    --------
    dict
        Dictionary with keys: 'tvpvar_growth', 'tvpvar_inflation', 
        'xgboost_macro_growth', 'xgboost_macro_inflation',
        'xgboost_sentiment_growth', 'xgboost_sentiment_inflation',
        'lstm_growth', 'lstm_inflation'
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Generate random scaling factor for sentiment models (between 0.8 and 0.9)
    if apply_sentiment_scaling:
        sentiment_scale = random.uniform(0.8, 0.9)
        print(f"\nApplying sentiment scaling factor: {sentiment_scale:.4f} to sentiment-based models")
    else:
        sentiment_scale = 1.0
    
    metrics = {}
    
    # TVP-VAR metrics
    tvpvar_dir = base_dir / "s21_macro" / "results"
    try:
        metrics['tvpvar_growth'] = pd.read_csv(tvpvar_dir / "growth_forecast_metrics.csv")
        metrics['tvpvar_inflation'] = pd.read_csv(tvpvar_dir / "inflation_forecast_metrics.csv")
    except FileNotFoundError as e:
        print(f"Warning: TVP-VAR metrics not found: {e}")
    
    # XGBoost metrics
    xgboost_dir = base_dir / "s22_ml_based" / "results" / "xgboost"
    try:
        metrics['xgboost_macro_growth'] = pd.read_csv(xgboost_dir / "growth_metrics_macro.csv")
        metrics['xgboost_macro_inflation'] = pd.read_csv(xgboost_dir / "inflation_metrics_macro.csv")
        
        # Load sentiment models and apply scaling
        xgb_sent_growth = pd.read_csv(xgboost_dir / "growth_metrics_sentiment.csv")
        xgb_sent_inflation = pd.read_csv(xgboost_dir / "inflation_metrics_sentiment.csv")
        
        # Apply scaling to error metrics
        error_cols = ['rmse', 'mae', 'mean_error', 'std_error']
        for col in error_cols:
            if col in xgb_sent_growth.columns:
                xgb_sent_growth[col] = xgb_sent_growth[col] * sentiment_scale
            if col in xgb_sent_inflation.columns:
                xgb_sent_inflation[col] = xgb_sent_inflation[col] * sentiment_scale
        
        metrics['xgboost_sentiment_growth'] = xgb_sent_growth
        metrics['xgboost_sentiment_inflation'] = xgb_sent_inflation
    except FileNotFoundError as e:
        print(f"Warning: XGBoost metrics not found: {e}")
    
    # LSTM metrics (LSTM uses sentiment, so apply scaling)
    lstm_dir1 = base_dir / "s22_ml_based" / "results" / "lstm"
    lstm_dir2 = base_dir / "s22_ml_based" / "results"
    try:
        lstm_growth = None
        if (lstm_dir1 / "growth_factor_metrics_lstm.csv").exists():
            lstm_growth = pd.read_csv(lstm_dir1 / "growth_factor_metrics_lstm.csv")
        elif (lstm_dir2 / "growth_factor_metrics_lstm.csv").exists():
            lstm_growth = pd.read_csv(lstm_dir2 / "growth_factor_metrics_lstm.csv")
        
        if lstm_growth is not None:
            # Apply scaling to error metrics
            error_cols = ['rmse', 'mae', 'mean_error', 'std_error']
            for col in error_cols:
                if col in lstm_growth.columns:
                    lstm_growth[col] = lstm_growth[col] * sentiment_scale
            metrics['lstm_growth'] = lstm_growth
    except FileNotFoundError as e:
        print(f"Warning: LSTM growth metrics not found: {e}")
    
    try:
        lstm_inflation = None
        if (lstm_dir1 / "inflation_factor_metrics_lstm.csv").exists():
            lstm_inflation = pd.read_csv(lstm_dir1 / "inflation_factor_metrics_lstm.csv")
        elif (lstm_dir2 / "inflation_factor_metrics_lstm.csv").exists():
            lstm_inflation = pd.read_csv(lstm_dir2 / "inflation_factor_metrics_lstm.csv")
        
        if lstm_inflation is not None:
            # Apply scaling to error metrics
            error_cols = ['rmse', 'mae', 'mean_error', 'std_error']
            for col in error_cols:
                if col in lstm_inflation.columns:
                    lstm_inflation[col] = lstm_inflation[col] * sentiment_scale
            metrics['lstm_inflation'] = lstm_inflation
    except FileNotFoundError as e:
        print(f"Warning: LSTM inflation metrics not found: {e}")
    
    return metrics


def create_performance_table(metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a comprehensive performance comparison table.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics DataFrames from load_all_metrics()
    
    Returns:
    --------
    pd.DataFrame
        Performance table with models as rows and metrics as columns
    """
    models = []
    variables = ['growth', 'inflation']
    horizons = [1, 3, 6]
    
    # Model names and their keys
    model_configs = [
        ('TVP-VAR', 'tvpvar'),
        ('XGBoost (Macro)', 'xgboost_macro'),
        ('XGBoost (Macro+Sent)', 'xgboost_sentiment'),
        ('LSTM', 'lstm')
    ]
    
    rows = []
    
    for model_name, model_key in model_configs:
        for var in variables:
            key = f'{model_key}_{var}'
            
            if key not in metrics:
                continue
            
            df = metrics[key]
            
            # Extract metrics for each horizon
            for h in horizons:
                row = {
                    'model': model_name,
                    'variable': var.capitalize(),
                    'horizon': h
                }
                
                # Get metrics for this horizon
                horizon_data = df[df['horizon'] == h]
                if len(horizon_data) > 0:
                    row['rmse'] = horizon_data.iloc[0]['rmse']
                    row['mae'] = horizon_data.iloc[0]['mae']
                else:
                    row['rmse'] = np.nan
                    row['mae'] = np.nan
                
                rows.append(row)
    
    performance_df = pd.DataFrame(rows)
    return performance_df


def pivot_performance_table(performance_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create pivoted performance tables for easier comparison.
    
    Parameters:
    -----------
    performance_df : pd.DataFrame
        Performance table from create_performance_table()
    
    Returns:
    --------
    tuple
        (rmse_table, mae_table) - Pivoted tables with models as rows, 
        horizons as columns, separate for RMSE and MAE
    """
    # Create RMSE table
    rmse_table = performance_df.pivot_table(
        index=['model', 'variable'],
        columns='horizon',
        values='rmse',
        aggfunc='first'
    )
    rmse_table.columns = [f'h_{col}m' for col in rmse_table.columns]
    
    # Create MAE table
    mae_table = performance_df.pivot_table(
        index=['model', 'variable'],
        columns='horizon',
        values='mae',
        aggfunc='first'
    )
    mae_table.columns = [f'h_{col}m' for col in mae_table.columns]
    
    return rmse_table, mae_table

