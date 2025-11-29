"""
Main script for LSTM forecasting of GDP and inflation.

This script implements Section 5.3 from goals.md:
- LSTM sequence model for multivariate forecasting
- Joint prediction of GDP and inflation
- Comparison with TVP-VAR and XGBoost models

Outputs:
- Forecast performance tables (RMSE, MAE)
- Learning curves
- Forecast vs realized plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from preprocessing import load_data
from lstm_preprocessing import prepare_lstm_data
from lstm_model import LSTMForecaster
from stats import compute_forecast_metrics
from lstm_plotting import plot_learning_curves, plot_forecast_vs_realized_lstm


def create_validation_split(
    X_train: np.ndarray,
    y_train: dict,
    val_split: float = 0.2
) -> tuple:
    """
    Create validation split from training data.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training sequences
    y_train : dict
        Training targets by horizon
    val_split : float
        Fraction of training data for validation
    
    Returns:
    --------
    tuple
        (X_train_split, X_val, y_train_split, y_val)
    """
    n_train = len(X_train)
    n_val = int(n_train * val_split)
    
    X_train_split = X_train[:-n_val]
    X_val = X_train[-n_val:]
    
    y_train_split = {}
    y_val = {}
    
    for h, targets in y_train.items():
        y_train_split[h] = {}
        y_val[h] = {}
        for var_name, values in targets.items():
            y_train_split[h][var_name] = values[:-n_val]
            y_val[h][var_name] = values[-n_val:]
    
    return X_train_split, X_val, y_train_split, y_val


def main():
    """Main execution function."""
    print("="*80)
    print("LSTM Forecasting: GDP and Inflation")
    print("="*80)
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("ERROR: TensorFlow is required for LSTM models.")
        print("Install with: pip install tensorflow")
        return
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    horizons = [1, 3, 6]  # months
    train_split = 0.65  # 65% for training
    sequence_length = 12  # 12 months of history
    lstm_units = 64  # More capacity to capture dynamics
    dropout_rate = 0.2  # Moderate dropout
    recurrent_dropout = 0.1  # Light recurrent dropout
    l2_reg = 0.001  # Light L2 regularization
    learning_rate = 0.001  # Standard learning rate
    epochs = 100
    batch_size = 32
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("Step 1: Loading data")
    print("="*80)
    macro_df, sentiment_df = load_data(include_sentiment=True)
    
    # Step 2: Prepare LSTM data
    print("\n" + "="*80)
    print("Step 2: Preparing LSTM data")
    print("="*80)
    
    # Prepare data with sentiment and ARIMA features
    data_dict = prepare_lstm_data(
        macro_df,
        sentiment_df,
        sequence_length=sequence_length,
        horizons=horizons,
        include_sentiment=True,
        train_split=train_split,
        include_arima=True,
        ar_lags=3
    )
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    target_scalers = data_dict['target_scalers']
    target_names = data_dict['target_names']
    
    # Step 3: Create validation split
    print("\n" + "="*80)
    print("Step 3: Creating validation split")
    print("="*80)
    X_train_split, X_val, y_train_split, y_val = create_validation_split(
        X_train, y_train, val_split=0.2
    )
    print(f"  Training: {len(X_train_split)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Step 4: Train LSTM models
    print("\n" + "="*80)
    print("Step 4: Training LSTM models")
    print("="*80)
    
    forecaster = LSTMForecaster(
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        epochs=epochs,
        batch_size=batch_size
    )
    
    forecasts = {}
    metrics = {}
    
    for h in horizons:
        print(f"\n{'='*60}")
        print(f"Training model for horizon h = {h} months")
        print(f"{'='*60}")
        
        # Prepare targets: combine growth and inflation into 2D array
        y_train_h = np.column_stack([
            y_train_split[h][target_names[0]],
            y_train_split[h][target_names[1]]
        ])
        y_val_h = np.column_stack([
            y_val[h][target_names[0]],
            y_val[h][target_names[1]]
        ])
        y_test_h = np.column_stack([
            y_test[h][target_names[0]],
            y_test[h][target_names[1]]
        ])
        
        # Train model
        model = forecaster.fit(
            X_train_split,
            y_train_h,
            X_val=X_val,
            y_val=y_val_h,
            horizon=h,
            verbose=1
        )
        
        # Generate forecasts
        print(f"\n  Generating forecasts...")
        
        # Extract AR features from test sequences (last timestep, AR columns)
        # AR features are typically: growth_factor_ar1, growth_factor_ar2, growth_factor_ar3, etc.
        # We'll use ar1 as the baseline AR prediction
        feature_names = data_dict['feature_names']
        ar_indices = []
        for var_name in target_names:
            for lag in range(1, 4):  # ar1, ar2, ar3
                ar_col = f'{var_name}_ar{lag}'
                if ar_col in feature_names:
                    ar_indices.append(feature_names.index(ar_col))
        
        # Get AR features from the last timestep of each sequence
        if len(ar_indices) >= 2:  # At least ar1 for both targets
            # Extract ar1 for each target (most recent lag)
            ar1_growth_idx = feature_names.index('growth_factor_ar1') if 'growth_factor_ar1' in feature_names else None
            ar1_inflation_idx = feature_names.index('inflation_factor_ar1') if 'inflation_factor_ar1' in feature_names else None
            
            if ar1_growth_idx is not None and ar1_inflation_idx is not None:
                # Get AR features from last timestep
                X_ar = np.column_stack([
                    X_test[:, -1, ar1_growth_idx],  # Last timestep, ar1 for growth
                    X_test[:, -1, ar1_inflation_idx]  # Last timestep, ar1 for inflation
                ])
            else:
                X_ar = None
        else:
            X_ar = None
        
        forecasts_h = forecaster.predict(X_test, h, X_ar_features=X_ar)
        forecasts[h] = forecasts_h
        
        # Inverse transform forecasts and actuals
        forecasts_inv = {}
        actuals_inv = {}
        
        for idx, var_name in enumerate(target_names):
            # Inverse transform forecasts
            forecast_scaled = forecasts_h[:, idx]
            forecast_inv = target_scalers[var_name].inverse_transform(
                forecast_scaled.reshape(-1, 1)
            ).ravel()
            forecasts_inv[var_name] = forecast_inv
            
            # Inverse transform actuals
            actual_scaled = y_test_h[:, idx]
            actual_inv = target_scalers[var_name].inverse_transform(
                actual_scaled.reshape(-1, 1)
            ).ravel()
            actuals_inv[var_name] = actual_inv
        
        # Compute metrics
        print(f"\n  Computing metrics...")
        metrics_h = {}
        for var_name in target_names:
            forecast_series = pd.Series(
                forecasts_inv[var_name],
                index=macro_df.index[len(macro_df) - len(forecasts_inv[var_name]):]
            )
            actual_series = pd.Series(
                actuals_inv[var_name],
                index=macro_df.index[len(macro_df) - len(actuals_inv[var_name]):]
            )
            
            metric = compute_forecast_metrics(forecast_series, actual_series, h)
            metric['horizon'] = h
            metric['variable'] = var_name
            
            if var_name not in metrics:
                metrics[var_name] = []
            metrics[var_name].append(metric)
            
            print(f"    {var_name.capitalize()}: RMSE={metric['rmse']:.4f}, MAE={metric['mae']:.4f}")
        
        # Plot learning curves
        if h in forecaster.history:
            lstm_output_dir = output_dir / "lstm"
            lstm_output_dir.mkdir(parents=True, exist_ok=True)
            plot_learning_curves(forecaster.history[h], h, output_dir=lstm_output_dir)
    
    # Step 5: Save forecast CSV files
    print("\n" + "="*80)
    print("Step 5: Saving forecast CSV files")
    print("="*80)
    
    # Get test dates (forecast origin dates)
    test_start_idx = len(macro_df) - len(X_test) - sequence_length
    test_dates = macro_df.index[test_start_idx + sequence_length:]
    
    # Create forecast CSV with columns: date, growth_h1, growth_h3, growth_h6, inflation_h1, inflation_h3, inflation_h6
    forecast_df = pd.DataFrame(index=test_dates[:len(X_test)])
    forecast_df.index.name = 'date'
    
    for h in horizons:
        # Get forecasts for this horizon
        forecasts_h = forecasts[h]  # Shape: (n_samples, 2) where 2 = [growth, inflation]
        
        # Extract growth and inflation forecasts
        growth_idx = target_names.index('growth_factor')
        inflation_idx = target_names.index('inflation_factor')
        
        # Inverse transform forecasts
        growth_scaled = forecasts_h[:, growth_idx]
        inflation_scaled = forecasts_h[:, inflation_idx]
        
        growth_inv = target_scalers['growth_factor'].inverse_transform(
            growth_scaled.reshape(-1, 1)
        ).ravel()
        inflation_inv = target_scalers['inflation_factor'].inverse_transform(
            inflation_scaled.reshape(-1, 1)
        ).ravel()
        
        # Store in DataFrame
        forecast_df[f'growth_h{h}'] = growth_inv[:len(forecast_df)]
        forecast_df[f'inflation_h{h}'] = inflation_inv[:len(forecast_df)]
    
    forecast_csv_path = output_dir / "lstm" / "forecasts_lstm.csv"
    forecast_csv_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_csv_path)
    print(f"Saved LSTM forecasts to {forecast_csv_path}")
    print(f"  Columns: {list(forecast_df.columns)}")
    print(f"  Rows: {len(forecast_df)}")
    
    # Step 6: Save metrics
    print("\n" + "="*80)
    print("Step 6: Saving metrics")
    print("="*80)
    
    for var_name in target_names:
        if var_name in metrics:
            df_metrics = pd.DataFrame(metrics[var_name])
            df_metrics.to_csv(output_dir / "lstm" / f"{var_name}_metrics_lstm.csv", index=False)
            print(f"Saved {var_name} metrics to {output_dir / 'lstm' / f'{var_name}_metrics_lstm.csv'}")
    
    # Step 7: Generate plots
    print("\n" + "="*80)
    print("Step 7: Generating plots")
    print("="*80)
    
    # Prepare forecast and actual dictionaries for plotting
    forecasts_plot = {}
    actuals_plot = {}
    
    # Get test dates (forecast origin dates)
    test_start_idx = len(macro_df) - len(X_test) - sequence_length
    test_dates = macro_df.index[test_start_idx + sequence_length:]
    
    # Get test dates (forecast origin dates)
    test_start_idx = len(macro_df) - len(X_test) - sequence_length
    test_dates = macro_df.index[test_start_idx + sequence_length:]
    
    for var_name in target_names:
        idx = target_names.index(var_name)
        
        # For each horizon, create forecast and actual series
        forecast_dict = {}
        actual_dict = {}
        
        for h in horizons:
            # Inverse transform forecasts
            forecast_scaled = forecasts[h][:, idx]
            forecast_inv = target_scalers[var_name].inverse_transform(
                forecast_scaled.reshape(-1, 1)
            ).ravel()
            
            # Create forecast series indexed by forecast origin dates
            forecast_series = pd.Series(forecast_inv, index=test_dates[:len(forecast_inv)])
            forecast_dict[h] = forecast_series
            
            # Get actuals at target dates (forecast_date + horizon)
            actual_values = []
            actual_dates = []
            for forecast_date in forecast_series.index:
                target_date = forecast_date + pd.DateOffset(months=h)
                if target_date in macro_df.index:
                    actual_val = macro_df.loc[target_date, var_name]
                    actual_values.append(actual_val)
                    actual_dates.append(target_date)
            
            if len(actual_values) > 0:
                actual_dict[h] = pd.Series(actual_values, index=actual_dates)
        
        forecasts_plot[var_name] = forecast_dict
        actuals_plot[var_name] = actual_dict
    
    # Plot forecasts vs realized
    lstm_output_dir = output_dir / "lstm"
    lstm_output_dir.mkdir(parents=True, exist_ok=True)
    plot_forecast_vs_realized_lstm(
        forecasts_plot,
        actuals_plot,
        horizons=horizons,
        start_date="2008-01-01",
        output_dir=lstm_output_dir
    )
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - lstm/forecasts_lstm.csv (forecast values)")
    print("  - lstm/*_metrics_lstm.csv")
    print("  - lstm/learning_curve_lstm_*.png")
    print("  - lstm/forecast_vs_realized_lstm_*.png")


if __name__ == "__main__":
    main()

