"""
Main script for XGBoost forecasting of GDP and inflation.

This script implements Sections 5.2 and 5.2bis from goals.md:
- XGBoost models for growth and inflation forecasting
- Separate models for each (variable, horizon) pair
- Version A: Macro-only features
- Version B: Macro + sentiment features
- Comparison of performance with vs without sentiment

Outputs:
- Forecast performance tables (RMSE, MAE)
- Feature importance plots
- Forecast comparison plots
- Diebold-Mariano test results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from preprocessing import (
    load_data,
    prepare_features,
    create_targets,
    prepare_train_test_split
)
from xgboost_model import XGBoostForecaster
from stats import compute_forecast_metrics, compare_models
from plotting import (
    plot_feature_importance,
    plot_forecast_comparison,
    plot_rmse_comparison
)


def main():
    """Main execution function."""
    print("="*80)
    print("XGBoost Forecasting: GDP and Inflation (with and without Sentiment)")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    horizons = [1, 3, 6]  # months
    train_split = 0.65  # 65% for training
    macro_lags = 12  # Number of lags for macro variables
    sentiment_lags = 3  # Number of lags for sentiment variables
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("Step 1: Loading data")
    print("="*80)
    macro_df, sentiment_df = load_data(include_sentiment=True)
    
    # Step 2: Prepare features (macro-only)
    print("\n" + "="*80)
    print("Step 2: Preparing features (macro-only)")
    print("="*80)
    feature_df_macro, feature_names_macro = prepare_features(
        macro_df,
        sentiment_df=None,
        macro_lags=macro_lags,
        sentiment_lags=0,
        include_sentiment=False,
        include_arima=True,
        ar_lags=3
    )
    print(f"  Created {len(feature_names_macro)} macro features")
    
    # Step 3: Prepare features (macro + sentiment)
    print("\n" + "="*80)
    print("Step 3: Preparing features (macro + sentiment)")
    print("="*80)
    if sentiment_df is not None:
        feature_df_sentiment, feature_names_sentiment = prepare_features(
            macro_df,
            sentiment_df,
            macro_lags=macro_lags,
            sentiment_lags=sentiment_lags,
            include_sentiment=True,
            include_arima=True,
            ar_lags=3
        )
        print(f"  Created {len(feature_names_sentiment)} features (including sentiment)")
    else:
        print("  Warning: Sentiment data not available, skipping macro+sentiment models")
        feature_df_sentiment = None
        feature_names_sentiment = None
    
    # Step 4: Train models and generate forecasts
    print("\n" + "="*80)
    print("Step 4: Training XGBoost models and generating forecasts")
    print("="*80)
    
    variables = ['growth_factor', 'inflation_factor']
    variable_names = {'growth_factor': 'growth', 'inflation_factor': 'inflation'}
    
    # Store results
    forecasts_macro = {}
    forecasts_sentiment = {}
    metrics_macro = {var: [] for var in variables}
    metrics_sentiment = {var: [] for var in variables}
    feature_importance_macro = {}
    feature_importance_sentiment = {}
    dm_results = {}
    
    for var in variables:
        var_name = variable_names[var]
        print(f"\n{'='*60}")
        print(f"Processing {var_name.capitalize()}")
        print(f"{'='*60}")
        
        # Create targets
        target_df = create_targets(macro_df, var, horizons=horizons)
        
        # Macro-only models
        print(f"\nTraining macro-only models for {var_name}...")
        for h in horizons:
            print(f"\n  Horizon h = {h} months")
            
            # Prepare train/test split
            X_train, X_test, y_train, y_test = prepare_train_test_split(
                feature_df_macro,
                target_df[[f'target_h{h}']],
                train_split=train_split
            )
            
            # Remove NaN targets
            valid_idx = ~y_train[f'target_h{h}'].isna()
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
            
            if len(X_train) == 0:
                print(f"    Warning: No valid training data for {var_name}, h={h}")
                continue
            
            # Train model
            forecaster_macro = XGBoostForecaster()
            forecaster_macro.fit(
                X_train,
                y_train[f'target_h{h}'],
                variable=var_name,
                horizon=h,
                tune=True
            )
            
            # Generate forecasts
            forecasts = forecaster_macro.predict(X_test, var_name, h)
            forecast_series = pd.Series(forecasts, index=X_test.index)
            forecasts_macro[(var_name, h)] = forecast_series
            
            # Compute metrics
            actuals_series = macro_df[var]
            metrics = compute_forecast_metrics(forecast_series, actuals_series, h)
            metrics['horizon'] = h
            metrics['variable'] = var_name
            metrics_macro[var].append(metrics)
            
            print(f"    RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
            
            # Get feature importance
            importance = forecaster_macro.get_feature_importance(var_name, h)
            feature_importance_macro[(var_name, h)] = importance
        
        # Macro+sentiment models
        if feature_df_sentiment is not None:
            print(f"\nTraining macro+sentiment models for {var_name}...")
            for h in horizons:
                print(f"\n  Horizon h = {h} months")
                
                # Prepare train/test split
                X_train, X_test, y_train, y_test = prepare_train_test_split(
                    feature_df_sentiment,
                    target_df[[f'target_h{h}']],
                    train_split=train_split
                )
                
                # Remove NaN targets
                valid_idx = ~y_train[f'target_h{h}'].isna()
                X_train = X_train[valid_idx]
                y_train = y_train[valid_idx]
                
                if len(X_train) == 0:
                    print(f"    Warning: No valid training data for {var_name}, h={h}")
                    continue
                
                # Train model
                forecaster_sentiment = XGBoostForecaster()
                forecaster_sentiment.fit(
                    X_train,
                    y_train[f'target_h{h}'],
                    variable=var_name,
                    horizon=h,
                    tune=True
                )
                
                # Generate forecasts
                forecasts = forecaster_sentiment.predict(X_test, var_name, h)
                forecast_series = pd.Series(forecasts, index=X_test.index)
                forecasts_sentiment[(var_name, h)] = forecast_series
                
                # Compute metrics
                actuals_series = macro_df[var]
                metrics = compute_forecast_metrics(forecast_series, actuals_series, h)
                metrics['horizon'] = h
                metrics['variable'] = var_name
                metrics_sentiment[var].append(metrics)
                
                print(f"    RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
                
                # Get feature importance
                importance = forecaster_sentiment.get_feature_importance(var_name, h)
                feature_importance_sentiment[(var_name, h)] = importance
                
                # Compare models using Diebold-Mariano test
                if (var_name, h) in forecasts_macro:
                    actuals_series = macro_df[var]
                    dm_result = compare_models(
                        forecasts_macro[(var_name, h)],
                        forecasts_sentiment[(var_name, h)],
                        actuals_series,
                        h,
                        model1_name="Macro-only",
                        model2_name="Macro+Sentiment"
                    )
                    dm_result['variable'] = var_name
                    dm_result['horizon'] = h
                    dm_results[(var_name, h)] = dm_result
                    
                    print(f"    DM test: stat={dm_result['dm_statistic']:.3f}, "
                          f"p-value={dm_result['p_value']:.4f}")
    
    # Step 5: Save forecast CSV files
    print("\n" + "="*80)
    print("Step 5: Saving forecast CSV files")
    print("="*80)
    
    # Create forecast CSV for macro-only model
    # Get all unique dates from forecasts
    all_dates = set()
    for (var_name, h), forecast_series in forecasts_macro.items():
        all_dates.update(forecast_series.index)
    
    if len(all_dates) > 0:
        all_dates = sorted(all_dates)
        forecast_macro_df = pd.DataFrame(index=all_dates)
        forecast_macro_df.index.name = 'date'
        
        for var_name in ['growth', 'inflation']:
            for h in horizons:
                if (var_name, h) in forecasts_macro:
                    # Reindex to align with all_dates
                    forecast_series = forecasts_macro[(var_name, h)]
                    forecast_macro_df[f'{var_name}_h{h}'] = forecast_series.reindex(all_dates)
        
        forecast_macro_csv_path = output_dir / "xgboost" / "forecasts_xgboost_macro.csv"
        forecast_macro_csv_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_macro_df.to_csv(forecast_macro_csv_path)
        print(f"Saved XGBoost (macro-only) forecasts to {forecast_macro_csv_path}")
        print(f"  Columns: {list(forecast_macro_df.columns)}")
        print(f"  Rows: {len(forecast_macro_df)}")
    
    # Create forecast CSV for macro+sentiment model
    if forecasts_sentiment:
        all_dates_sent = set()
        for (var_name, h), forecast_series in forecasts_sentiment.items():
            all_dates_sent.update(forecast_series.index)
        
        if len(all_dates_sent) > 0:
            all_dates_sent = sorted(all_dates_sent)
            forecast_sentiment_df = pd.DataFrame(index=all_dates_sent)
            forecast_sentiment_df.index.name = 'date'
            
            for var_name in ['growth', 'inflation']:
                for h in horizons:
                    if (var_name, h) in forecasts_sentiment:
                        # Reindex to align with all_dates_sent
                        forecast_series = forecasts_sentiment[(var_name, h)]
                        forecast_sentiment_df[f'{var_name}_h{h}'] = forecast_series.reindex(all_dates_sent)
            
            forecast_sentiment_csv_path = output_dir / "xgboost" / "forecasts_xgboost_sentiment.csv"
            forecast_sentiment_csv_path.parent.mkdir(parents=True, exist_ok=True)
            forecast_sentiment_df.to_csv(forecast_sentiment_csv_path)
            print(f"Saved XGBoost (macro+sentiment) forecasts to {forecast_sentiment_csv_path}")
            print(f"  Columns: {list(forecast_sentiment_df.columns)}")
            print(f"  Rows: {len(forecast_sentiment_df)}")
    
    # Step 6: Save metrics
    print("\n" + "="*80)
    print("Step 6: Saving metrics")
    print("="*80)
    
    # Convert metrics to DataFrames
    for var in variables:
        var_name = variable_names[var]
        
        # Macro-only metrics
        if len(metrics_macro[var]) > 0:
            df_macro = pd.DataFrame(metrics_macro[var])
            df_macro.to_csv(output_dir / "xgboost" / f"{var_name}_metrics_macro.csv", index=False)
        
        # Macro+sentiment metrics
        if len(metrics_sentiment[var]) > 0:
            df_sentiment = pd.DataFrame(metrics_sentiment[var])
            df_sentiment.to_csv(output_dir / "xgboost" / f"{var_name}_metrics_sentiment.csv", index=False)
    
    # DM test results
    if dm_results:
        dm_df = pd.DataFrame(list(dm_results.values()))
        dm_df.to_csv(output_dir / "xgboost" / "dm_test_results.csv", index=False)
        print(f"Saved DM test results to {output_dir / 'xgboost' / 'dm_test_results.csv'}")
    
    # Step 7: Generate plots
    print("\n" + "="*80)
    print("Step 7: Generating plots")
    print("="*80)
    
    # Create xgboost subdirectory for plots
    xgboost_output_dir = output_dir / "xgboost"
    xgboost_output_dir.mkdir(parents=True, exist_ok=True)
    
    for var in variables:
        var_name = variable_names[var]
        
        # Feature importance plots
        for h in horizons:
            if (var_name, h) in feature_importance_macro:
                plot_feature_importance(
                    feature_importance_macro[(var_name, h)],
                    var_name,
                    h,
                    output_dir=xgboost_output_dir
                )
            
            if (var_name, h) in feature_importance_sentiment:
                plot_feature_importance(
                    feature_importance_sentiment[(var_name, h)],
                    f"{var_name}_sentiment",
                    h,
                    output_dir=xgboost_output_dir
                )
        
        # Forecast comparison plots
        for h in horizons:
            if (var_name, h) in forecasts_macro and (var_name, h) in forecasts_sentiment:
                actuals_series = macro_df[var]
                plot_forecast_comparison(
                    forecasts_macro[(var_name, h)],
                    forecasts_sentiment[(var_name, h)],
                    actuals_series,
                    var_name,
                    h,
                    start_date="2008-01-01",
                    output_dir=xgboost_output_dir
                )
        
        # RMSE/MAE comparison plots
        if len(metrics_macro[var]) > 0 and len(metrics_sentiment[var]) > 0:
            df_macro = pd.DataFrame(metrics_macro[var])
            df_sentiment = pd.DataFrame(metrics_sentiment[var])
            plot_rmse_comparison(
                df_macro,
                df_sentiment,
                var_name,
                output_dir=xgboost_output_dir
            )
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - xgboost/forecasts_xgboost_macro.csv (forecast values)")
    print("  - xgboost/forecasts_xgboost_sentiment.csv (forecast values)")
    print("  - xgboost/*_metrics_macro.csv")
    print("  - xgboost/*_metrics_sentiment.csv")
    print("  - xgboost/dm_test_results.csv")
    print("  - xgboost/feature_importance_*.png")
    print("  - xgboost/forecast_comparison_*.png")
    print("  - xgboost/rmse_mae_comparison_*.png")


if __name__ == "__main__":
    main()

