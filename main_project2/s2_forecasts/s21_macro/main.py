"""
Main script for TVP-VAR forecasting of GDP and inflation.

This script implements Section 5.1 from goals.md:
- 4-variable TVP-VAR (growth, inflation, policy, volatility)
- Forecast horizons: 1, 3, 6 months
- Out-of-sample evaluation
- Time-varying coefficients analysis
- Impulse response functions

Outputs:
- Forecast performance tables (RMSE, MAE)
- Time-varying coefficient plots
- Impulse response snapshots
- Forecast vs realized plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from preprocessing import load_macro_data, prepare_forecast_data, select_lag_order
from tvpvar_model import RollingTVPVAR
from stats import compute_forecast_metrics, compare_forecasts, fit_static_var
from plotting import (
    plot_forecast_vs_realized,
    plot_time_varying_coefficients,
    plot_forecast_performance,
    plot_aggregated_irfs
)


def main():
    """Main execution function."""
    print("="*80)
    print("TVP-VAR Forecasting: GDP and Inflation")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    horizons = [1, 3, 6]  # months
    train_split = 0.65  # 65% for training
    max_lags = 3  # Maximum lags to consider for lag selection
    window_size = None  # None = expanding window, or specify number for rolling window
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("Step 1: Loading macro data")
    print("="*80)
    macro_df = load_macro_data()
    
    # Step 2: Prepare forecast data
    print("\n" + "="*80)
    print("Step 2: Preparing forecast data")
    print("="*80)
    train_data, test_data, full_data, test_dates = prepare_forecast_data(
        macro_df, train_split=train_split, horizons=horizons
    )
    
    # Step 3: Select lag order
    print("\n" + "="*80)
    print("Step 3: Selecting optimal lag order")
    print("="*80)
    optimal_lag = select_lag_order(train_data, max_lags=max_lags, ic='bic')
    
    # Step 4: Fit TVP-VAR and generate forecasts
    print("\n" + "="*80)
    print("Step 4: Fitting TVP-VAR and generating forecasts")
    print("="*80)
    
    # Initialize model
    tvpvar_model = RollingTVPVAR(
        lag_order=optimal_lag,
        window_size=window_size,
        min_window=60  # Minimum 60 months for estimation
    )
    
    # Generate forecasts
    forecasts = tvpvar_model.fit_forecast(
        train_data=full_data,  # Use full data up to each forecast date
        test_dates=test_dates,
        horizons=horizons
    )
    
    # Step 5: Save forecast CSV files
    print("\n" + "="*80)
    print("Step 5: Saving forecast CSV files")
    print("="*80)
    
    # Create forecast CSV with columns: date, growth_h1, growth_h3, growth_h6, inflation_h1, inflation_h3, inflation_h6
    forecast_df = pd.DataFrame(index=forecasts['growth'].index)
    forecast_df.index.name = 'date'
    
    for h in horizons:
        forecast_df[f'growth_h{h}'] = forecasts['growth'][f'h_{h}']
        forecast_df[f'inflation_h{h}'] = forecasts['inflation'][f'h_{h}']
    
    forecast_csv_path = output_dir / "forecasts_tvpvar.csv"
    forecast_df.to_csv(forecast_csv_path)
    print(f"Saved TVP-VAR forecasts to {forecast_csv_path}")
    print(f"  Columns: {list(forecast_df.columns)}")
    print(f"  Rows: {len(forecast_df)}")
    
    # Step 6: Compute forecast metrics
    print("\n" + "="*80)
    print("Step 6: Computing forecast performance metrics")
    print("="*80)
    
    # Extract actual values
    growth_actuals = full_data['growth_factor']
    inflation_actuals = full_data['inflation_factor']
    
    # Compute metrics for growth
    growth_metrics = compute_forecast_metrics(
        forecasts['growth'],
        growth_actuals,
        horizons=horizons
    )
    print("\nGrowth Forecast Performance:")
    print(growth_metrics.to_string(index=False))
    
    # Compute metrics for inflation
    inflation_metrics = compute_forecast_metrics(
        forecasts['inflation'],
        inflation_actuals,
        horizons=horizons
    )
    print("\nInflation Forecast Performance:")
    print(inflation_metrics.to_string(index=False))
    
    # Save metrics
    growth_metrics.to_csv(output_dir / "growth_forecast_metrics.csv", index=False)
    inflation_metrics.to_csv(output_dir / "inflation_forecast_metrics.csv", index=False)
    
    # Combine metrics
    combined_metrics = pd.concat([
        growth_metrics.assign(variable='growth'),
        inflation_metrics.assign(variable='inflation')
    ], ignore_index=True)
    combined_metrics.to_csv(output_dir / "forecast_performance_table.csv", index=False)
    
    # Step 7: Compare with static VAR (optional)
    print("\n" + "="*80)
    print("Step 7: Comparing with static VAR")
    print("="*80)
    
    try:
        # Fit static VAR on training data
        static_var = fit_static_var(train_data, optimal_lag)
        
        # Generate static VAR forecasts
        static_forecasts = {
            'growth': pd.DataFrame(index=test_dates, columns=[f'h_{h}' for h in horizons]),
            'inflation': pd.DataFrame(index=test_dates, columns=[f'h_{h}' for h in horizons])
        }
        
        var_names = list(train_data.columns)
        growth_idx = var_names.index('growth_factor')
        inflation_idx = var_names.index('inflation_factor')
        
        for forecast_date in test_dates:
            available_data = full_data.loc[:forecast_date]
            
            if len(available_data) < optimal_lag + max(horizons):
                continue
            
            # Use last lag_order observations for forecast
            last_data = available_data.values[-optimal_lag:]
            
            for h in horizons:
                try:
                    forecast = static_var.forecast(last_data, steps=h)
                    static_forecasts['growth'].loc[forecast_date, f'h_{h}'] = forecast[-1, growth_idx]
                    static_forecasts['inflation'].loc[forecast_date, f'h_{h}'] = forecast[-1, inflation_idx]
                except:
                    continue
        
        # Compare forecasts
        print("\nComparing TVP-VAR vs Static VAR:")
        for var_name in ['growth', 'inflation']:
            comparison = compare_forecasts(
                forecasts[var_name],
                static_forecasts[var_name],
                growth_actuals if var_name == 'growth' else inflation_actuals,
                horizons=horizons,
                model1_name="TVP-VAR",
                model2_name="Static VAR"
            )
            print(f"\n{var_name.capitalize()}:")
            print(comparison.to_string(index=False))
            comparison.to_csv(output_dir / f"{var_name}_tvpvar_vs_static_comparison.csv", index=False)
        
    except Exception as e:
        print(f"Warning: Could not compare with static VAR: {e}")
        static_forecasts = None
    
    # Step 8: Generate plots
    print("\n" + "="*80)
    print("Step 8: Generating plots")
    print("="*80)
    
    # Forecast vs realized plots (starting from 2008-2009)
    plot_forecast_vs_realized(
        forecasts['growth'],
        growth_actuals,
        horizons=horizons,
        variable_name="Growth Factor",
        start_date="2008-01-01",
        output_dir=output_dir
    )
    
    plot_forecast_vs_realized(
        forecasts['inflation'],
        inflation_actuals,
        horizons=horizons,
        variable_name="Inflation Factor",
        start_date="2008-01-01",
        output_dir=output_dir
    )
    
    # Forecast performance plots (combined)
    plot_forecast_performance(
        growth_metrics,
        inflation_metrics,
        output_dir=output_dir
    )
    
    # Time-varying coefficients plots
    print("\nPlotting time-varying coefficients...")
    for var_name in ['growth_factor', 'inflation_factor']:
        try:
            coefs = tvpvar_model.get_time_varying_coefficients(var_name)
            # Plot intercept (coef_0) and a few key coefficients
            plot_time_varying_coefficients(
                coefs[['coef_0']],  # Just plot intercept for now
                variable_name=var_name,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Warning: Could not plot coefficients for {var_name}: {e}")
    
    # Impulse response functions (for selected dates)
    print("\nComputing impulse response functions...")
    sample_dates = test_dates[::max(1, len(test_dates) // 3)]  # Sample 3 dates
    sample_dates = sample_dates[:3]  # Limit to 3 dates
    
    # Collect IRF data for aggregated plots
    policy_irf_dict = {}
    vol_irf_dict = {}
    
    for forecast_date in sample_dates:
        try:
            # Policy shock
            irf_policy, lower_policy, upper_policy = tvpvar_model.compute_impulse_responses(
                forecast_date,
                periods=20,
                shock_var='monetary_policy_factor',
                response_vars=['growth_factor', 'inflation_factor'],
                include_ci=True,
                ci_level=0.95
            )
            policy_irf_dict[forecast_date] = (irf_policy, lower_policy, upper_policy)
            
            # Volatility shock
            irf_vol, lower_vol, upper_vol = tvpvar_model.compute_impulse_responses(
                forecast_date,
                periods=20,
                shock_var='market_volatility_factor',
                response_vars=['growth_factor', 'inflation_factor'],
                include_ci=True,
                ci_level=0.95
            )
            vol_irf_dict[forecast_date] = (irf_vol, lower_vol, upper_vol)
        except Exception as e:
            print(f"Warning: Could not compute IRF for {forecast_date}: {e}")
    
    # Create aggregated IRF plots
    if policy_irf_dict:
        print("\nCreating aggregated IRF plots...")
        plot_aggregated_irfs(
            policy_irf_dict,
            shock_var='monetary_policy_factor',
            response_vars=['growth_factor', 'inflation_factor'],
            forecast_dates=list(policy_irf_dict.keys()),
            output_dir=output_dir
        )
    
    if vol_irf_dict:
        plot_aggregated_irfs(
            vol_irf_dict,
            shock_var='market_volatility_factor',
            response_vars=['growth_factor', 'inflation_factor'],
            forecast_dates=list(vol_irf_dict.keys()),
            output_dir=output_dir
        )
    
    # Note: Forecast comparison plots removed per user request
    
    # Step 9: Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - forecasts_tvpvar.csv (forecast values)")
    print("  - forecast_performance_table.csv")
    print("  - growth_forecast_metrics.csv")
    print("  - inflation_forecast_metrics.csv")
    print("  - forecast_vs_realized_*.png")
    print("  - forecast_performance_*.png")
    print("  - time_varying_coefficients_*.png")
    print("  - irf_*.png")
    if static_forecasts is not None:
        print("  - *_tvpvar_vs_static_comparison.csv")
        print("  - forecast_comparison_*.png")


if __name__ == "__main__":
    main()

