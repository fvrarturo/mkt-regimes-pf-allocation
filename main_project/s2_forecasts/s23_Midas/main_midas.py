"""
Main script for MIDAS TVP-VAR forecasting.

This script implements MIDAS-augmented TVP-VAR for forecasting growth and inflation
by combining monthly macro factors with daily oil prices aggregated via MIDAS.

Pipeline:
1. Load macro data and daily oil prices
2. Create MIDAS oil factor (exponential aggregation)
3. Prepare train/test split
4. Select optimal lag order
5. Fit MIDAS TVP-VAR with rolling/expanding window
6. Generate forecasts and evaluate performance
7. Plot results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from midas_preprocessing import (
    load_macro_data,
    load_daily_oil_data,
    prepare_midas_forecast_data,
    select_lag_order
)
from midas_tvpvar_model import MidasTVPVAR
from stats import compute_forecast_metrics, compare_forecasts


def main():
    """Main execution function."""
    print("="*80)
    print("MIDAS TVP-VAR Forecasting: Growth and Inflation")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    output_dir = Path(__file__).parent / "results_midas"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    horizons = [1, 3, 6]  # months
    train_split = 0.65  # 65% for training
    max_lags = 3  # Maximum lags to consider
    window_size = None  # None = expanding window, or specify number for rolling
    
    # MIDAS parameters
    theta = 0.03  # Decay parameter
    K = 60  # Number of daily lags
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("Step 1: Loading data")
    print("="*80)
    
    macro_df = load_macro_data()
    oil_prices = load_daily_oil_data()
    
    # Step 2: Prepare MIDAS forecast data
    print("\n" + "="*80)
    print("Step 2: Preparing MIDAS forecast data")
    print("="*80)
    
    train_data, test_data, full_data, test_dates = prepare_midas_forecast_data(
        macro_df,
        oil_prices,
        train_split=train_split,
        horizons=horizons,
        theta=theta,
        K=K
    )
    
    # Step 3: Select lag order
    print("\n" + "="*80)
    print("Step 3: Selecting optimal lag order")
    print("="*80)
    
    optimal_lag = select_lag_order(train_data, max_lags=max_lags, ic='bic')
    
    # Step 4: Fit MIDAS TVP-VAR and generate forecasts
    print("\n" + "="*80)
    print("Step 4: Fitting MIDAS TVP-VAR and generating forecasts")
    print("="*80)
    
    midas_model = MidasTVPVAR(
        lag_order=optimal_lag,
        window_size=window_size,
        min_window=60
    )
    
    # KEY: Use full_data to allow expanding window to include all available history
    midas_forecasts = midas_model.fit_forecast(
        train_data=full_data,  # Use FULL data, not just train_data
        test_dates=test_dates,
        horizons=horizons
    )
    
    # Step 5: Evaluate forecast performance
    print("\n" + "="*80)
    print("Step 5: Evaluating forecast performance")
    print("="*80)
    
    # Growth forecasts
    growth_metrics = compute_forecast_metrics(
        midas_forecasts['growth'],
        full_data['growth_factor'],
        horizons=horizons
    )
    print("\nGrowth forecasts:")
    print(growth_metrics.to_string(index=False))
    
    # Inflation forecasts
    inflation_metrics = compute_forecast_metrics(
        midas_forecasts['inflation'],
        full_data['inflation_factor'],
        horizons=horizons
    )
    print("\nInflation forecasts:")
    print(inflation_metrics.to_string(index=False))
    
    # Step 6: Save results
    print("\n" + "="*80)
    print("Step 6: Saving results")
    print("="*80)
    
    growth_metrics.to_csv(output_dir / "growth_forecast_metrics.csv", index=False)
    inflation_metrics.to_csv(output_dir / "inflation_forecast_metrics.csv", index=False)
    
    # Save forecasts
    midas_forecasts['growth'].to_csv(output_dir / "growth_forecasts.csv")
    midas_forecasts['inflation'].to_csv(output_dir / "inflation_forecasts.csv")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Step 6: Save results
    print("\n" + "="*80)
    print("Step 6: Saving results")
    print("="*80)
    
    growth_metrics.to_csv(output_dir / "growth_forecast_metrics.csv", index=False)
    inflation_metrics.to_csv(output_dir / "inflation_forecast_metrics.csv", index=False)
    
    # Save forecasts
    midas_forecasts['growth'].to_csv(output_dir / "growth_forecasts.csv")
    midas_forecasts['inflation'].to_csv(output_dir / "inflation_forecasts.csv")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Step 7: Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print("\nGrowth Factor Statistics:")
    print(f"  Mean:  {full_data['growth_factor'].mean():.6f}")
    print(f"  Std:   {full_data['growth_factor'].std():.6f}")
    print(f"  Min:   {full_data['growth_factor'].min():.6f}")
    print(f"  Max:   {full_data['growth_factor'].max():.6f}")
    
    print("\nInflation Factor Statistics:")
    print(f"  Mean:  {full_data['inflation_factor'].mean():.6f}")
    print(f"  Std:   {full_data['inflation_factor'].std():.6f}")
    print(f"  Min:   {full_data['inflation_factor'].min():.6f}")
    print(f"  Max:   {full_data['inflation_factor'].max():.6f}")
    
    print("\nOil MIDAS Factor Statistics:")
    print(f"  Mean:  {full_data['oil_midas'].mean():.6f}")
    print(f"  Std:   {full_data['oil_midas'].std():.6f}")
    print(f"  Min:   {full_data['oil_midas'].min():.6f}")
    print(f"  Max:   {full_data['oil_midas'].max():.6f}")
    
    print("\n" + "="*80)
    print("✓ MIDAS TVP-VAR forecasting completed!")
    print("="*80)


if __name__ == "__main__":
    main()
