"""
Main script for ARX(3) model.

This script orchestrates the full pipeline:
1. Data loading
2. Feature engineering
3. Model training
4. Trading strategy evaluation
5. Visualization
"""

import matplotlib.pyplot as plt
import pandas as pd

from new_ARX3_data_loader import prepare_model_data
from new_ARX3_feature_engineering import build_arx_features, split_train_val_test
from new_ARX3_models import train_linear_model, train_lasso_model
from new_ARX3_trading import (
    compute_trading_strategies, 
    compute_benchmarks, 
    evaluate_strategies
)
from new_ARX3_plotting import plot_actual_vs_predicted, plot_cumulative_returns

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (12, 5)


def main():
    """Main execution function."""
    print("=" * 80)
    print("ARX(3) Model Pipeline")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\n[1/5] Loading data...")
    df = prepare_model_data()
    print(f"Final modeling dataframe: {df.shape}")
    
    # Step 2: Build ARX features
    print("\n[2/5] Building ARX(3) features...")
    arx_df = build_arx_features(df)
    print(f"ARX dataframe: {arx_df.shape}")
    
    # Step 3: Split data
    print("\n[3/5] Splitting data into train/val/test...")
    splits = split_train_val_test(arx_df)
    print(f"Train: {splits['X_train'].shape}")
    print(f"Val:   {splits['X_val'].shape}")
    print(f"Test:  {splits['X_test'].shape}")
    
    # Step 4: Train models
    print("\n[4/5] Training models...")
    
    # Linear Regression
    lin_results = train_linear_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        splits["X_test"], splits["y_test"]
    )
    
    # LASSO
    lasso_results = train_lasso_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        splits["X_test"], splits["y_test"]
    )
    
    # Step 5: Trading strategies
    print("\n[5/5] Evaluating trading strategies...")
    
    # Linear Regression strategies
    strategy_A_lin = compute_trading_strategies(
        lin_results["y_hat_train"],
        lin_results["y_hat_test"],
        splits["ret_next_test"],
        strategy_type="bands"
    )
    
    strategy_B_lin = compute_trading_strategies(
        lin_results["y_hat_train"],
        lin_results["y_hat_test"],
        splits["ret_next_test"],
        strategy_type="sign"
    )
    
    benchmarks_lin = compute_benchmarks(
        splits["ret_next_test"],
        avg_weight=strategy_A_lin["weights_stock"].mean()
    )
    
    print("\n--- Linear Regression Strategies ---")
    evaluate_strategies(strategy_A_lin, strategy_B_lin, benchmarks_lin)
    
    # LASSO strategies
    strategy_A_lasso = compute_trading_strategies(
        lasso_results["y_hat_train"],
        lasso_results["y_hat_test"],
        splits["ret_next_test"],
        strategy_type="bands"
    )
    
    strategy_B_lasso = compute_trading_strategies(
        lasso_results["y_hat_train"],
        lasso_results["y_hat_test"],
        splits["ret_next_test"],
        strategy_type="sign"
    )
    
    benchmarks_lasso = compute_benchmarks(
        splits["ret_next_test"],
        avg_weight=strategy_A_lasso["weights_stock"].mean()
    )
    
    print("\n--- LASSO Strategies ---")
    evaluate_strategies(strategy_A_lasso, strategy_B_lasso, benchmarks_lasso)
    
    # Step 6: Visualizations
    print("\nGenerating plots...")
    
    # Plot Linear Regression predictions
    plot_actual_vs_predicted(
        splits["y_test"],
        lin_results["y_hat_test"],
        title="ARX(3) Model (Test Set)",
        model_name="ARX(3)",
        color="blue"
    )
    
    # Plot LASSO predictions
    plot_actual_vs_predicted(
        splits["y_test"],
        lasso_results["y_hat_test"],
        title="LASSO ARX(3) Model (Test Set)",
        model_name="LASSO ARX(3)",
        color="red"
    )
    
    # Plot cumulative returns (Linear)
    plot_cumulative_returns(
        strategy_A_lin["returns"],
        strategy_B_lin["returns"],
        benchmarks_lin,
        title="ARX(3) Model (Test Period)"
    )
    
    # Plot cumulative returns (LASSO)
    plot_cumulative_returns(
        strategy_A_lasso["returns"],
        strategy_B_lasso["returns"],
        benchmarks_lasso,
        title="LASSO ARX(3) Model (Test Period)"
    )
    
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

