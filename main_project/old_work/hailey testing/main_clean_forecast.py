"""
Main script for data cleaning and forecasting.
Equivalent to clean + forecast.ipynb
"""

from pathlib import Path
import pandas as pd
from data_merger import prepare_forecasting_data
from forecast_models import (
    prepare_train_test_split,
    train_random_forest,
    train_xgboost,
    train_individual_models,
    plot_feature_importance
)

if __name__ == "__main__":
    # Set up paths
    # Go up: hailey testing -> old_work -> main_project
    base_dir = Path(__file__).parent.parent.parent  # Go up to main_project
    
    yahoo_path = base_dir / "data" / "forecasting data" / "macro_yahoo_vix_yieldcurve_dxy.csv"
    macro_merged_path = base_dir / "data" / "forecasting data" / "merged_macro_dataset.csv"
    tips_path = base_dir / "data" / "forecasting data" / "us_treasury_tips_breakeven.csv"
    macro_final_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    
    # Prepare forecasting data
    print("Preparing forecasting data...")
    combined_df = prepare_forecasting_data(
        yahoo_path,
        macro_merged_path,
        tips_path,
        macro_final_path,
        start_date="1980-01-01"
    )
    
    # Define features and targets
    feature_cols = [
        col for col in combined_df.columns
        if col not in ['date', 'month'] and col not in [
            'inflation_factor', 'growth_factor',
            'monetary_policy_factor', 'market_volatility_factor'
        ]
    ]
    target_cols = [
        'inflation_factor', 'growth_factor',
        'monetary_policy_factor', 'market_volatility_factor'
    ]
    
    print(f"\nFeature columns: {feature_cols}")
    print(f"Target columns: {target_cols}")
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        combined_df, feature_cols, target_cols, test_size=0.2, shuffle=False
    )
    
    print(f"\nTrain set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # Train Random Forest
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)
    rf_model, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    print(f"\nTraining Set:")
    print(f"  R² Score: {rf_results['train_r2']:.4f}")
    print(f"  RMSE: {rf_results['train_rmse']:.4f}")
    print(f"  MAE: {rf_results['train_mae']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  R² Score: {rf_results['test_r2']:.4f}")
    print(f"  RMSE: {rf_results['test_rmse']:.4f}")
    print(f"  MAE: {rf_results['test_mae']:.4f}")
    
    print(f"\nOverfitting Gap: {rf_results['overfitting_gap']:.4f}")
    print(f"OOB Score: {rf_results['oob_score']:.4f}")
    
    # Plot feature importance
    plot_feature_importance(
        rf_model,
        feature_cols,
        title="Random Forest Feature Importance",
        top_n=15
    )
    
    # Train XGBoost
    print("\n" + "="*60)
    print("Training XGBoost Model")
    print("="*60)
    xgb_model, xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    print(f"\nTraining Set:")
    print(f"  R² Score: {xgb_results['train_r2']:.4f}")
    print(f"  RMSE: {xgb_results['train_rmse']:.4f}")
    print(f"  MAE: {xgb_results['train_mae']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  R² Score: {xgb_results['test_r2']:.4f}")
    print(f"  RMSE: {xgb_results['test_rmse']:.4f}")
    print(f"  MAE: {xgb_results['test_mae']:.4f}")
    
    print(f"\nOverfitting Gap: {xgb_results['overfitting_gap']:.4f}")
    
    # Plot feature importance
    plot_feature_importance(
        xgb_model,
        feature_cols,
        title="XGBoost Feature Importance",
        top_n=15
    )
    
    # Train individual models
    print("\n" + "="*60)
    print("Training Individual Models for Each Target")
    print("="*60)
    individual_models = train_individual_models(
        combined_df, feature_cols, target_cols, test_size=0.2
    )
    
    for target_var, (model, results) in individual_models.items():
        print(f"\n{target_var}:")
        print(f"  Train R²: {results['train_r2']:.4f}, RMSE: {results['train_rmse']:.4f}")
        print(f"  Test R²: {results['test_r2']:.4f}, RMSE: {results['test_rmse']:.4f}")
        print(f"  Overfitting Gap: {results['overfitting_gap']:.4f}")

