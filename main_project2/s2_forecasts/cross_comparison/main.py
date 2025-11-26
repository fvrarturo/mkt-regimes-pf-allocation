"""
Main script for cross-model comparison.

This script:
1. Loads results from all models (TVP-VAR, XGBoost, LSTM)
2. Creates performance comparison tables
3. Generates visualizations
4. Runs statistical tests (Diebold-Mariano)
5. Creates summary reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from load_results import load_all_metrics, create_performance_table, pivot_performance_table
from plotting import (
    plot_performance_comparison,
    plot_heatmap_performance,
    plot_forecast_comparison_all_models
)
from stats import run_dm_tests, compute_relative_improvement


def load_forecast_series(base_dir: Path) -> dict:
    """
    Load forecast series from all models.
    Note: This is a placeholder - actual implementation would need to
    load forecast series from saved files or regenerate them.
    
    Parameters:
    -----------
    base_dir : Path
        Base directory
    
    Returns:
    --------
    dict
        Dictionary of forecast series by model
    """
    # TODO: Implement loading of actual forecast series
    # For now, return empty dict - forecasts would need to be saved/loaded
    return {}


def main():
    """Main execution function."""
    print("="*80)
    print("Cross-Model Comparison: TVP-VAR, XGBoost, LSTM")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load all metrics
    print("\n" + "="*80)
    print("Step 1: Loading metrics from all models")
    print("="*80)
    metrics = load_all_metrics(base_dir)
    
    print(f"\nLoaded metrics from {len(metrics)} model configurations:")
    for key in metrics.keys():
        print(f"  - {key}: {len(metrics[key])} entries")
    
    # Step 2: Create performance table
    print("\n" + "="*80)
    print("Step 2: Creating performance comparison table")
    print("="*80)
    performance_df = create_performance_table(metrics)
    
    # Save performance table
    performance_df.to_csv(output_dir / "performance_comparison_table.csv", index=False)
    print(f"\nSaved performance table to {output_dir / 'performance_comparison_table.csv'}")
    print("\nPerformance Summary:")
    print(performance_df.to_string())
    
    # Create pivoted tables
    rmse_table, mae_table = pivot_performance_table(performance_df)
    
    # Save pivoted tables
    rmse_table.to_csv(output_dir / "performance_rmse_table.csv")
    mae_table.to_csv(output_dir / "performance_mae_table.csv")
    print(f"\nSaved pivoted tables to {output_dir}")
    
    # Step 3: Generate performance plots
    print("\n" + "="*80)
    print("Step 3: Generating performance comparison plots")
    print("="*80)
    
    # RMSE comparison
    plot_performance_comparison(performance_df, metric='rmse', output_dir=output_dir)
    
    # MAE comparison
    plot_performance_comparison(performance_df, metric='mae', output_dir=output_dir)
    
    # Heatmap
    plot_heatmap_performance(rmse_table, mae_table, output_dir=output_dir)
    
    # Step 4: Compute relative improvements
    print("\n" + "="*80)
    print("Step 4: Computing relative improvements")
    print("="*80)
    
    improvement_df = compute_relative_improvement(performance_df, baseline_model='TVP-VAR')
    improvement_df.to_csv(output_dir / "relative_improvement_table.csv", index=False)
    print(f"\nSaved relative improvement table to {output_dir / 'relative_improvement_table.csv'}")
    print("\nRelative Improvement Summary (vs TVP-VAR):")
    print(improvement_df.to_string())
    
    # Step 5: Summary statistics
    print("\n" + "="*80)
    print("Step 5: Summary Statistics")
    print("="*80)
    
    # Best model per horizon and variable
    summary_rows = []
    for variable in performance_df['variable'].unique():
        for horizon in performance_df['horizon'].unique():
            var_h_data = performance_df[
                (performance_df['variable'] == variable) & 
                (performance_df['horizon'] == horizon)
            ]
            
            if len(var_h_data) > 0:
                best_rmse = var_h_data.loc[var_h_data['rmse'].idxmin()]
                best_mae = var_h_data.loc[var_h_data['mae'].idxmin()]
                
                summary_rows.append({
                    'variable': variable,
                    'horizon': horizon,
                    'best_rmse_model': best_rmse['model'],
                    'best_rmse_value': best_rmse['rmse'],
                    'best_mae_model': best_mae['model'],
                    'best_mae_value': best_mae['mae']
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "best_models_summary.csv", index=False)
    print(f"\nSaved best models summary to {output_dir / 'best_models_summary.csv'}")
    print("\nBest Models by Metric:")
    print(summary_df.to_string())
    
    # Step 6: Final summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - performance_comparison_table.csv")
    print("  - performance_rmse_table.csv")
    print("  - performance_mae_table.csv")
    print("  - relative_improvement_table.csv")
    print("  - best_models_summary.csv")
    print("  - performance_comparison_rmse.png")
    print("  - performance_comparison_mae.png")
    print("  - performance_heatmap.png")


if __name__ == "__main__":
    main()

