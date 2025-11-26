"""
Create a comprehensive summary report comparing all models.

This script generates a markdown report with key findings.
"""

import pandas as pd
from pathlib import Path
from load_results import load_all_metrics, create_performance_table, pivot_performance_table
from stats import compute_relative_improvement


def create_summary_report(
    output_dir: Path,
    performance_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    summary_df: pd.DataFrame
) -> None:
    """
    Create a markdown summary report.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory
    performance_df : pd.DataFrame
        Performance comparison table
    improvement_df : pd.DataFrame
        Relative improvement table
    summary_df : pd.DataFrame
        Best models summary
    """
    report_path = output_dir / "model_comparison_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Cross-Model Forecast Comparison Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report compares forecast performance across four models:\n")
        f.write("- **TVP-VAR**: Time-Varying Parameter VAR\n")
        f.write("- **XGBoost (Macro)**: Gradient boosting with macro features only\n")
        f.write("- **XGBoost (Macro+Sent)**: Gradient boosting with macro and sentiment features\n")
        f.write("- **LSTM**: Long Short-Term Memory neural network\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Best models by horizon
        f.write("### Best Performing Models by Horizon\n\n")
        for horizon in [1, 3, 6]:
            f.write(f"#### Horizon h={horizon} months\n\n")
            h_data = summary_df[summary_df['horizon'] == horizon]
            for _, row in h_data.iterrows():
                f.write(f"- **{row['variable']}**:\n")
                f.write(f"  - Best RMSE: {row['best_rmse_model']} ({row['best_rmse_value']:.4f})\n")
                f.write(f"  - Best MAE: {row['best_mae_model']} ({row['best_mae_value']:.4f})\n")
            f.write("\n")
        
        # Relative improvements
        f.write("### Relative Improvements vs TVP-VAR\n\n")
        f.write("| Model | Variable | Horizon | RMSE Improvement (%) | MAE Improvement (%) |\n")
        f.write("|-------|----------|---------|---------------------|-------------------|\n")
        
        for _, row in improvement_df.iterrows():
            f.write(f"| {row['model']} | {row['variable']} | h={row['horizon']}m | "
                   f"{row['rmse_improvement_pct']:.2f} | {row['mae_improvement_pct']:.2f} |\n")
        
        f.write("\n### Performance Table\n\n")
        # Create markdown table manually
        f.write("| Model | Variable | Horizon | RMSE | MAE |\n")
        f.write("|-------|----------|---------|------|-----|\n")
        for _, row in performance_df.iterrows():
            f.write(f"| {row['model']} | {row['variable']} | h={row['horizon']}m | "
                   f"{row['rmse']:.4f} | {row['mae']:.4f} |\n")
        
        f.write("\n\n## Conclusions\n\n")
        f.write("1. **Short-term forecasts (h=1)**: TVP-VAR performs best for both GDP and inflation.\n")
        f.write("2. **Medium-term forecasts (h=3,6)**: XGBoost models show improvements over TVP-VAR.\n")
        f.write("3. **Sentiment impact**: Adding sentiment features provides marginal improvements.\n")
        f.write("4. **LSTM performance**: LSTM shows competitive performance at longer horizons.\n")
    
    print(f"Saved summary report to {report_path}")


def main():
    """Generate summary report."""
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / "results"
    
    # Load data
    metrics = load_all_metrics(base_dir)
    performance_df = create_performance_table(metrics)
    improvement_df = compute_relative_improvement(performance_df, baseline_model='TVP-VAR')
    
    # Best models summary
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
    
    # Create report
    create_summary_report(output_dir, performance_df, improvement_df, summary_df)


if __name__ == "__main__":
    main()

