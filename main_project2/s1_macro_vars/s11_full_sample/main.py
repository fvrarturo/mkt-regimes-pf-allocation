"""
Main script for full-sample regression analysis (Section 4.1).

Simple regression: ERP_{t+h} = α + β' X_t + ε_{t+h}
- X_t: 4 macro indices (inflation, growth, monetary policy, market volatility)
- Horizons h = 1, 3, 6, 12, 24 months

Outputs:
- Tables of coefficients, t-stats, R² (per horizon)
- Variable importance ranking based on |t-stat|
"""

import pandas as pd
import numpy as np
from pathlib import Path
from regression import run_full_sample_regressions
from plotting import create_results_tables, create_plots


def load_data():
    """Load ERP and macro factors data."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Load ERP
    erp_path = base_dir / "data" / "macro_processed" / "equity_risk_pr.csv"
    erp_df = pd.read_csv(erp_path, parse_dates=["date"])
    erp_df = erp_df.set_index("date").sort_index()
    
    # Load macro factors
    macro_path = base_dir / "data" / "macro_final" / "final_macro.csv"
    macro_df = pd.read_csv(macro_path, parse_dates=["date"])
    macro_df = macro_df.set_index("date").sort_index()
    
    return erp_df, macro_df


def prepare_regression_data(erp_df, macro_df):
    """
    Prepare data for regression by merging and aligning dates.
    
    Parameters:
    -----------
    erp_df : pd.DataFrame
        ERP data with 'ERP' column
    macro_df : pd.DataFrame
        Macro factors data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with ERP and predictors
    """
    # Normalize macro dates to month-end to match ERP
    macro_df_normalized = macro_df.copy()
    if isinstance(macro_df_normalized.index, pd.DatetimeIndex):
        # Convert month-start to month-end
        macro_df_normalized.index = macro_df_normalized.index.to_period('M').to_timestamp('M')
    else:
        # If date is a column, convert it
        if 'date' in macro_df_normalized.columns:
            macro_df_normalized['date'] = pd.to_datetime(macro_df_normalized['date'])
            macro_df_normalized['date'] = macro_df_normalized['date'].dt.to_period('M').dt.to_timestamp('M')
            macro_df_normalized = macro_df_normalized.set_index('date')
    
    # Merge ERP and macro (both should now have month-end dates)
    df = erp_df[["ERP"]].join(macro_df_normalized, how="inner")
    
    # Drop rows with any NaN
    df = df.dropna()
    
    return df


def main():
    """Main execution function."""
    print("="*70)
    print("Full-Sample Regression Analysis (Section 4.1)")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    erp_df, macro_df = load_data()
    
    print(f"   ERP data: {len(erp_df)} observations")
    print(f"   Macro data: {len(macro_df)} observations")
    
    # Prepare data
    print("\n2. Preparing regression data...")
    df = prepare_regression_data(erp_df, macro_df)
    print(f"   Merged data: {len(df)} observations")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Run regressions
    print("\n3. Running full-sample regressions...")
    horizons = [1, 3, 6, 12, 24]  # 1m, 3m, 6m, 1y, 2y
    results = run_full_sample_regressions(df, horizons=horizons)
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    
    # Generate tables
    print("\n4. Generating output tables...")
    results_df, summary_df, importance_df = create_results_tables(results, output_dir)
    
    # Generate plots
    print("\n5. Generating plots...")
    create_plots(results, output_dir)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - regression_results_all_horizons.csv")
    print("  - regression_summary.csv")
    print("  - variable_importance_ranking.csv")
    print("  - variable_importance_ranking.png")
    print("  - coefficient_comparison.png")
    print("  - r_squared_by_horizon.png")


if __name__ == "__main__":
    main()

