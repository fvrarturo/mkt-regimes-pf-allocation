"""
Main script for regime-based trading strategy.
Equivalent to basic regime.ipynb
"""

from pathlib import Path
from data_loader import load_all_data
from regime_identification import identify_regimes
from strategy_construction import construct_strategy
from performance_evaluation import plot_performance_comparison, plot_cumulative_returns

if __name__ == "__main__":
    # Set up paths
    # Go up: hailey testing -> old_work -> main_project
    base_dir = Path(__file__).parent.parent.parent  # Go up to main_project
    start_date = "1990-01-01"
    
    # Load data
    print("Loading data...")
    df, stock_df, bond_df = load_all_data(base_dir, start_date)
    
    # Identify regimes
    print("\nIdentifying regimes...")
    df, growth_threshold, inflation_threshold = identify_regimes(df, method='mode')
    
    # Construct strategy
    print("\nConstructing strategy...")
    df_strategy = construct_strategy(df, stock_df, bond_df)
    
    print(f"\nStrategy dataframe shape: {df_strategy.shape}")
    print(f"\nAllocation by Regime:")
    allocations = {
        'HG_HI': {'stocks': 0.60, 'bonds': 0.40},
        'HG_LI': {'stocks': 0.70, 'bonds': 0.30},
        'LG_HI': {'stocks': 0.35, 'bonds': 0.65},
        'LG_LI': {'stocks': 0.45, 'bonds': 0.55}
    }
    for regime, alloc in allocations.items():
        regime_count = (df_strategy['regime_lagged'] == regime).sum()
        print(f"  {regime}: {alloc['stocks']:.0%} stocks / {alloc['bonds']:.0%} bonds (n={regime_count})")
    
    # Evaluate performance
    print("\nEvaluating performance...")
    plot_performance_comparison(df_strategy)
    plot_cumulative_returns(df_strategy)

