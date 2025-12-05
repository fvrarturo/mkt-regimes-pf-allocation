"""
LASSO-based conditional regression trading strategy with monthly retraining.
Tracks variable inclusion over time and generates visualizations.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from data_loader import load_market_data, load_all_macro_variables
from performance import compute_performance_metrics
from plotting import plot_cumulative_returns_all_strategies
from trading import run_trading_strategy
from lasso_conditional_forecasts import LassoConditionalForecaster
from plot_lasso_variables import plot_hmm_variable_inclusion_weighted, plot_variable_inclusion_over_time

START_DATE = pd.Timestamp("2002-03-31")
RESULTS_DIR = Path(__file__).parent / "results" / "lasso_strategies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main function to run LASSO conditional regression strategies."""
    print("="*80)
    print("LASSO CONDITIONAL REGRESSION TRADING STRATEGY")
    print("="*80)
    
    # Load data
    print("\nLoading market data...")
    equity_ret, bond_ret, erp = load_market_data()
    
    print("\nLoading macro variables...")
    all_macro_df = load_all_macro_variables()
    
    # Align to common date index
    common_index = equity_ret.index.intersection(all_macro_df.index).intersection(erp.index)
    equity_ret = equity_ret.reindex(common_index)
    bond_ret = bond_ret.reindex(common_index)
    erp = erp.reindex(common_index)
    all_macro_df = all_macro_df.reindex(common_index)
    
    # Initialize LASSO forecaster
    print("\nInitializing LASSO conditional forecaster...")
    base_dir = Path(__file__).parent.parent.parent
    
    forecaster = LassoConditionalForecaster(
        hmm_combination="2vars_inflation_market_volatility",
        hmm_k=4,
        alpha_range=(0.001, 1.0),
        n_alphas=50,
        cv_folds=5
    )
    
    # Generate forecasts with monthly retraining
    # Pass macro_df directly to avoid reloading
    print("\nGenerating forecasts with monthly retraining...")
    hmm_forecasts, two_by_two_forecasts = forecaster.forecast_rolling(
        erp=erp,
        macro_df=all_macro_df,
        start_date=START_DATE,
        base_dir=base_dir
    )
    
    # Run trading strategies
    print("\nRunning trading strategies...")
    strategies = {}
    
    # HMM LASSO strategy
    if not hmm_forecasts.empty:
        hmm_forecasts_filtered = hmm_forecasts[hmm_forecasts.index >= START_DATE].dropna()
        if not hmm_forecasts_filtered.empty:
            hmm_strategy = run_trading_strategy(
                name="hmm_lasso",
                forecasts=hmm_forecasts_filtered,
                equity_returns=equity_ret,
                bond_returns=bond_ret,
                min_weight=0.0,
                max_weight=1.0
            )
            strategies["hmm_lasso"] = {
                "returns": hmm_strategy.returns,
                "weights": hmm_strategy.weights,
                "forecast": hmm_strategy.forecast,
                "metrics": compute_performance_metrics(hmm_strategy.returns)
            }
            print(f"✓ HMM LASSO strategy: {len(hmm_strategy.returns)} periods")
    
    # 2x2 LASSO strategy
    if not two_by_two_forecasts.empty:
        two_by_two_forecasts_filtered = two_by_two_forecasts[two_by_two_forecasts.index >= START_DATE].dropna()
        if not two_by_two_forecasts_filtered.empty:
            two_by_two_strategy = run_trading_strategy(
                name="2x2_lasso",
                forecasts=two_by_two_forecasts_filtered,
                equity_returns=equity_ret,
                bond_returns=bond_ret,
                min_weight=0.1,
                max_weight=0.9
            )
            strategies["2x2_lasso"] = {
                "returns": two_by_two_strategy.returns,
                "weights": two_by_two_strategy.weights,
                "forecast": two_by_two_strategy.forecast,
                "metrics": compute_performance_metrics(two_by_two_strategy.returns)
            }
            print(f"✓ 2x2 LASSO strategy: {len(two_by_two_strategy.returns)} periods")
    
    # Save strategy returns
    print("\nSaving strategy returns...")
    for name, strategy_data in strategies.items():
        returns_df = pd.DataFrame({
            "date": strategy_data["returns"].index,
            "return": strategy_data["returns"].values,
            "weight": strategy_data["weights"].values,
            "forecast": strategy_data["forecast"].values
        })
        returns_df.to_csv(RESULTS_DIR / f"{name}_returns.csv", index=False)
        print(f"✓ Saved {name}_returns.csv")
    
    # Save performance metrics
    print("\nComputing performance metrics...")
    metrics_rows = []
    for name, strategy_data in strategies.items():
        metrics = strategy_data["metrics"].copy()
        metrics["strategy"] = name
        metrics_rows.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "strategy_performance_summary.csv", index=False)
    print("\nPerformance Summary:")
    print(metrics_df.to_string(index=False))
    
    # Plot cumulative returns
    print("\nPlotting cumulative returns...")
    plot_cumulative_returns_all_strategies(
        strategies,
        output_dir=RESULTS_DIR
    )
    
    # Plot variable inclusion over time
    print("\nPlotting variable inclusion over time...")
    
    # Get macro variable names
    macro_vars = list(forecaster.macro_variables.columns)
    
    # Plot HMM variable inclusion (weighted by regime probabilities)
    if forecaster.hmm_variable_inclusion and forecaster.hmm_regime_probabilities:
        plot_hmm_variable_inclusion_weighted(
            variable_inclusion_history=forecaster.hmm_variable_inclusion,
            regime_probabilities_history=forecaster.hmm_regime_probabilities,
            macro_variables=macro_vars,
            output_path=RESULTS_DIR / "hmm_lasso_variable_inclusion.png"
        )
    
    # Plot 2x2 variable inclusion (binary)
    if forecaster.two_by_two_variable_inclusion:
        # For 2x2, we need to determine which regime is active at each date
        # We'll mark a variable as included if it's in the active regime at that date
        two_by_two_binary_inclusion = {}
        
        # Get 2x2 regime assignments for each date
        macro_final_path = base_dir / 'data' / 'macro_final' / 'final_macro.csv'
        macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
        macro_final = macro_final.set_index('date').sort_index()
        
        for date in sorted(forecaster.two_by_two_variable_inclusion.keys()):
            if date not in macro_final.index:
                continue
            
            macro_row = macro_final.reindex([date])
            if macro_row.isna().any().any():
                continue
            
            regime_assignment = forecaster.regime_def.classify_dataframe(
                macro_row.reset_index(),
                growth_col='growth_factor',
                inflation_col='inflation_factor'
            )[0]
            
            # Get variables included in this regime
            included_vars = forecaster.two_by_two_variable_inclusion[date].get(regime_assignment, [])
            
            # Create binary inclusion dict: date -> Dict[regime] -> List[var]
            # For 2x2, we only care about the active regime, but we'll store it as regime 0 for plotting
            two_by_two_binary_inclusion[date] = {regime_assignment: included_vars}
        
        # Create a modified plotting function call that handles 2x2 correctly
        # We need to create a matrix showing which variables are included at each date
        dates = sorted(two_by_two_binary_inclusion.keys())
        inclusion_matrix = np.zeros((len(dates), len(macro_vars)))
        
        for i, date in enumerate(dates):
            regime_inclusions = two_by_two_binary_inclusion[date]
            # Get the active regime (should only be one)
            active_regime = list(regime_inclusions.keys())[0]
            included_vars = regime_inclusions[active_regime]
            
            for j, var in enumerate(macro_vars):
                inclusion_matrix[i, j] = 1.0 if var in included_vars else 0.0
        
        # Create plot
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Use binary colormap (light blue = 0, dark blue = 1)
        colors = ['#E3F2FD', '#1976D2']  # Light blue, dark blue
        cmap = mcolors.ListedColormap(colors)
        im = ax.imshow(inclusion_matrix.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], label='Included')
        cbar.ax.set_yticklabels(['Not Included', 'Included'])
        
        # Set labels
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Macro Variable', fontsize=12, fontweight='bold')
        ax.set_title('2x2 LASSO Variable Inclusion Over Time', fontsize=14, fontweight='bold')
        
        # Set y-axis labels to variable names
        ax.set_yticks(range(len(macro_vars)))
        ax.set_yticklabels(macro_vars, fontsize=9)
        
        # Set x-axis labels to years
        year_positions = []
        year_labels = []
        for i, date in enumerate(dates):
            year = date.year
            if year not in year_labels:
                year_positions.append(i)
                year_labels.append(year)
        
        # Show every 2-3 years to avoid crowding
        if len(year_positions) > 10:
            step = max(1, len(year_positions) // 10)
            year_positions = year_positions[::step]
            year_labels = year_labels[::step]
        
        ax.set_xticks(year_positions)
        ax.set_xticklabels(year_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "2x2_lasso_variable_inclusion.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved 2x2 variable inclusion plot")
        plt.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"\nGenerated files:")
    print(f"  - Strategy returns CSVs")
    print(f"  - Performance summary CSV")
    print(f"  - Cumulative returns plot")
    print(f"  - Variable inclusion plots (HMM and 2x2)")


if __name__ == "__main__":
    main()

