"""
Main script for strategy evaluation (Section 6).

Implements:
- 6.1 Regime-based strategy
- 6.2 Extremeness-based strategy
- 6.3 Forecast-based strategy
- 6.4 Ensemble strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_loader import (
    load_market_data,
    load_regime_probabilities,
    load_extremeness_scores,
    load_forecasts
)
from strategy_regime import regime_strategy
from strategy_extremeness import (
    extremeness_binary_strategy,
    extremeness_regime_combined_strategy
)
from strategy_forecast import forecast_strategy
from strategy_ensemble import ensemble_strategy
from plotting import (
    plot_cumulative_returns,
    plot_equity_weights,
    plot_performance_comparison
)
from performance import compute_performance_metrics


def compute_benchmark_returns(
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    weight: float = 0.5
) -> pd.Series:
    """
    Compute benchmark returns (50/50 static allocation).
    
    Parameters:
    -----------
    equity_returns : pd.Series
        Equity returns
    bond_returns : pd.Series
        Bond returns
    weight : float
        Equity weight (default: 0.5)
    
    Returns:
    --------
    pd.Series
        Benchmark returns
    """
    aligned = pd.DataFrame({
        'equity_return': equity_returns,
        'bond_return': bond_returns
    }).dropna()
    
    benchmark_returns = (
        weight * aligned['equity_return'] +
        (1 - weight) * aligned['bond_return']
    )
    
    return benchmark_returns


def main():
    """Main execution function."""
    print("="*80)
    print("Strategy Evaluation: Regime, Extremeness, and Forecast-Based Strategies")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 0: Load market data
    print("\n" + "="*80)
    print("Step 0: Loading market data")
    print("="*80)
    equity_returns, bond_returns, erp = load_market_data(base_dir)
    print(f"Loaded market data:")
    print(f"  Equity returns: {len(equity_returns)} observations")
    print(f"  Bond returns: {len(bond_returns)} observations")
    print(f"  ERP: {len(erp)} observations")
    
    # Compute benchmark (50/50)
    benchmark_returns = compute_benchmark_returns(equity_returns, bond_returns, weight=0.5)
    benchmark_metrics = compute_performance_metrics(benchmark_returns)
    print(f"\nBenchmark (50/50) performance:")
    print(f"  Annualized return: {benchmark_metrics['annualized_return']:.4f}")
    print(f"  Sharpe ratio: {benchmark_metrics['sharpe_ratio']:.4f}")
    print(f"  Max drawdown: {benchmark_metrics['max_drawdown']:.4f}")
    
    # Load regime probabilities
    print("\n" + "="*80)
    print("Step 1: Loading regime probabilities")
    print("="*80)
    regime_probs = load_regime_probabilities(base_dir)
    if len(regime_probs) > 0:
        print(f"Loaded regime probabilities: {len(regime_probs)} observations")
        print(f"  Regime probability columns: {[col for col in regime_probs.columns if col.startswith('prob_R')]}")
    else:
        print("Warning: No regime probabilities found. Regime strategies will be skipped.")
    
    # Load extremeness scores
    print("\n" + "="*80)
    print("Step 2: Loading extremeness scores")
    print("="*80)
    extremeness = load_extremeness_scores(base_dir)
    if len(extremeness) > 0:
        print(f"Loaded extremeness scores: {len(extremeness)} observations")
        print(f"  Extremeness range: [{extremeness.min():.4f}, {extremeness.max():.4f}]")
    else:
        print("Warning: No extremeness scores found. Extremeness strategies will be skipped.")
    
    # Load forecasts
    print("\n" + "="*80)
    print("Step 3: Loading forecasts")
    print("="*80)
    forecasts = load_forecasts(base_dir)
    print(f"Forecast loading: {len(forecasts)} models available")
    # Note: Forecasts would need to be implemented to load actual forecast series
    
    # Store all strategy results
    all_strategies = {}
    
    # Step 4: Regime-based strategy
    if len(regime_probs) > 0:
        print("\n" + "="*80)
        print("Step 4: Implementing regime-based strategy")
        print("="*80)
        regime_results = regime_strategy(
            regime_probs,
            equity_returns,
            bond_returns
        )
        all_strategies['Regime-Based'] = regime_results
        print(f"\nRegime strategy performance:")
        print(f"  Sharpe ratio: {regime_results['metrics']['sharpe_ratio']:.4f}")
        print(f"  Annualized return: {regime_results['metrics']['annualized_return']:.4f}")
        print(f"  Max drawdown: {regime_results['metrics']['max_drawdown']:.4f}")
        
        # Plot
        plot_cumulative_returns(
            regime_results['returns'],
            benchmark_returns,
            strategy_name="Regime-Based",
            output_dir=output_dir
        )
        plot_equity_weights(
            regime_results['weights'],
            strategy_name="Regime-Based",
            regime_probs=regime_probs,
            output_dir=output_dir
        )
    
    # Step 5: Extremeness-based strategy
    if len(extremeness) > 0:
        print("\n" + "="*80)
        print("Step 5: Implementing extremeness-based strategy")
        print("="*80)
        
        # Binary extremeness strategy
        extremeness_binary_results = extremeness_binary_strategy(
            extremeness,
            equity_returns,
            bond_returns
        )
        all_strategies['Extremeness (Binary)'] = extremeness_binary_results
        print(f"\nExtremeness binary strategy performance:")
        print(f"  Sharpe ratio: {extremeness_binary_results['metrics']['sharpe_ratio']:.4f}")
        print(f"  Crash avoidance score: {extremeness_binary_results['metrics'].get('crash_avoidance_score', np.nan):.4f}")
        
        # Combined extremeness + regime strategy
        if len(regime_probs) > 0 and 'Regime-Based' in all_strategies:
            extremeness_combined_results = extremeness_regime_combined_strategy(
                extremeness,
                all_strategies['Regime-Based']['weights'],
                equity_returns,
                bond_returns
            )
            all_strategies['Extremeness + Regime'] = extremeness_combined_results
            print(f"\nExtremeness + Regime strategy performance:")
            print(f"  Sharpe ratio: {extremeness_combined_results['metrics']['sharpe_ratio']:.4f}")
        
        # Plot extremeness strategy
        plot_cumulative_returns(
            extremeness_binary_results['returns'],
            benchmark_returns,
            strategy_name="Extremeness (Binary)",
            output_dir=output_dir
        )
        plot_equity_weights(
            extremeness_binary_results['weights'],
            strategy_name="Extremeness (Binary)",
            output_dir=output_dir
        )
    
    # Step 6: Forecast-based strategies
    # Note: This requires forecast series to be loaded/available
    # For now, create placeholder structure
    print("\n" + "="*80)
    print("Step 6: Forecast-based strategies")
    print("="*80)
    print("Note: Forecast-based strategies require forecast series to be loaded.")
    print("This will be implemented once forecast data is available.")
    
    # Step 7: Ensemble strategy
    if len(regime_probs) > 0 and len(extremeness) > 0:
        print("\n" + "="*80)
        print("Step 7: Implementing ensemble strategy")
        print("="*80)
        # Note: Requires forecast ERP - placeholder for now
        print("Note: Ensemble strategy requires forecast ERP. Skipping for now.")
    
    # Step 8: Performance comparison
    print("\n" + "="*80)
    print("Step 8: Performance comparison")
    print("="*80)
    
    # Add benchmark to comparison
    all_strategies['Benchmark (50/50)'] = {
        'returns': benchmark_returns,
        'metrics': benchmark_metrics,
        'weights': pd.Series(0.5, index=benchmark_returns.index)
    }
    
    # Create comparison table
    metrics_df = plot_performance_comparison(all_strategies, output_dir=output_dir)
    metrics_df.to_csv(output_dir / "strategy_performance_comparison.csv", index=False)
    print(f"\nSaved performance comparison to {output_dir / 'strategy_performance_comparison.csv'}")
    
    # Step 9: Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - strategy_performance_comparison.csv")
    print("  - performance_comparison_all_strategies.png")
    print("  - cumulative_returns_*.png")
    print("  - equity_weights_*.png")


if __name__ == "__main__":
    main()

