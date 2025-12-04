"""
Unified trading strategy evaluation (Step 3).

Workflow:
1. Load equity/bond returns and macro data.
2. Generate HMM-based ERP forecasts using:
   - Forecast-based: Uses forecasts at T to determine regime mix at T+1
   - Actual-based: Uses actual values at T to determine regime mix
   - Fixed 50/50 benchmark
3. Convert forecasts into portfolio weights (SP500 vs T-Bills) and compute returns.
4. Save performance summary and plots.
"""

from pathlib import Path

import pandas as pd

from data_loader import load_market_data, load_all_macro_variables
from performance import compute_hit_rate, compute_performance_metrics, compute_turnover
from plotting import plot_performance_comparison, plot_cumulative_returns_all_strategies, plot_weights_over_time
from trading import run_trading_strategy
from hmm_forecasts import generate_all_hmm_strategies
from two_by_two_forecasts import generate_all_2x2_strategies

START_DATE = pd.Timestamp("2002-03-31")
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    equity_ret, bond_ret, erp = load_market_data()
    
    # Load all macro variables for HMM strategies
    print("\nLoading all macro variables for HMM strategies...")
    all_macro_df = load_all_macro_variables()

    # Align to common date index
    common_index = equity_ret.index.intersection(all_macro_df.index)
    equity_ret = equity_ret.reindex(common_index)
    bond_ret = bond_ret.reindex(common_index)
    erp = erp.reindex(common_index)
    all_macro_df = all_macro_df.reindex(common_index)

    # Load forecast data for HMM strategies
    print("\nLoading macro forecasts for HMM strategies...")
    forecast_path = Path(__file__).parent / "inputs" / "macro_forecasts.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast file not found at {forecast_path}")
    
    forecast_df = pd.read_csv(forecast_path, parse_dates=["date"])
    
    # Generate HMM-based strategies for all combinations with K=4
    print("\nGenerating HMM-based strategies for all combinations (K=4)...")
    hmm_strategies = generate_all_hmm_strategies(
        all_macro_df,
        forecast_df,
        base_dir=Path(__file__).parent.parent,
        k=4
    )
    
    # Generate 2x2 regime-based strategies
    print("\nGenerating 2x2 regime-based strategies...")
    two_by_two_strategies = generate_all_2x2_strategies(
        all_macro_df,
        forecast_df,
        base_dir=Path(__file__).parent.parent
    )
    
    # Filter to START_DATE and evaluate all HMM strategies
    scenario_forecasts = {}
    hmm_evaluation_results = []
    
    for name, series in hmm_strategies.items():
        if len(series) > 0:
            series = series[series.index >= START_DATE]
            if len(series) > 0:
                scenario_forecasts[name] = series
                
                # Evaluate this strategy to get returns
                strategy = run_trading_strategy(
                    name=name,
                    forecasts=series,
                    equity_returns=equity_ret,
                    bond_returns=bond_ret,
                )
                
                metrics = compute_performance_metrics(strategy.returns)
                hmm_evaluation_results.append({
                    "strategy": name,
                    "annualized_return": metrics["annualized_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "annualized_volatility": metrics["annualized_volatility"],
                    "max_drawdown": metrics["max_drawdown"],
                    "total_return": metrics["total_return"]
                })
    
    # Rank HMM strategies by annualized return
    hmm_rankings = pd.DataFrame(hmm_evaluation_results).sort_values("annualized_return", ascending=False)
    hmm_rankings["rank"] = range(1, len(hmm_rankings) + 1)
    hmm_rankings = hmm_rankings[["rank", "strategy", "annualized_return", "sharpe_ratio", "annualized_volatility", "max_drawdown", "total_return"]]
    
    # Save rankings to CSV
    hmm_rankings.to_csv(RESULTS_DIR / "hmm_strategy_rankings.csv", index=False)
    print("\nHMM Strategy Rankings (by Annualized Return):")
    print(hmm_rankings.to_string(index=False))
    
    # Select only the best HMM strategy
    best_hmm_strategy = hmm_rankings.iloc[0]["strategy"]
    print(f"\nBest HMM strategy: {best_hmm_strategy}")
    
    # Combine best HMM strategy with 2x2 strategy
    all_strategies = {}
    if best_hmm_strategy in scenario_forecasts:
        all_strategies[best_hmm_strategy] = scenario_forecasts[best_hmm_strategy]
    if "2x2_actual_based" in two_by_two_strategies:
        all_strategies["2x2_actual_based"] = two_by_two_strategies["2x2_actual_based"]
    
    # Filter to START_DATE
    scenario_forecasts = {}
    for name, series in all_strategies.items():
        if len(series) > 0:
            series = series[series.index >= START_DATE]
            scenario_forecasts[name] = series

    benchmark_returns = 0.5 * equity_ret + 0.5 * bond_ret
    
    strategies = {}
    metrics_rows = []
    
    # Store average weights for fixed portfolio benchmarks
    avg_weights = {}

    for name, forecast in scenario_forecasts.items():
        strategy = run_trading_strategy(
            name=name,
            forecasts=forecast,
            equity_returns=equity_ret,
            bond_returns=bond_ret,
        )
        strategies[name] = {
            "returns": strategy.returns,
            "weights": strategy.weights,
            "forecast": strategy.forecast,
        }
        
        # Compute average weight for fixed portfolio benchmarks
        if not strategy.weights.empty:
            avg_weight = strategy.weights.mean()
            avg_weights[name] = avg_weight

        metrics = compute_performance_metrics(strategy.returns)
        hit_stats = compute_hit_rate(strategy.forecast, erp.shift(-1))
        metrics.update({
            "strategy": name,
            "hit_rate": hit_stats["hit_rate"],
            "n_predictions": hit_stats["n_observations"],
            "turnover": compute_turnover(strategy.weights),
        })
        metrics_rows.append(metrics)
        strategies[name]["metrics"] = metrics
        strategies[name]["avg_weight"] = avg_weights.get(name, 0.5)

        # Save individual time series
        strategy_output = RESULTS_DIR / f"{name}_returns.csv"
        pd.DataFrame({
            "return": strategy.returns,
            "weight": strategy.weights,
            "forecast": strategy.forecast.reindex(strategy.returns.index)
        }).to_csv(strategy_output)
    
    # Create fixed portfolio benchmarks for best HMM and 2x2 strategies
    # Compute fixed portfolio returns based on average weights
    for strategy_name in [best_hmm_strategy, "2x2_actual_based"]:
        if strategy_name in avg_weights:
            avg_weight = avg_weights[strategy_name]
            fixed_name = f"{strategy_name}_fixed_portfolio"
            
            # Get common dates
            if strategy_name in strategies and not strategies[strategy_name]["returns"].empty:
                common_dates = strategies[strategy_name]["returns"].index
                fixed_returns = avg_weight * equity_ret.reindex(common_dates) + (1 - avg_weight) * bond_ret.reindex(common_dates)
                
                strategies[fixed_name] = {
                    "returns": fixed_returns,
                    "weights": pd.Series(avg_weight, index=common_dates),
                    "forecast": pd.Series(0.0, index=common_dates),  # Not used for fixed portfolio
                    "avg_weight": avg_weight,
                    "metrics": compute_performance_metrics(fixed_returns)
                }

    performance_df = pd.DataFrame(metrics_rows).sort_values("sharpe_ratio", ascending=False)
    performance_df.to_csv(RESULTS_DIR / "strategy_performance_summary.csv", index=False)

    # Cumulative returns plot (includes fixed portfolios)
    plot_cumulative_returns_all_strategies(
        strategies, 
        benchmark_returns=benchmark_returns, 
        output_dir=RESULTS_DIR,
        show_fixed_portfolios=True
    )

    # Performance comparison plot
    plot_performance_comparison(strategies, benchmark_returns=benchmark_returns, output_dir=RESULTS_DIR)
    
    # Plot weights over time (with regime probabilities/positions)
    plot_weights_over_time(strategies, all_macro_df=all_macro_df, base_dir=Path(__file__).parent.parent, output_dir=RESULTS_DIR)

    print("\nEvaluation complete.")
    print(performance_df.to_string(index=False))


if __name__ == "__main__":
    main()
