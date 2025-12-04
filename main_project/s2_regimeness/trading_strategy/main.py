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
from plotting import plot_performance_comparison, plot_cumulative_returns_all_strategies
from trading import run_trading_strategy
from hmm_forecasts import generate_all_hmm_strategies

START_DATE = pd.Timestamp("2000-01-31")
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
    
    # Generate HMM-based strategies
    print("\nGenerating HMM-based strategies...")
    hmm_strategies = generate_all_hmm_strategies(
        all_macro_df,
        forecast_df,
        base_dir=Path(__file__).parent.parent
    )
    
    # Filter to START_DATE
    scenario_forecasts = {}
    for name, series in hmm_strategies.items():
        if len(series) > 0:
            series = series[series.index >= START_DATE]
            scenario_forecasts[name] = series

    benchmark_returns = 0.5 * equity_ret + 0.5 * bond_ret
    
    # Add fixed 50/50 benchmark forecast (zero forecast, will result in 0.5 weight)
    benchmark_forecast = pd.Series(0.0, index=common_index[common_index >= START_DATE])
    scenario_forecasts["fixed_50_50_benchmark"] = benchmark_forecast
    
    strategies = {}
    metrics_rows = []

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

        # Save individual time series
        strategy_output = RESULTS_DIR / f"{name}_returns.csv"
        pd.DataFrame({
            "return": strategy.returns,
            "weight": strategy.weights,
            "forecast": strategy.forecast.reindex(strategy.returns.index)
        }).to_csv(strategy_output)

    performance_df = pd.DataFrame(metrics_rows).sort_values("sharpe_ratio", ascending=False)
    performance_df.to_csv(RESULTS_DIR / "strategy_performance_summary.csv", index=False)

    # Cumulative returns plot
    plot_cumulative_returns_all_strategies(strategies, benchmark_returns=benchmark_returns, output_dir=RESULTS_DIR)

    # Performance comparison plot
    plot_performance_comparison(strategies, benchmark_returns=benchmark_returns, output_dir=RESULTS_DIR)

    print("\nEvaluation complete.")
    print(performance_df.to_string(index=False))


if __name__ == "__main__":
    main()
