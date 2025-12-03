"""
Unified trading strategy evaluation (Step 3).

Workflow:
1. Load equity/bond returns and macro/regime data.
2. Build ERP forecasts using:
   - Rolling linear regression (full-sample baseline)
   - Regime-conditioned expectation (HMM)
   - Extremeness-conditioned expectation (Isolation Forest + PCA distance)
3. Apply accuracy scenarios (40/60/80%) to macro-driven forecasts.
4. Convert forecasts into portfolio weights (SP500 vs T-Bills) and compute returns.
5. Save performance summary and optional plots.
"""

from pathlib import Path

import pandas as pd

from data_loader import load_market_data, load_macro_features
from forecasts import fit_conditional_forecaster, generate_macro_forecasts
from performance import compute_hit_rate, compute_performance_metrics, compute_turnover
from plotting import (
    plot_cumulative_returns,
    plot_performance_comparison,
    plot_cumulative_returns_by_accuracy,
    export_strategies_by_accuracy
)
from trading import run_trading_strategy

START_DATE = pd.Timestamp("2000-01-31")
RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    equity_ret, bond_ret, erp = load_market_data()
    macro_df = load_macro_features()

    # Align to common date index
    common_index = equity_ret.index.intersection(macro_df.index)
    equity_ret = equity_ret.reindex(common_index)
    bond_ret = bond_ret.reindex(common_index)
    erp = erp.reindex(common_index)
    macro_df = macro_df.reindex(common_index)

    print("\nFitting conditional regression models...")
    forecaster = fit_conditional_forecaster(macro_df, erp)

    accuracy_levels = [0.4, 0.6, 0.8, 1.0]
    scenario_forecasts = {}

    for acc in accuracy_levels:
        macro_hat = generate_macro_forecasts(macro_df, acc, seed=int(acc * 1000))
        forecast_dict = forecaster.forecast_all(macro_hat)
        for model_name, series in forecast_dict.items():
            series = series[series.index >= START_DATE]
            key = model_name if acc >= 0.999 else f"{model_name}_acc_{int(acc * 100)}"
            scenario_forecasts[key] = series

    benchmark_returns = 0.5 * equity_ret + 0.5 * bond_ret
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

    # Plot cumulative returns grouped by accuracy level
    plot_cumulative_returns_by_accuracy(
        strategies,
        benchmark_returns,
        output_dir=RESULTS_DIR
    )

    # Performance comparison plots grouped by accuracy level
    plot_performance_comparison(strategies, benchmark_returns=benchmark_returns, output_dir=RESULTS_DIR)

    # Export CSV files grouped by accuracy level
    export_strategies_by_accuracy(strategies, output_dir=RESULTS_DIR)

    print("\nEvaluation complete.")
    print(performance_df.to_string(index=False))


if __name__ == "__main__":
    main()
