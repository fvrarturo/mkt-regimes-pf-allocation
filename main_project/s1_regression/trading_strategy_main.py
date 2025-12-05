"""
Main script for trading strategy with full sample.

This script orchestrates the full pipeline:
1. Data loading
2. ERP analysis
3. Strategy weight calculation
4. Performance evaluation
5. Visualization
"""

import matplotlib.pyplot as plt
import pandas as pd

from trading_strategy_data_loader import load_market_data, build_monthly_returns
from trading_strategy_analysis import print_erp_statistics, calculate_z_scores
from trading_strategy_weights import compute_strategy_weights
from trading_strategy_performance import (
    evaluate_stock_cash_strategy,
    evaluate_stock_bond_strategy
)

plt.rcParams["figure.dpi"] = 120


def main():
    """Main execution function."""
    print("=" * 80)
    print("Trading Strategy - Full Sample")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1/5] Loading market data...")
    sp500, tbill = load_market_data(start="1990-01-02", end=None)
    print("S&P span:", sp500.index.min(), "to", sp500.index.max())
    print("T-bill span:", tbill.index.min(), "to", tbill.index.max())
    
    # Step 2: Build monthly returns
    print("\n[2/5] Building monthly returns...")
    data = build_monthly_returns(sp500, tbill)
    print("\nFirst rows of monthly ERP data:")
    print(data.head())
    
    # Step 3: Analyze ERP
    print("\n[3/5] Analyzing ERP statistics...")
    print_erp_statistics(data["erp"])
    
    # Step 4: Calculate z-scores and weights
    print("\n[4/5] Calculating strategy weights...")
    z_scores = calculate_z_scores(data["erp"])
    weights_df = compute_strategy_weights(z_scores)
    
    # Combine weights with returns
    strategy = weights_df.join(data[["sp500_ret", "rf_ret"]], how="inner")
    
    # Step 5: Evaluate strategies
    print("\n[5/5] Evaluating strategies...")
    
    # Stock-Cash Strategy
    sc_results = evaluate_stock_cash_strategy(strategy, cash_ret=0.0)
    
    print("\n===== Geometric Annualized Returns – STOCK–CASH =====")
    print("Dynamic ERP strategy           : {:.4%}".format(sc_results["g_dyn_annual"]))
    print("Benchmark (avg risky/safe mix) : {:.4%}".format(sc_results["g_bench_annual"]))
    
    print("\n===== Annualized Sharpe Ratios – STOCK–CASH (vs 3M T-Bill) =====")
    print("Dynamic ERP strategy           : {:.3f}".format(sc_results["sharpe_dyn"]))
    print("Benchmark (avg risky/safe mix) : {:.3f}".format(sc_results["sharpe_bench"]))
    
    # Stock-Bond Strategy
    sb_results = evaluate_stock_bond_strategy(data, weights_df["w_stock"])
    
    print("\n===== Geometric Annualized Returns – STOCK–BOND =====")
    print("Dynamic stock–bond strategy            : {:.4%}".format(sb_results["g_dyn_annual"]))
    print("Benchmark stock–bond (avg weights)     : {:.4%}".format(sb_results["g_bench_annual"]))
    
    print("\n===== Annualized Sharpe Ratios – STOCK–BOND (vs 3M T-Bill) =====")
    print("Dynamic stock–bond strategy            : {:.3f}".format(sb_results["sharpe_dyn"]))
    print("Benchmark stock–bond (avg weights)     : {:.3f}".format(sb_results["sharpe_bench"]))
    
    # Step 6: Visualizations
    print("\nGenerating plots...")
    
    # Plot ERP time series
    data_plot = data.copy()
    data_plot.index = data_plot.index.to_timestamp("M")
    
    plt.figure(figsize=(12, 6))
    plt.plot(data_plot.index, data_plot["erp"], label="ERP")
    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.title("Monthly Equity Risk Premium (S&P 500 – 3M T-Bill)")
    plt.ylabel("Monthly ERP")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()
    
    # Plot ERP distribution
    erp = data["erp"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(erp, bins=40, edgecolor="black")
    plt.title("Distribution of Monthly Equity Risk Premium")
    plt.xlabel("Monthly ERP (S&P – 3M T-bill)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # Plot cumulative performance - Stock-Cash
    if isinstance(strategy.index, pd.PeriodIndex):
        idx_plot = strategy.index.to_timestamp("M")
    else:
        idx_plot = strategy.index
    
    cum_dyn = (1 + sc_results["ret_dyn"]).cumprod()
    cum_bench = (1 + sc_results["ret_bench"]).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(idx_plot, cum_dyn, label="Dynamic ERP strategy")
    plt.plot(idx_plot, cum_bench, label="Benchmark (avg risky/safe mix)")
    plt.title("Cumulative Growth of $1 – Stock–Cash Strategy")
    plt.ylabel("Cumulative Value")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot cumulative performance - Stock-Bond
    if isinstance(data.index, pd.PeriodIndex):
        idx_plot_sb = data.index.to_timestamp("M")
    else:
        idx_plot_sb = data.index
    
    cum_dyn_sb = (1 + sb_results["ret_dyn"]).cumprod()
    cum_bench_sb = (1 + sb_results["ret_bench"]).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(idx_plot_sb, cum_dyn_sb, label="Dynamic stock–bond")
    plt.plot(idx_plot_sb, cum_bench_sb, label="Benchmark (avg risky/safe mix)")
    plt.title("Cumulative Growth of $1 – Stock–Bond Strategy")
    plt.ylabel("Cumulative Value")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

