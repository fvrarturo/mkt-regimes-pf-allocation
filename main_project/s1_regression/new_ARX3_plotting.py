"""
Plotting functions for ARX(3) model results.

This module handles visualization of:
- Actual vs predicted ERP
- Cumulative portfolio performance
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_actual_vs_predicted(y_test, y_hat_test, title="ARX(3) Model (Test Set)", 
                             model_name="ARX(3)", color="blue"):
    """
    Plot actual vs predicted ERP_next.
    
    Parameters
    ----------
    y_test : pd.Series
        Actual ERP values
    y_hat_test : array-like
        Predicted ERP values
    title : str
        Plot title
    model_name : str
        Model name for legend
    color : str
        Color for predicted line
    """
    idx_test_ts = y_test.index.to_timestamp()
    
    plt.figure(figsize=(12, 5))
    plt.plot(idx_test_ts, y_test, label="Actual erp_next", color="black", linewidth=2)
    plt.plot(idx_test_ts, y_hat_test, label=f"Predicted erp_next ({model_name})", 
             color=color, linewidth=2)
    
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Monthly ERP – {title}")
    plt.ylabel("Next-month ERP")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(ret_A, ret_B, benchmarks, title="ARX(3) Model (Test Period)"):
    """
    Plot cumulative portfolio returns.
    
    Parameters
    ----------
    ret_A : pd.Series
        Strategy A returns
    ret_B : pd.Series
        Strategy B returns
    benchmarks : dict
        Dictionary with benchmark returns
    title : str
        Plot title
    """
    idx_ts = ret_A.index.to_timestamp()
    
    cum_A = (1 + ret_A).cumprod()
    cum_B = (1 + ret_B).cumprod()
    cum_bench_avg = None
    if benchmarks.get("bench_avg") is not None:
        cum_bench_avg = (1 + benchmarks["bench_avg"]).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(idx_ts, cum_A, label="Strategy A (bands)", linewidth=2)
    plt.plot(idx_ts, cum_B, label="Strategy B (sign)", linewidth=2)
    if cum_bench_avg is not None:
        plt.plot(idx_ts, cum_bench_avg, label="Benchmark (Avg Dynamic Mix)", linewidth=2)
    
    plt.title(f"Cumulative Growth of $1 – {title}")
    plt.ylabel("Cumulative Value")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

