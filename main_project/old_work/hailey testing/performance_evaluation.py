"""
Performance Evaluation Module
Evaluates and visualizes strategy performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from pathlib import Path


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate performance metrics for a return series.
    
    Parameters:
    -----------
    returns : pd.Series
        Monthly return series
    
    Returns:
    --------
    Dict with performance metrics
    """
    monthly_return = returns.mean()
    annual_return = monthly_return * 12
    monthly_vol = returns.std()
    annual_vol = monthly_vol * np.sqrt(12)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(12)
    
    return {
        'monthly_return': monthly_return,
        'annual_return': annual_return,
        'monthly_volatility': monthly_vol,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe
    }


def plot_performance_comparison(
    df_strategy: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Plot comprehensive performance comparison.
    
    Parameters:
    -----------
    df_strategy : pd.DataFrame
        Strategy DataFrame with returns
    output_path : Path, optional
        Path to save the plot
    """
    # Calculate metrics
    regime_metrics = calculate_performance_metrics(df_strategy['regime_portfolio_return'])
    benchmark_metrics = calculate_performance_metrics(df_strategy['benchmark_return'])
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Monthly Returns Over Time
    ax1 = axes[0, 0]
    ax1.plot(
        df_strategy['date'],
        df_strategy['regime_portfolio_return'],
        label='Regime-Based Strategy',
        linewidth=2,
        alpha=0.7
    )
    ax1.plot(
        df_strategy['date'],
        df_strategy['benchmark_return'],
        label='50/50 Benchmark',
        linewidth=2,
        linestyle='--',
        alpha=0.7
    )
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Monthly Return')
    ax1.set_title('Monthly Returns: Regime-Based vs 50/50 Benchmark')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance Metrics Comparison
    ax2 = axes[0, 1]
    metrics = ['Annual Return', 'Annual Vol', 'Sharpe Ratio']
    regime_values = [
        regime_metrics['annual_return'],
        regime_metrics['annual_volatility'],
        regime_metrics['sharpe_ratio']
    ]
    benchmark_values = [
        benchmark_metrics['annual_return'],
        benchmark_metrics['annual_volatility'],
        benchmark_metrics['sharpe_ratio']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, regime_values, width, label='Regime-Based', alpha=0.8)
    ax2.bar(x + width/2, benchmark_values, width, label='50/50 Benchmark', alpha=0.8)
    ax2.set_ylabel('Value')
    ax2.set_title('Performance Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Monthly Returns Distribution
    ax3 = axes[1, 0]
    ax3.hist(
        df_strategy['regime_portfolio_return'],
        bins=30,
        alpha=0.7,
        label='Regime-Based',
        edgecolor='black'
    )
    ax3.hist(
        df_strategy['benchmark_return'],
        bins=30,
        alpha=0.7,
        label='50/50 Benchmark',
        edgecolor='black'
    )
    ax3.set_xlabel('Monthly Return')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Monthly Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Return Difference Over Time
    ax4 = axes[1, 1]
    df_strategy['return_difference'] = (
        df_strategy['regime_portfolio_return'] - df_strategy['benchmark_return']
    )
    ax4.plot(
        df_strategy['date'],
        df_strategy['return_difference'],
        linewidth=2,
        color='purple',
        alpha=0.7
    )
    ax4.fill_between(
        df_strategy['date'],
        df_strategy['return_difference'],
        0,
        alpha=0.3,
        color='purple'
    )
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Return Difference')
    ax4.set_title('Monthly Outperformance: Regime-Based vs 50/50 Benchmark')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print metrics
    print(f"\n{'='*70}")
    print(f"REGIME-BASED STRATEGY PERFORMANCE")
    print(f"{'='*70}")
    for key, value in regime_metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.6f}")
    
    print(f"\n{'='*70}")
    print(f"50/50 BENCHMARK PERFORMANCE")
    print(f"{'='*70}")
    for key, value in benchmark_metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.6f}")


def plot_cumulative_returns(
    df_strategy: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Plot cumulative returns.
    
    Parameters:
    -----------
    df_strategy : pd.DataFrame
        Strategy DataFrame with returns
    output_path : Path, optional
        Path to save the plot
    """
    # Calculate cumulative returns
    df_strategy['regime_cumulative_return'] = (
        (1 + df_strategy['regime_portfolio_return']).cumprod()
    )
    df_strategy['benchmark_cumulative_return'] = (
        (1 + df_strategy['benchmark_return']).cumprod()
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot cumulative returns
    ax.plot(
        df_strategy['date'],
        df_strategy['regime_cumulative_return'],
        label='Regime-Based Strategy',
        linewidth=2.5,
        alpha=0.8,
        color='#1f77b4'
    )
    ax.plot(
        df_strategy['date'],
        df_strategy['benchmark_cumulative_return'],
        label='50/50 Benchmark',
        linewidth=2.5,
        linestyle='--',
        alpha=0.8,
        color='#ff7f0e'
    )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Growth of $1', fontsize=12)
    ax.set_title(
        'Cumulative Returns: Regime-Based Strategy vs 50/50 Benchmark',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print final values
    regime_final = df_strategy['regime_cumulative_return'].iloc[-1]
    benchmark_final = df_strategy['benchmark_cumulative_return'].iloc[-1]
    
    print(f"\n{'='*70}")
    print(f"CUMULATIVE RETURN ANALYSIS")
    print(f"{'='*70}")
    print(f"\nRegime-Based Strategy:")
    print(f"  Starting Value: $1.00")
    print(f"  Ending Value: ${regime_final:.2f}")
    print(f"  Total Return: {(regime_final - 1) * 100:.2f}%")
    
    print(f"\n50/50 Benchmark:")
    print(f"  Starting Value: $1.00")
    print(f"  Ending Value: ${benchmark_final:.2f}")
    print(f"  Total Return: {(benchmark_final - 1) * 100:.2f}%")
    
    print(f"\nOutperformance:")
    print(f"  Absolute Difference: ${regime_final - benchmark_final:.2f}")
    print(f"  Relative Outperformance: {((regime_final / benchmark_final) - 1) * 100:.2f}%")

