"""
Plotting functions for HMM-based strategy visualization.

Functions:
- plot_cumulative_returns_all_strategies: Plot cumulative returns for all strategies
- plot_performance_comparison: Compare all strategies with bar charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10


def plot_cumulative_returns_all_strategies(
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot cumulative returns for all strategies on a single plot.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts with 'returns' key
    benchmark_returns : pd.Series, optional
        Benchmark returns (50/50) for comparison (not used, kept for compatibility)
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define colors for strategies
    strategy_colors = {
        'hmm_forecast_based': '#2E86AB',      # Blue
        'hmm_actual_based': '#A23B72',        # Purple
        'fixed_50_50_benchmark': 'darkgray'   # Gray
    }
    
    # Plot all strategies (including fixed_50_50_benchmark which serves as the benchmark)
    for name, results in strategies.items():
        if 'returns' not in results or results['returns'].empty:
            continue
        
        returns = results['returns']
        cum_returns = (1 + returns).cumprod()
        
        color = strategy_colors.get(name, '#6C757D')  # Default gray
        linewidth = 2.5
        linestyle = '--' if 'benchmark' in name else '-'
        
        # Format label
        label = name.replace('_', ' ').title()
        if 'Fixed 50 50 Benchmark' in label:
            label = 'Benchmark (50/50)'
        
        ax.plot(cum_returns.index, cum_returns.values,
               label=label, 
               linewidth=linewidth, alpha=0.9, color=color, linestyle=linestyle)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Returns: All HMM-Based Strategies', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "cumulative_returns_all_strategies.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative returns plot to {output_dir / filename}")
    
    plt.close()


def plot_performance_comparison(
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create a single comparison plot with all strategies.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts
    benchmark_returns : pd.Series, optional
        Benchmark returns (50/50) for comparison
    output_dir : Path, optional
        Directory to save plots
    """
    from performance import compute_performance_metrics
    
    # Collect all strategies
    strategy_list = []
    
    for name, results in strategies.items():
        if 'metrics' not in results:
            continue
        strategy_list.append((name, results['metrics']))
    
    # Get all strategy dates for benchmark alignment
    all_strategy_dates = set()
    for name, results in strategies.items():
        if 'returns' in results and not results['returns'].empty:
            all_strategy_dates.update(results['returns'].index)
    
    # Compute benchmark metrics
    benchmark_metrics = None
    if benchmark_returns is not None and not benchmark_returns.empty and all_strategy_dates:
        common_dates = sorted(all_strategy_dates)
        benchmark_aligned = benchmark_returns.reindex(common_dates).dropna()
        if len(benchmark_aligned) > 0:
            benchmark_metrics = compute_performance_metrics(benchmark_aligned)
    
    # Build combined metrics dataframe
    all_metrics = []
    all_labels = []
    all_positions = []
    
    # Position counter
    position = 0
    
    # Add benchmark first (position 0)
    if benchmark_metrics is not None:
        all_metrics.append({
            'sharpe_ratio': benchmark_metrics['sharpe_ratio'],
            'annualized_volatility': benchmark_metrics['annualized_volatility'],
            'max_drawdown': benchmark_metrics['max_drawdown'],
            'annualized_return': benchmark_metrics['annualized_return']
        })
        all_labels.append('Benchmark (50/50)')
        all_positions.append(position)
        position += 1
    
    # Add all strategies
    strategy_list = sorted(strategy_list, key=lambda x: x[0])
    for name, metrics in strategy_list:
        all_metrics.append({
            'sharpe_ratio': metrics['sharpe_ratio'],
            'annualized_volatility': metrics['annualized_volatility'],
            'max_drawdown': metrics['max_drawdown'],
            'annualized_return': metrics['annualized_return']
        })
        all_labels.append(name)
        all_positions.append(position)
        position += 1
    
    if not all_metrics:
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df['strategy'] = all_labels
    metrics_df['position'] = all_positions
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Color scheme: benchmark in darkgray, HMM strategies in blue shades
    colors = ['darkgray' if 'Benchmark' in label else '#3A5F8F' for label in all_labels]
    
    # Sharpe ratio
    ax = axes[0, 0]
    bars = ax.bar(metrics_df['position'], metrics_df['sharpe_ratio'], color=colors, width=0.8)
    # Highlight benchmark
    if benchmark_metrics is not None:
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Volatility
    ax = axes[0, 1]
    bars = ax.bar(metrics_df['position'], metrics_df['annualized_volatility'], color=colors, width=0.8)
    if benchmark_metrics is not None:
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Annualized Volatility Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatility', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Max Drawdown
    ax = axes[1, 0]
    bars = ax.bar(metrics_df['position'], metrics_df['max_drawdown'], color=colors, width=0.8)
    if benchmark_metrics is not None:
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Max Drawdown', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Annualized Return
    ax = axes[1, 1]
    bars = ax.bar(metrics_df['position'], metrics_df['annualized_return'], color=colors, width=0.8)
    if benchmark_metrics is not None:
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Annualized Return Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance Comparison - HMM-Based Strategies', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "performance_comparison_all_strategies.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {output_dir / filename}")
    
    plt.close()
    
    return metrics_df


