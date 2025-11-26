"""
Plotting functions for strategy visualization.

Functions:
- plot_cumulative_returns: Compare strategy vs benchmark
- plot_equity_weights: Show weight path over time
- plot_regime_transitions: Overlay regime changes with weights
- plot_performance_comparison: Compare all strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10


def plot_cumulative_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str = "Strategy",
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot cumulative returns comparison.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns (50/50)
    strategy_name : str
        Name of strategy
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Compute cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    # Plot
    ax.plot(strategy_cum.index, strategy_cum.values, 
           label=strategy_name, linewidth=2.5, alpha=0.9)
    ax.plot(benchmark_cum.index, benchmark_cum.values, 
           label='Benchmark (50/50)', linewidth=2, alpha=0.7, linestyle='--', color='gray')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title(f'{strategy_name} vs Benchmark: Cumulative Returns', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cumulative_returns_{strategy_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative returns plot to {output_dir / filename}")
    
    plt.close()


def plot_equity_weights(
    weights: pd.Series,
    strategy_name: str = "Strategy",
    regime_probs: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot equity weights over time with optional regime overlay.
    
    Parameters:
    -----------
    weights : pd.Series
        Equity weights over time
    strategy_name : str
        Name of strategy
    regime_probs : pd.DataFrame, optional
        Regime probabilities for overlay
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot weights
    ax.plot(weights.index, weights.values, 
           label='Equity Weight', linewidth=2, alpha=0.8, color='steelblue')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Benchmark (50%)')
    
    # Add regime shading if provided
    if regime_probs is not None:
        # Get dominant regime
        prob_cols = [col for col in regime_probs.columns if col.startswith('prob_R')]
        if len(prob_cols) > 0:
            dominant_regime = regime_probs[prob_cols].idxmax(axis=1)
            
            # Create regime colors
            regime_colors = {col: plt.cm.Set3(i) for i, col in enumerate(prob_cols)}
            
            # Shade regions by regime
            prev_regime = None
            start_idx = None
            for idx in regime_probs.index:
                current_regime = dominant_regime.loc[idx]
                if current_regime != prev_regime:
                    if prev_regime is not None and start_idx is not None:
                        ax.axvspan(start_idx, idx, alpha=0.1, 
                                 color=regime_colors.get(prev_regime, 'gray'))
                    start_idx = idx
                    prev_regime = current_regime
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity Weight', fontsize=12)
    ax.set_title(f'{strategy_name}: Equity Weight Over Time', 
                fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"equity_weights_{strategy_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved equity weights plot to {output_dir / filename}")
    
    plt.close()


def plot_performance_comparison(
    strategies: Dict[str, Dict],
    output_dir: Optional[Path] = None
) -> None:
    """
    Create comparison table and plots for all strategies.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts
    output_dir : Path, optional
        Directory to save plots
    """
    # Extract metrics
    metrics_list = []
    for name, results in strategies.items():
        metrics = results['metrics'].copy()
        metrics['strategy'] = name
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sharpe ratio
    ax = axes[0, 0]
    metrics_df.plot(x='strategy', y='sharpe_ratio', kind='bar', ax=ax, legend=False)
    ax.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Volatility
    ax = axes[0, 1]
    metrics_df.plot(x='strategy', y='annualized_volatility', kind='bar', ax=ax, legend=False, color='coral')
    ax.set_title('Annualized Volatility Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatility', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Max Drawdown
    ax = axes[1, 0]
    metrics_df.plot(x='strategy', y='max_drawdown', kind='bar', ax=ax, legend=False, color='lightcoral')
    ax.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Max Drawdown', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Annualized Return
    ax = axes[1, 1]
    metrics_df.plot(x='strategy', y='annualized_return', kind='bar', ax=ax, legend=False, color='lightgreen')
    ax.set_title('Annualized Return Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "performance_comparison_all_strategies.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {output_dir / filename}")
    
    plt.close()
    
    return metrics_df

