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
import re
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


def plot_cumulative_returns_by_accuracy(
    strategies: Dict[str, Dict],
    benchmark_returns: pd.Series,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot cumulative returns grouped by accuracy level, showing all models together.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts with 'returns' key
    benchmark_returns : pd.Series
        Benchmark returns (50/50)
    output_dir : Path, optional
        Directory to save plots
    """
    # Group strategies by accuracy level
    accuracy_groups = {}
    
    for name, results in strategies.items():
        if 'returns' not in results or results['returns'].empty:
            continue
        
        # Extract accuracy level from strategy name
        # Pattern: _acc_40, _acc_60, _acc_80, or no _acc_ suffix (100%)
        match = re.search(r'_acc_(\d+)', name)
        if match:
            acc_level = int(match.group(1))
        else:
            # No _acc_ suffix means 100% accuracy
            acc_level = 100
        
        if acc_level not in accuracy_groups:
            accuracy_groups[acc_level] = {}
        accuracy_groups[acc_level][name] = results['returns']
    
    # Create a plot for each accuracy level
    for acc_level in sorted(accuracy_groups.keys()):
        strategies_at_level = accuracy_groups[acc_level]
        
        if not strategies_at_level:
            continue
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Define colors for different models
        model_colors = {
            'full_regression': '#1f77b4',  # blue
            'regime_hmm': '#ff7f0e',  # orange
            'extreme_isolation': '#2ca02c',  # green
            'extreme_pca': '#d62728',  # red
        }
        
        # Plot each strategy
        for name, returns in strategies_at_level.items():
            # Get base model name (without accuracy suffix)
            base_name = re.sub(r'_acc_\d+', '', name)
            
            # Compute cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Get color for this model
            color = None
            for model_key, model_color in model_colors.items():
                if base_name.startswith(model_key):
                    color = model_color
                    break
            
            if color is None:
                color = plt.cm.tab10(len(ax.lines) % 10)
            
            # Create label
            if acc_level == 100:
                label = base_name.replace('_', ' ').title()
            else:
                label = f"{base_name.replace('_', ' ').title()} ({acc_level}%)"
            
            ax.plot(cum_returns.index, cum_returns.values,
                   label=label, linewidth=2.5, alpha=0.9, color=color)
        
        # Plot benchmark
        # Align benchmark to common date range
        all_dates = set()
        for returns in strategies_at_level.values():
            all_dates.update(returns.index)
        common_dates = sorted(all_dates)
        benchmark_aligned = benchmark_returns.reindex(common_dates)
        benchmark_cum = (1 + benchmark_aligned).cumprod()
        
        ax.plot(benchmark_cum.index, benchmark_cum.values,
               label='Benchmark (50/50)', linewidth=2, alpha=0.7,
               linestyle='--', color='gray')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title(f'Cumulative Returns Comparison - {acc_level}% Accuracy',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"cumulative_returns_acc_{acc_level}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved cumulative returns plot (accuracy {acc_level}%) to {output_dir / filename}")
        
        plt.close()


def plot_performance_comparison(
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create a single comparison plot with strategies grouped by accuracy level.
    
    Layout: Benchmark (left) | 40% strategies | gap | 60% strategies | gap | 80% strategies | gap | 100% strategies
    
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
    
    # Group strategies by accuracy level
    accuracy_groups = {}
    
    for name, results in strategies.items():
        if 'metrics' not in results:
            continue
        
        # Extract accuracy level from strategy name
        match = re.search(r'_acc_(\d+)', name)
        if match:
            acc_level = int(match.group(1))
        else:
            # No _acc_ suffix means 100% accuracy
            acc_level = 100
        
        if acc_level not in accuracy_groups:
            accuracy_groups[acc_level] = []
        accuracy_groups[acc_level].append((name, results['metrics']))
    
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
    
    # Build combined metrics dataframe with grouping
    all_metrics = []
    all_labels = []
    all_acc_levels = []
    all_positions = []
    
    # Position counter with gaps between accuracy groups
    position = 0
    gap_size = 1  # Space between accuracy groups
    
    # Add benchmark first (position 0)
    if benchmark_metrics is not None:
        all_metrics.append({
            'sharpe_ratio': benchmark_metrics['sharpe_ratio'],
            'annualized_volatility': benchmark_metrics['annualized_volatility'],
            'max_drawdown': benchmark_metrics['max_drawdown'],
            'annualized_return': benchmark_metrics['annualized_return']
        })
        all_labels.append('Benchmark (50/50)')
        all_acc_levels.append('benchmark')
        all_positions.append(position)
        position += 1 + gap_size  # Add gap after benchmark
    
    # Add strategies grouped by accuracy level (40, 60, 80, 100)
    for acc_level in [40, 60, 80, 100]:
        if acc_level not in accuracy_groups:
            continue
        
        strategies_at_level = accuracy_groups[acc_level]
        if not strategies_at_level:
            continue
        
        # Sort strategies by name for consistency
        strategies_at_level = sorted(strategies_at_level, key=lambda x: x[0])
        
        # Add each strategy at this accuracy level
        for name, metrics in strategies_at_level:
            all_metrics.append({
                'sharpe_ratio': metrics['sharpe_ratio'],
                'annualized_volatility': metrics['annualized_volatility'],
                'max_drawdown': metrics['max_drawdown'],
                'annualized_return': metrics['annualized_return']
            })
            all_labels.append(name)
            all_acc_levels.append(acc_level)
            all_positions.append(position)
            position += 1
        
        # Add gap after this accuracy group
        position += gap_size
    
    if not all_metrics:
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df['strategy'] = all_labels
    metrics_df['acc_level'] = all_acc_levels
    metrics_df['position'] = all_positions
    
    # Define color shades for different accuracy levels
    # Using different shades of blue/teal for different accuracies
    accuracy_colors = {
        40: '#8B9DC3',   # Light blue-gray
        60: '#5B7FAE',   # Medium blue-gray
        80: '#3A5F8F',   # Darker blue-gray
        100: '#1E3A5F',  # Darkest blue-gray
        'benchmark': 'darkgray'
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Helper function to get colors
    def get_colors(acc_levels):
        return [accuracy_colors.get(acc, 'gray') for acc in acc_levels]
    
    # Sharpe ratio
    ax = axes[0, 0]
    colors = get_colors(metrics_df['acc_level'])
    bars = ax.bar(metrics_df['position'], metrics_df['sharpe_ratio'], color=colors, width=0.8)
    # Highlight benchmark
    for i, acc in enumerate(metrics_df['acc_level']):
        if acc == 'benchmark':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Volatility
    ax = axes[0, 1]
    colors = get_colors(metrics_df['acc_level'])
    bars = ax.bar(metrics_df['position'], metrics_df['annualized_volatility'], color=colors, width=0.8)
    for i, acc in enumerate(metrics_df['acc_level']):
        if acc == 'benchmark':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Annualized Volatility Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatility', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Max Drawdown
    ax = axes[1, 0]
    colors = get_colors(metrics_df['acc_level'])
    bars = ax.bar(metrics_df['position'], metrics_df['max_drawdown'], color=colors, width=0.8)
    for i, acc in enumerate(metrics_df['acc_level']):
        if acc == 'benchmark':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Max Drawdown', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Annualized Return
    ax = axes[1, 1]
    colors = get_colors(metrics_df['acc_level'])
    bars = ax.bar(metrics_df['position'], metrics_df['annualized_return'], color=colors, width=0.8)
    for i, acc in enumerate(metrics_df['acc_level']):
        if acc == 'benchmark':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Annualized Return Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance Comparison - All Strategies Grouped by Accuracy', 
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


def export_strategies_by_accuracy(
    strategies: Dict[str, Dict],
    output_dir: Optional[Path] = None
) -> None:
    """
    Export strategy data grouped by accuracy level to CSV files.
    
    Each CSV file contains columns: "strategy_name_return", "strategy_name_weight", 
    "strategy_name_forecast" for all strategies at that accuracy level.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts with 'returns', 'weights', 'forecast' keys
    output_dir : Path, optional
        Directory to save CSV files
    """
    # Group strategies by accuracy level
    accuracy_groups = {}
    
    for name, results in strategies.items():
        if 'returns' not in results or results['returns'].empty:
            continue
        
        # Extract accuracy level from strategy name
        match = re.search(r'_acc_(\d+)', name)
        if match:
            acc_level = int(match.group(1))
        else:
            # No _acc_ suffix means 100% accuracy
            acc_level = 100
        
        if acc_level not in accuracy_groups:
            accuracy_groups[acc_level] = {}
        accuracy_groups[acc_level][name] = {
            'returns': results['returns'],
            'weights': results['weights'],
            'forecast': results.get('forecast', pd.Series(dtype=float))
        }
    
    # Export CSV for each accuracy level (only 40%, 60%, 80%)
    for acc_level in [40, 60, 80]:
        if acc_level not in accuracy_groups:
            continue
        
        strategies_at_level = accuracy_groups[acc_level]
        
        if not strategies_at_level:
            continue
        
        # Collect all dates from all strategies
        all_dates = set()
        for name, data in strategies_at_level.items():
            all_dates.update(data['returns'].index)
        
        common_dates = sorted(all_dates)
        
        # Build combined dataframe
        combined_data = {}
        
        for name, data in strategies_at_level.items():
            # Align data to common dates
            returns_aligned = data['returns'].reindex(common_dates)
            weights_aligned = data['weights'].reindex(common_dates)
            forecast_aligned = data['forecast'].reindex(common_dates)
            
            # Add columns for this strategy
            combined_data[f"{name}_return"] = returns_aligned
            combined_data[f"{name}_weight"] = weights_aligned
            combined_data[f"{name}_forecast"] = forecast_aligned
        
        combined_df = pd.DataFrame(combined_data, index=common_dates)
        
        # Save to CSV
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"strategies_acc_{acc_level}.csv"
            combined_df.to_csv(output_dir / filename)
            print(f"Saved strategies CSV (accuracy {acc_level}%) to {output_dir / filename}")
            print(f"  Columns: {list(combined_df.columns)}")

