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
    output_dir: Optional[Path] = None,
    show_fixed_portfolios: bool = True
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
    show_fixed_portfolios : bool
        If True, plot fixed portfolio benchmarks as dotted lines
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define colors for strategies
    strategy_colors = {
        'hmm': '#A23B72',        # Purple for any HMM strategy
        '2x2': '#FCBF49',        # Yellow for any 2x2 strategy
        'fixed_50_50_benchmark': 'darkgray'   # Gray
    }
    
    # Plot all strategies (excluding fixed portfolios for now)
    for name, results in strategies.items():
        if 'returns' not in results or results['returns'].empty:
            continue
        
        # Skip fixed portfolios - we'll plot them separately
        if show_fixed_portfolios and 'fixed_portfolio' in name:
            continue
        
        # Skip the general benchmark - we'll handle it separately
        if name == 'fixed_50_50_benchmark':
            continue
        
        returns = results['returns']
        cum_returns = (1 + returns).cumprod()
        
        # Get color based on strategy type
        if name.startswith('hmm_'):
            color = strategy_colors.get('hmm', '#A23B72')
            label = 'HMM Based'
        elif name.startswith('2x2_'):
            color = strategy_colors.get('2x2', '#FCBF49')
            label = '2x2 Based'
        else:
            color = strategy_colors.get(name, '#6C757D')
            label = name.replace('_', ' ').title()
        
        linewidth = 2.5
        linestyle = '-'
        
        ax.plot(cum_returns.index, cum_returns.values,
               label=label, 
               linewidth=linewidth, alpha=0.9, color=color, linestyle=linestyle)
    
    # Plot fixed portfolio benchmarks as dotted lines
    if show_fixed_portfolios:
        for name, results in strategies.items():
            if 'fixed_portfolio' in name and 'returns' in results and not results['returns'].empty:
                # Extract base strategy name (e.g., 'hmm_forecast_based' from 'hmm_forecast_based_fixed_portfolio')
                base_name = name.replace('_fixed_portfolio', '')
                color = strategy_colors.get(base_name, '#6C757D')
                
                returns = results['returns']
                cum_returns = (1 + returns).cumprod()
                
                # Format label with average weight
                avg_weight = results.get('avg_weight', 0.5)
                equity_pct = int(avg_weight * 100)
                bond_pct = 100 - equity_pct
                
                # Use simplified base name
                if base_name.startswith('hmm_'):
                    base_label = 'HMM Based'
                elif base_name.startswith('2x2_'):
                    base_label = '2x2 Based'
                else:
                    base_label = base_name.replace('_', ' ').title()
                
                label = f"{base_label} Fixed ({equity_pct}/{bond_pct})"
                
                ax.plot(cum_returns.index, cum_returns.values,
                       label=label,
                       linewidth=2.0, alpha=0.7, color=color, linestyle=':', dashes=(5, 5))
    
    # Plot the general 50/50 benchmark
    if 'fixed_50_50_benchmark' in strategies:
        results = strategies['fixed_50_50_benchmark']
        if 'returns' in results and not results['returns'].empty:
            returns = results['returns']
            cum_returns = (1 + returns).cumprod()
            ax.plot(cum_returns.index, cum_returns.values,
                   label='Benchmark (50/50)',
                   linewidth=2.5, alpha=0.9, color='darkgray', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Returns: All Regime-Based Strategies', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
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
        
        # Use simplified labels
        if name.startswith('hmm_'):
            label = 'HMM Based'
        elif name.startswith('2x2_'):
            label = '2x2 Based'
        else:
            label = name.replace('_', ' ').title()
        
        all_labels.append(label)
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
        
        # Save metrics to CSV
        csv_filename = "performance_comparison_all_strategies.csv"
        # Reorder columns for better readability
        csv_df = metrics_df[['strategy', 'sharpe_ratio', 'annualized_return', 'annualized_volatility', 'max_drawdown']].copy()
        csv_df.to_csv(output_dir / csv_filename, index=False)
        print(f"Saved performance comparison data to {output_dir / csv_filename}")
    
    plt.close()
    
    return metrics_df


def plot_weights_over_time(
    strategies: Dict[str, Dict],
    all_macro_df: Optional[pd.DataFrame] = None,
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Plot equity weights and regime weights/positions over time for HMM and 2x2 strategies.
    
    Creates two subplots:
    1. Top: Portfolio weights (equity weights) over time
    2. Bottom: Regime probabilities for HMM, regime positions for 2x2
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts with 'weights' key
    all_macro_df : pd.DataFrame, optional
        Full macro dataset for computing regime probabilities/assignments
    base_dir : Path, optional
        Base directory for loading models
    output_dir : Path, optional
        Directory to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax1 = axes[0]  # Portfolio weights
    ax2 = axes[1]  # Regime weights/positions
    
    # Define colors for strategies (match any HMM or 2x2 strategy)
    strategy_colors = {
        'hmm': '#A23B72',        # Purple for any HMM strategy
        '2x2': '#FCBF49',        # Yellow for any 2x2 strategy
    }
    
    # Regime colors for HMM (4 regimes)
    regime_colors_hmm = ['#8B0000', '#FF6347', '#4169E1', '#32CD32']  # Dark red, tomato, royal blue, lime green
    
    # Find HMM and 2x2 strategies
    hmm_strategy_name = None
    two_by_two_strategy_name = None
    
    for name in strategies.keys():
        if name.startswith('hmm_') and 'fixed_portfolio' not in name and 'benchmark' not in name:
            hmm_strategy_name = name
        elif name.startswith('2x2_') and 'fixed_portfolio' not in name and 'benchmark' not in name:
            two_by_two_strategy_name = name
    
    # === TOP SUBPLOT: Portfolio Weights ===
    for name, results in strategies.items():
        if 'weights' not in results or results['weights'].empty:
            continue
        
        # Skip fixed portfolios and benchmarks
        if 'fixed_portfolio' in name or 'benchmark' in name:
            continue
        
        # Only plot actual-based strategies (HMM or 2x2)
        if not (name.startswith('hmm_') or name.startswith('2x2_')):
            continue
        
        weights = results['weights']
        
        # Get color based on strategy type
        if name.startswith('hmm_'):
            color = strategy_colors.get('hmm', '#A23B72')
            label = 'HMM Based'
        elif name.startswith('2x2_'):
            color = strategy_colors.get('2x2', '#FCBF49')
            label = '2x2 Based'
        else:
            color = '#6C757D'
            label = name.replace('_', ' ').title()
        
        linewidth = 2.0
        alpha = 0.8
        
        ax1.plot(weights.index, weights.values,
               label=label, 
               linewidth=linewidth, alpha=alpha, color=color)
    
    # Add horizontal line at 0.5 (50/50)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='50/50')
    
    ax1.set_ylabel('Equity Weight', fontsize=12, fontweight='bold')
    ax1.set_title('Portfolio Weights Over Time', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3)
    
    # === BOTTOM SUBPLOT: Regime Weights/Positions ===
    # Create dual y-axes: left for HMM probabilities, right for 2x2 regimes
    ax2_left = ax2  # Left axis for HMM probabilities
    ax2_right = ax2.twinx()  # Right axis for 2x2 regimes
    
    if all_macro_df is not None and base_dir is not None:
        # Import here to avoid circular imports
        from hmm_forecasts import load_hmm_model_and_coefficients, get_regime_probabilities
        from two_by_two_forecasts import load_2x2_regime_definitions_and_coefficients, get_hard_regime_assignment
        
        # Plot HMM regime probabilities on left axis
        if hmm_strategy_name:
            try:
                # Extract combination and K from strategy name
                # Format: hmm_2vars_inflation_market_volatility_k4_actual_based
                parts = hmm_strategy_name.split('_')
                k_idx = parts.index('k4') if 'k4' in parts else None
                if k_idx is not None:
                    k = int(parts[k_idx].replace('k', ''))
                    # Reconstruct combination name (everything between hmm_ and _k4)
                    combo_parts = parts[1:k_idx]
                    combination = '_'.join(combo_parts)
                    
                    # Load HMM model and get regime probabilities
                    hmm_model, _, macro_df = load_hmm_model_and_coefficients(
                        base_dir=base_dir,
                        combination=combination,
                        k=k
                    )
                    
                    # Get regime probabilities aligned to strategy dates
                    hmm_probs = get_regime_probabilities(hmm_model, all_macro_df)
                    
                    # Align to strategy dates
                    if hmm_strategy_name in strategies:
                        strategy_dates = strategies[hmm_strategy_name]['weights'].index
                        hmm_probs_aligned = hmm_probs.reindex(strategy_dates).dropna()
                        
                        # Plot each regime probability on left axis
                        for regime_idx in range(hmm_model.n_regimes):
                            prob_col = f'prob_R{regime_idx}'
                            if prob_col in hmm_probs_aligned.columns:
                                ax2_left.plot(hmm_probs_aligned.index, hmm_probs_aligned[prob_col].values,
                                           label=f'HMM R{regime_idx}', 
                                           linewidth=1.5, alpha=0.7, 
                                           color=regime_colors_hmm[regime_idx % len(regime_colors_hmm)],
                                           linestyle='-')
            except Exception as e:
                print(f"Warning: Could not plot HMM regime probabilities: {e}")
                import traceback
                traceback.print_exc()
        
        # Plot 2x2 regime positions on right axis
        if two_by_two_strategy_name:
            try:
                # Load 2x2 regime definitions
                regime_def, _, macro_df = load_2x2_regime_definitions_and_coefficients(
                    base_dir=base_dir
                )
                
                # Get hard regime assignments
                growth_actual = all_macro_df["growth_factor"]
                inflation_actual = all_macro_df["inflation_factor"]
                two_by_two_regimes = get_hard_regime_assignment(
                    regime_def, growth_actual, inflation_actual
                )
                
                # Align to strategy dates
                if two_by_two_strategy_name in strategies:
                    strategy_dates = strategies[two_by_two_strategy_name]['weights'].index
                    regimes_aligned = two_by_two_regimes.reindex(strategy_dates).dropna()
                    
                    # Plot each regime category as discontinuous lines
                    # Create separate series for each regime (0, 1, 2, 3)
                    regime_colors_2x2 = ['#FFD700', '#FF8C00', '#FF4500', '#DC143C']  # Gold, dark orange, red orange, crimson
                    
                    for regime_id in range(4):
                        # Create a series where values are only shown when that regime is active
                        regime_mask = (regimes_aligned == regime_id)
                        regime_values = regimes_aligned.copy()
                        regime_values[~regime_mask] = np.nan
                        
                        # Plot as step function with gaps (discontinuous)
                        ax2_right.step(regime_values.index, regime_values.values,
                                     where='post', label=f'2x2 R{regime_id}',
                                     linewidth=2.0, alpha=0.8, 
                                     color=regime_colors_2x2[regime_id],
                                     linestyle='-', drawstyle='steps-post')
            except Exception as e:
                print(f"Warning: Could not plot 2x2 regime positions: {e}")
                import traceback
                traceback.print_exc()
    
    # Configure left axis (HMM probabilities)
    ax2_left.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2_left.set_ylabel('HMM Regime Probability', fontsize=12, fontweight='bold', color='#A23B72')
    ax2_left.set_ylim([-0.05, 1.05])
    ax2_left.tick_params(axis='y', labelcolor='#A23B72')
    ax2_left.grid(alpha=0.3, axis='y')
    
    # Configure right axis (2x2 regimes)
    ax2_right.set_ylabel('2x2 Regime Position', fontsize=12, fontweight='bold', color='#FCBF49')
    ax2_right.set_ylim([-0.2, 3.2])
    ax2_right.set_yticks([0, 1, 2, 3])
    ax2_right.tick_params(axis='y', labelcolor='#FCBF49')
    
    # Combine legends
    lines_left, labels_left = ax2_left.get_legend_handles_labels()
    lines_right, labels_right = ax2_right.get_legend_handles_labels()
    ax2_left.legend(lines_left + lines_right, labels_left + labels_right, 
                   loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    ax2_left.set_title('Regime Weights (HMM) and Positions (2x2) Over Time', fontsize=13, fontweight='bold')
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "weights_over_time.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved weights plot to {output_dir / filename}")
    
    plt.close()


