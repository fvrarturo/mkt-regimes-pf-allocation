"""
Plotting functions for ERP forecasting strategy visualization.
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
    equity_returns: Optional[pd.Series] = None,
    bond_returns: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None,
    suffix: str = ""
) -> None:
    """
    Plot cumulative returns for all strategies on a single plot.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts with 'returns' and 'weights' keys
    equity_returns : pd.Series, optional
        Equity returns for benchmark calculation
    bond_returns : pd.Series, optional
        Bond returns for benchmark calculation
    output_dir : Path, optional
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define colors for strategies
    strategy_colors = {
        'xgboost': '#2E86AB',           # Blue
        'lstm': '#A23B72',              # Purple
        'xgboost_groq': '#FCBF49',      # Yellow
        'xgboost_openai': '#F77F00',    # Orange
    }
    
    # Plot all strategies
    for name, results in strategies.items():
        if 'returns' not in results or results['returns'].empty:
            continue
        
        returns = results['returns']
        cum_returns = (1 + returns).cumprod()
        
        # Get color and label
        color = strategy_colors.get(name, '#6C757D')
        
        # Format label
        if name == 'xgboost':
            label = 'XGBoost'
        elif name == 'lstm':
            label = 'LSTM'
        elif name == 'xgboost_groq':
            label = 'XGBoost + Groq Sentiment'
        elif name == 'xgboost_openai':
            label = 'XGBoost + OpenAI Sentiment'
        else:
            label = name.replace('_', ' ').title()
        
        ax.plot(cum_returns.index, cum_returns.values,
               label=label, 
               linewidth=2.5, alpha=0.9, color=color, linestyle='-')
        
        # Add fixed portfolio benchmark (average mix)
        if 'weights' in results and equity_returns is not None and bond_returns is not None:
            weights = results['weights']
            if not weights.empty:
                # Calculate average weight over full period
                avg_weight = weights.mean()
                
                # Get common dates
                common_dates = returns.index.intersection(equity_returns.index).intersection(bond_returns.index)
                if len(common_dates) > 0:
                    equity_aligned = equity_returns.reindex(common_dates)
                    bond_aligned = bond_returns.reindex(common_dates)
                    
                    # Create fixed portfolio returns
                    fixed_returns = avg_weight * equity_aligned + (1 - avg_weight) * bond_aligned
                    fixed_cum_returns = (1 + fixed_returns).cumprod()
                    
                    # Format benchmark label
                    equity_pct = int(avg_weight * 100)
                    bond_pct = 100 - equity_pct
                    benchmark_label = f"{label} Fixed ({equity_pct}/{bond_pct})"
                    
                    # Plot as dotted line with same color
                    ax.plot(fixed_cum_returns.index, fixed_cum_returns.values,
                           label=benchmark_label,
                           linewidth=2.0, alpha=0.7, color=color, linestyle=':')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Returns: ERP Forecasting Strategies', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"cumulative_returns_all_strategies{suffix}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative returns plot to {output_dir / filename}")
    
    plt.close()


def plot_performance_comparison(
    strategies: Dict[str, Dict],
    output_dir: Optional[Path] = None,
    suffix: str = ""
) -> None:
    """
    Create a comparison plot with all strategies.
    
    Parameters:
    -----------
    strategies : dict
        Dictionary mapping strategy names to results dicts
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
    
    if not strategy_list:
        return
    
    # Build combined metrics dataframe
    all_metrics = []
    all_labels = []
    all_positions = []
    
    # Position counter
    position = 0
    
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
        if name == 'xgboost':
            label = 'XGBoost'
        elif name == 'lstm':
            label = 'LSTM'
        elif name == 'xgboost_groq':
            label = 'XGBoost + Groq'
        elif name == 'xgboost_openai':
            label = 'XGBoost + OpenAI'
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
    
    # Color scheme
    colors = ['#2E86AB' if 'XGBoost' in label and 'Sentiment' not in label 
              else '#A23B72' if 'LSTM' in label
              else '#FCBF49' if 'Groq' in label
              else '#F77F00' if 'OpenAI' in label
              else '#6C757D' 
              for label in all_labels]
    
    # Sharpe ratio
    ax = axes[0, 0]
    bars = ax.bar(metrics_df['position'], metrics_df['sharpe_ratio'], color=colors, width=0.8)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Volatility
    ax = axes[0, 1]
    bars = ax.bar(metrics_df['position'], metrics_df['annualized_volatility'], color=colors, width=0.8)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Annualized Volatility Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatility', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Max Drawdown
    ax = axes[1, 0]
    bars = ax.bar(metrics_df['position'], metrics_df['max_drawdown'], color=colors, width=0.8)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Maximum Drawdown Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Max Drawdown', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Annualized Return
    ax = axes[1, 1]
    bars = ax.bar(metrics_df['position'], metrics_df['annualized_return'], color=colors, width=0.8)
    ax.set_xticks(metrics_df['position'])
    ax.set_xticklabels(metrics_df['strategy'], rotation=45, ha='right', fontsize=9)
    ax.set_title('Annualized Return Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance Comparison - ERP Forecasting Strategies', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"performance_comparison_all_strategies{suffix}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {output_dir / filename}")
        
        # Save metrics to CSV
        csv_filename = f"performance_comparison_all_strategies{suffix}.csv"
        csv_df = metrics_df[['strategy', 'sharpe_ratio', 'annualized_return', 'annualized_volatility', 'max_drawdown']].copy()
        csv_df.to_csv(output_dir / csv_filename, index=False)
        print(f"Saved performance comparison data to {output_dir / csv_filename}")
    
    plt.close()

