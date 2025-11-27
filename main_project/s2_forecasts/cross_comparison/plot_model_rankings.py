"""
Create model ranking visualizations.

Shows which model performs best at each horizon and variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from load_results import load_all_metrics, create_performance_table

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_model_rankings(
    performance_df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create ranking plots showing best models by metric.
    
    Parameters:
    -----------
    performance_df : pd.DataFrame
        Performance comparison table
    output_dir : Path, optional
        Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    variables = ['Growth', 'Inflation']
    metrics = ['rmse', 'mae']
    horizons = [1, 3, 6]
    
    for var_idx, variable in enumerate(variables):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[var_idx, metric_idx]
            
            var_data = performance_df[performance_df['variable'] == variable]
            
            # Create ranking data
            rankings = []
            for h in horizons:
                h_data = var_data[var_data['horizon'] == h].copy()
                h_data = h_data.sort_values(metric)
                h_data['rank'] = range(1, len(h_data) + 1)
                
                for _, row in h_data.iterrows():
                    rankings.append({
                        'horizon': h,
                        'model': row['model'],
                        'rank': row['rank'],
                        'value': row[metric]
                    })
            
            ranking_df = pd.DataFrame(rankings)
            
            # Pivot for heatmap
            pivot_rank = ranking_df.pivot_table(
                index='model',
                columns='horizon',
                values='rank',
                aggfunc='first'
            )
            
            # Create heatmap
            sns.heatmap(pivot_rank, annot=True, fmt='.0f', cmap='RdYlGn_r',
                       cbar_kws={'label': 'Rank (1=Best)'}, ax=ax, 
                       linewidths=0.5, vmin=1, vmax=4)
            
            ax.set_title(f'{variable} - {metric.upper()} Rankings', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Forecast Horizon (months)', fontsize=11)
            ax.set_ylabel('Model', fontsize=11)
            ax.set_xticklabels([f'h={h}m' for h in horizons])
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "model_rankings.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved model rankings plot to {output_dir / filename}")
    
    plt.close()


def plot_improvement_bars(
    improvement_df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create bar plots showing relative improvements.
    
    Parameters:
    -----------
    improvement_df : pd.DataFrame
        Relative improvement table
    output_dir : Path, optional
        Output directory
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    variables = ['Growth', 'Inflation']
    metrics = ['rmse_improvement_pct', 'mae_improvement_pct']
    metric_labels = ['RMSE Improvement (%)', 'MAE Improvement (%)']
    
    for idx, (var, metric, label) in enumerate(zip(variables, metrics, metric_labels)):
        ax = axes[idx]
        
        var_data = improvement_df[improvement_df['variable'] == var]
        
        # Prepare data for grouped bar chart
        horizons = sorted(var_data['horizon'].unique())
        models = var_data['model'].unique()
        
        x = np.arange(len(horizons))
        width = 0.2
        
        for i, model in enumerate(models):
            model_data = var_data[var_data['model'] == model]
            values = []
            for h in horizons:
                h_data = model_data[model_data['horizon'] == h]
                if len(h_data) > 0:
                    values.append(h_data.iloc[0][metric])
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Forecast Horizon (months)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{var} - Relative Improvement vs TVP-VAR', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f'h={h}m' for h in horizons])
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "improvement_bars.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved improvement bars plot to {output_dir / filename}")
    
    plt.close()


def main():
    """Generate ranking plots."""
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / "results"
    
    # Load data
    metrics = load_all_metrics(base_dir)
    performance_df = create_performance_table(metrics)
    
    from stats import compute_relative_improvement
    improvement_df = compute_relative_improvement(performance_df, baseline_model='TVP-VAR')
    
    # Create plots
    print("Generating model ranking plots...")
    plot_model_rankings(performance_df, output_dir)
    plot_improvement_bars(improvement_df, output_dir)
    print("Ranking plots created!")


if __name__ == "__main__":
    main()

