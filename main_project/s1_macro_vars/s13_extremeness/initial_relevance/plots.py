"""
Plotting functions for extremeness models visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_extremeness_timeseries(results_dict, model_name, output_dir):
    """
    Plot time series of extremeness index with multiple percentile thresholds.
    
    Parameters:
    -----------
    results_dict : dict
        Results dictionary from model
    model_name : str
        Name of the model
    output_dir : Path
        Output directory
    """
    results_df = results_dict['results_df']
    percentile_flags = results_dict.get('percentile_flags', {})
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot extremeness time series
    ax.plot(results_df.index, results_df['extremeness'], alpha=0.7, linewidth=1, color='black', label='Extremeness Index')
    
    # Add percentile threshold lines
    percentile_colors = {99: 'darkred', 95: 'red', 90: 'orange', 80: 'yellow'}
    percentile_labels = {99: '99th (1%)', 95: '95th (5%)', 90: '90th (10%)', 80: '80th (20%)'}
    
    for p in [99, 95, 90, 80]:
        if p in percentile_flags:
            threshold = percentile_flags[p]['threshold']
            ax.axhline(y=threshold, color=percentile_colors[p], linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'{percentile_labels[p]}')
    
    # Fill areas for different percentile levels
    if 99 in percentile_flags:
        ax.fill_between(results_df.index, 0, results_df['extremeness'], 
                       where=results_df['is_extreme_p99'], 
                       alpha=0.4, color='darkred', label='Top 1%')
    if 95 in percentile_flags:
        ax.fill_between(results_df.index, 0, results_df['extremeness'], 
                       where=results_df['is_extreme_p95'] & ~results_df['is_extreme_p99'], 
                       alpha=0.3, color='red', label='Top 5%')
    if 90 in percentile_flags:
        ax.fill_between(results_df.index, 0, results_df['extremeness'], 
                       where=results_df['is_extreme_p90'] & ~results_df['is_extreme_p95'], 
                       alpha=0.2, color='orange', label='Top 10%')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Extremeness Index', fontsize=12)
    ax.set_title(f'Extremeness Time Series: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeseries plot to {output_path}")


def plot_extremeness_histogram_combined(all_results, output_dir):
    """
    Plot combined histogram/density plots for all models (stacked vertically).
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with all model results
    output_dir : Path
        Output directory
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 5 * n_models))
    if n_models == 1:
        axes = [axes]
    
    percentile_colors = {99: 'darkred', 95: 'red', 90: 'orange', 80: 'yellow'}
    percentile_labels = {99: '99th (1%)', 95: '95th (5%)', 90: '90th (10%)', 80: '80th (20%)'}
    
    for idx, (model_name, results_dict) in enumerate(all_results.items()):
        results_df = results_dict['results_df']
        percentile_flags = results_dict.get('percentile_flags', {})
        ax = axes[idx]
        
        # Histogram
        ax.hist(results_df['extremeness'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        
        # Add percentile threshold lines
        for p in [99, 95, 90, 80]:
            if p in percentile_flags:
                threshold = percentile_flags[p]['threshold']
                ax.axvline(x=threshold, color=percentile_colors[p], linestyle='--', 
                          linewidth=2, alpha=0.8, label=percentile_labels[p])
        
        ax.set_xlabel('Extremeness Index', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Extremeness Distribution: All Models', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "extremeness_histogram_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined histogram plot to {output_path}")


def plot_extremeness_histogram(results_dict, model_name, output_dir):
    """
    Plot histogram/density of extremeness index with multiple percentile thresholds.
    
    Parameters:
    -----------
    results_dict : dict
        Results dictionary from model
    model_name : str
        Name of the model
    output_dir : Path
        Output directory
    """
    results_df = results_dict['results_df']
    percentile_flags = results_dict.get('percentile_flags', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(results_df['extremeness'], bins=50, alpha=0.7, edgecolor='black')
    
    # Add percentile threshold lines
    percentile_colors = {99: 'darkred', 95: 'red', 90: 'orange', 80: 'yellow'}
    percentile_labels = {99: '99th (1%)', 95: '95th (5%)', 90: '90th (10%)', 80: '80th (20%)'}
    
    for p in [99, 95, 90, 80]:
        if p in percentile_flags:
            threshold = percentile_flags[p]['threshold']
            ax1.axvline(x=threshold, color=percentile_colors[p], linestyle='--', 
                       linewidth=2, alpha=0.8, label=percentile_labels[p])
    
    ax1.set_xlabel('Extremeness Index', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram of Extremeness', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Density plot with percentile shading
    sns.kdeplot(data=results_df, x='extremeness', ax=ax2, color='blue', linewidth=2)
    
    # Add percentile threshold lines
    for p in [99, 95, 90, 80]:
        if p in percentile_flags:
            threshold = percentile_flags[p]['threshold']
            ax2.axvline(x=threshold, color=percentile_colors[p], linestyle='--', 
                       linewidth=2, alpha=0.8, label=percentile_labels[p])
    
    # Shade areas
    extremeness_vals = results_df['extremeness'].values
    x_min, x_max = extremeness_vals.min(), extremeness_vals.max()
    x_range = np.linspace(x_min, x_max, 1000)
    
    if 99 in percentile_flags:
        threshold_99 = percentile_flags[99]['threshold']
        ax2.fill_between(x_range, 0, 10, where=(x_range >= threshold_99), 
                        alpha=0.3, color='darkred')
    if 95 in percentile_flags:
        threshold_95 = percentile_flags[95]['threshold']
        ax2.fill_between(x_range, 0, 10, where=(x_range >= threshold_95) & (x_range < threshold_99 if 99 in percentile_flags else True), 
                        alpha=0.2, color='red')
    
    ax2.set_xlabel('Extremeness Index', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Density Plot with Percentiles', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle(f'Extremeness Distribution: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram plot to {output_path}")


def plot_extremeness_vs_erp_combined(all_results, output_dir):
    """
    Plot combined extremeness vs ERP scatter plots for all models (stacked vertically).
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with all model results
    output_dir : Path
        Output directory
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 6 * n_models))
    if n_models == 1:
        axes = [axes]
    
    percentile_colors = {99: 'darkred', 95: 'red', 90: 'orange', 80: 'yellow'}
    percentile_names = {99: 'Top 1%', 95: 'Top 5%', 90: 'Top 10%', 80: 'Top 20%'}
    
    for idx, (model_name, results_dict) in enumerate(all_results.items()):
        results_df = results_dict['results_df']
        percentile_flags = results_dict.get('percentile_flags', {})
        ax = axes[idx]
        
        # Color by percentile groups
        if 99 in percentile_flags and 'is_extreme_p99' in results_df.columns:
            p99_mask = results_df['is_extreme_p99']
            ax.scatter(results_df.loc[p99_mask, 'extremeness'], 
                      results_df.loc[p99_mask, 'ERP'],
                      alpha=0.8, s=40, label='Top 1%', color='darkred', marker='^')
        
        if 95 in percentile_flags and 'is_extreme_p95' in results_df.columns:
            p95_mask = results_df['is_extreme_p95'] & ~results_df.get('is_extreme_p99', False)
            ax.scatter(results_df.loc[p95_mask, 'extremeness'], 
                      results_df.loc[p95_mask, 'ERP'],
                      alpha=0.7, s=35, label='Top 5%', color='red', marker='s')
        
        if 90 in percentile_flags and 'is_extreme_p90' in results_df.columns:
            p90_mask = results_df['is_extreme_p90'] & ~results_df.get('is_extreme_p95', False)
            ax.scatter(results_df.loc[p90_mask, 'extremeness'], 
                      results_df.loc[p90_mask, 'ERP'],
                      alpha=0.6, s=30, label='Top 10%', color='orange', marker='D')
        
        if 80 in percentile_flags and 'is_extreme_p80' in results_df.columns:
            p80_mask = results_df['is_extreme_p80'] & ~results_df.get('is_extreme_p90', False)
            ax.scatter(results_df.loc[p80_mask, 'extremeness'], 
                      results_df.loc[p80_mask, 'ERP'],
                      alpha=0.5, s=25, label='Top 20%', color='yellow', marker='o')
        
        # Normal points (below 80th percentile)
        normal_mask = ~results_df.get('is_extreme_p80', results_df['is_extreme'])
        ax.scatter(results_df.loc[normal_mask, 'extremeness'], 
                  results_df.loc[normal_mask, 'ERP'],
                  alpha=0.4, s=15, label='Normal', color='blue', marker='.')
        
        # Add percentile threshold lines
        for p in [99, 95, 90, 80]:
            if p in percentile_flags:
                threshold = percentile_flags[p]['threshold']
                ax.axvline(x=threshold, color=percentile_colors[p], linestyle='--', 
                          linewidth=1, alpha=0.5)
        
        # Add correlation text
        correlation = np.corrcoef(results_df['extremeness'], results_df['ERP'])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Extremeness Index', fontsize=12)
        ax.set_ylabel('ERP', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=3)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Extremeness vs ERP: All Models', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / "extremeness_vs_erp_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined extremeness vs ERP plot to {output_path}")


def plot_extremeness_vs_erp(results_dict, model_name, output_dir):
    """
    Plot scatter: extremeness vs ERP with percentile groups.
    
    Parameters:
    -----------
    results_dict : dict
        Results dictionary from model
    model_name : str
        Name of the model
    output_dir : Path
        Output directory
    """
    results_df = results_dict['results_df']
    percentile_flags = results_dict.get('percentile_flags', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by percentile groups
    if 99 in percentile_flags and 'is_extreme_p99' in results_df.columns:
        p99_mask = results_df['is_extreme_p99']
        ax.scatter(results_df.loc[p99_mask, 'extremeness'], 
                  results_df.loc[p99_mask, 'ERP'],
                  alpha=0.8, s=40, label='Top 1%', color='darkred', marker='^')
    
    if 95 in percentile_flags and 'is_extreme_p95' in results_df.columns:
        p95_mask = results_df['is_extreme_p95'] & ~results_df.get('is_extreme_p99', False)
        ax.scatter(results_df.loc[p95_mask, 'extremeness'], 
                  results_df.loc[p95_mask, 'ERP'],
                  alpha=0.7, s=35, label='Top 5%', color='red', marker='s')
    
    if 90 in percentile_flags and 'is_extreme_p90' in results_df.columns:
        p90_mask = results_df['is_extreme_p90'] & ~results_df.get('is_extreme_p95', False)
        ax.scatter(results_df.loc[p90_mask, 'extremeness'], 
                  results_df.loc[p90_mask, 'ERP'],
                  alpha=0.6, s=30, label='Top 10%', color='orange', marker='D')
    
    if 80 in percentile_flags and 'is_extreme_p80' in results_df.columns:
        p80_mask = results_df['is_extreme_p80'] & ~results_df.get('is_extreme_p90', False)
        ax.scatter(results_df.loc[p80_mask, 'extremeness'], 
                  results_df.loc[p80_mask, 'ERP'],
                  alpha=0.5, s=25, label='Top 20%', color='yellow', marker='o')
    
    # Normal points (below 80th percentile)
    normal_mask = ~results_df.get('is_extreme_p80', results_df['is_extreme'])
    ax.scatter(results_df.loc[normal_mask, 'extremeness'], 
              results_df.loc[normal_mask, 'ERP'],
              alpha=0.4, s=15, label='Normal', color='blue', marker='.')
    
    # Add percentile threshold lines
    percentile_colors = {99: 'darkred', 95: 'red', 90: 'orange', 80: 'yellow'}
    for p in [99, 95, 90, 80]:
        if p in percentile_flags:
            threshold = percentile_flags[p]['threshold']
            ax.axvline(x=threshold, color=percentile_colors[p], linestyle='--', 
                      linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Extremeness Index', fontsize=12)
    ax.set_ylabel('ERP', fontsize=12)
    ax.set_title(f'Extremeness vs ERP: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    
    # Add correlation text
    correlation = np.corrcoef(results_df['extremeness'], results_df['ERP'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_vs_erp.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved extremeness vs ERP plot to {output_path}")


def plot_erp_by_state_all_models(all_results, output_dir):
    """
    Plot combined boxplots: ERP by normal vs extreme state for all models with multiple percentiles.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with all model results
        Keys: model names, Values: result dictionaries
    output_dir : Path
        Output directory
    """
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Prepare data for all models with multiple percentiles
    box_data = []
    positions = []
    box_labels = []
    
    # Position tracking
    current_pos = 1
    spacing = 3  # Space between model groups
    box_width = 0.5
    
    model_labels = []
    percentiles = [99, 95, 90, 80]
    percentile_colors = {99: 'darkred', 95: 'red', 90: 'orange', 80: 'yellow'}
    percentile_names = {99: 'Top 1%', 95: 'Top 5%', 90: 'Top 10%', 80: 'Top 20%'}
    
    boxes_per_model_list = []
    
    for model_name, results_dict in all_results.items():
        results_df = results_dict['results_df']
        model_labels.append(model_name)
        
        boxes_this_model = 0
        
        # Normal state (below 80th percentile)
        if 'is_extreme_p80' in results_df.columns:
            normal_mask = ~results_df['is_extreme_p80']
        else:
            normal_mask = ~results_df['is_extreme']
        normal_data = results_df[normal_mask]['ERP'].values
        if len(normal_data) > 0:
            box_data.append(normal_data)
            positions.append(current_pos + boxes_this_model)
            box_labels.append('Normal')
            boxes_this_model += 1
        
        # Extreme states for each percentile (non-overlapping)
        for p in percentiles:
            col_name = f'is_extreme_p{p}'
            if col_name in results_df.columns:
                # Get data for this percentile excluding higher percentiles
                if p == 99:
                    p_mask = results_df[col_name]
                elif p == 95:
                    p_mask = results_df[col_name] & ~results_df.get('is_extreme_p99', False)
                elif p == 90:
                    p_mask = results_df[col_name] & ~results_df.get('is_extreme_p95', False)
                elif p == 80:
                    p_mask = results_df[col_name] & ~results_df.get('is_extreme_p90', False)
                
                p_data = results_df[p_mask]['ERP'].values
                if len(p_data) > 0:
                    box_data.append(p_data)
                    positions.append(current_pos + boxes_this_model)
                    box_labels.append(percentile_names[p])
                    boxes_this_model += 1
        
        boxes_per_model_list.append(boxes_this_model)
        
        # Add spacing before next model group
        current_pos += spacing + boxes_this_model
    
    # Create boxplot
    bp = ax.boxplot(box_data, positions=positions, widths=box_width, patch_artist=True)
    
    # Color boxes
    colors = []
    color_idx = 0
    for i, label in enumerate(box_labels):
        if label == 'Normal':
            colors.append('lightblue')
        elif label == 'Top 1%':
            colors.append('darkred')
        elif label == 'Top 5%':
            colors.append('red')
        elif label == 'Top 10%':
            colors.append('orange')
        elif label == 'Top 20%':
            colors.append('yellow')
        else:
            colors.append('gray')
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Calculate group positions (center of each model's boxes)
    group_positions = []
    pos_idx = 0
    for i, boxes_count in enumerate(boxes_per_model_list):
        start_pos = positions[pos_idx]
        end_pos = positions[pos_idx + boxes_count - 1]
        group_pos = (start_pos + end_pos) / 2
        group_positions.append(group_pos)
        pos_idx += boxes_count
    
    ax.set_xticks(group_positions)
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Normal', alpha=0.7),
        Patch(facecolor='yellow', label='Top 20%', alpha=0.7),
        Patch(facecolor='orange', label='Top 10%', alpha=0.7),
        Patch(facecolor='red', label='Top 5%', alpha=0.7),
        Patch(facecolor='darkred', label='Top 1%', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add vertical lines to separate model groups
    pos_idx = 0
    for i in range(len(model_labels) - 1):
        pos_idx += boxes_per_model_list[i]
        last_box_i = positions[pos_idx - 1]
        first_box_i1 = positions[pos_idx]
        sep_pos = (last_box_i + first_box_i1) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_ylabel('ERP', fontsize=12)
    ax.set_title('ERP Distribution by Extremeness Percentiles: All Models', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "erp_boxplot_all_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined ERP boxplot to {output_path}")


def plot_pca_scree(pca_results, model_name, output_dir):
    """
    Plot scree plot of PCA eigenvalues.
    
    Parameters:
    -----------
    pca_results : dict
        PCA results dictionary
    model_name : str
        Name of the model
    output_dir : Path
        Output directory
    """
    explained_var = pca_results['explained_variance_ratio']
    cumulative_var = pca_results['cumulative_variance']
    n_components = pca_results['n_components']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    ax1.axvline(x=n_components, color='red', linestyle='--', linewidth=2, label=f'Selected: {n_components} PCs')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linewidth=2)
    ax2.axhline(y=pca_results['variance_threshold'], color='red', linestyle='--', 
               linewidth=2, label=f"Threshold: {pca_results['variance_threshold']*100:.0f}%")
    ax2.axvline(x=n_components, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'PCA Analysis: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_scree.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scree plot to {output_path}")


def plot_pca_biplot(pca_results, model_name, output_dir):
    """
    Plot biplot: PC1 vs PC2, colored by extremeness.
    
    Parameters:
    -----------
    pca_results : dict
        PCA results dictionary
    model_name : str
        Name of the model
    output_dir : Path
        Output directory
    """
    results_df = pca_results['results_df']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by extreme state
    normal_mask = ~results_df['is_extreme']
    extreme_mask = results_df['is_extreme']
    
    scatter1 = ax.scatter(results_df.loc[normal_mask, 'PC1'], 
                        results_df.loc[normal_mask, 'PC2'],
                        c=results_df.loc[normal_mask, 'extremeness'],
                        cmap='Blues', alpha=0.6, s=30, label='Normal')
    scatter2 = ax.scatter(results_df.loc[extreme_mask, 'PC1'], 
                        results_df.loc[extreme_mask, 'PC2'],
                        c=results_df.loc[extreme_mask, 'extremeness'],
                        cmap='Reds', alpha=0.8, s=50, label='Extreme', marker='^')
    
    ax.set_xlabel(f"PC1 ({pca_results['explained_variance_ratio'][0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_results['explained_variance_ratio'][1]*100:.1f}% variance)", fontsize=12)
    ax.set_title(f'PCA Biplot: {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.colorbar(scatter2, ax=ax, label='Extremeness')
    plt.tight_layout()
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_biplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved biplot to {output_path}")


def plot_extremeness_comparison(comparison_results, output_dir):
    """
    Plot comparison between two extremeness measures.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from compare_extremeness_measures
    output_dir : Path
        Output directory
    """
    comparison_df = comparison_results['comparison_df']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot of extremeness scores
    ax1 = axes[0]
    ax1.scatter(comparison_df.iloc[:, 0], comparison_df.iloc[:, 1], alpha=0.5)
    ax1.set_xlabel(comparison_df.columns[0], fontsize=12)
    ax1.set_ylabel(comparison_df.columns[1], fontsize=12)
    ax1.set_title(f'Extremeness Correlation: {comparison_results["correlation"]:.3f}', 
                 fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Overlap visualization
    ax2 = axes[1]
    overlap_matrix = np.array([
        [np.sum(~comparison_df.iloc[:, 2] & ~comparison_df.iloc[:, 3]),  # Both normal
         np.sum(~comparison_df.iloc[:, 2] & comparison_df.iloc[:, 3])],   # Model1 normal, Model2 extreme
        [np.sum(comparison_df.iloc[:, 2] & ~comparison_df.iloc[:, 3]),   # Model1 extreme, Model2 normal
         np.sum(comparison_df.iloc[:, 2] & comparison_df.iloc[:, 3])]     # Both extreme
    ])
    
    im = ax2.imshow(overlap_matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Normal', 'Extreme'])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Extreme'])
    ax2.set_xlabel('Model 2', fontsize=12)
    ax2.set_ylabel('Model 1', fontsize=12)
    ax2.set_title(f'Extreme State Overlap: {comparison_results["overlap_rate"]*100:.1f}%', 
                 fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{overlap_matrix[i, j]}', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    output_path = output_dir / "extremeness_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")

