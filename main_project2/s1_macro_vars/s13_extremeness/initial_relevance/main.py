"""
Main script for extremeness models analysis.

Runs Isolation Forest and PCA Distance models, generates statistics and plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import model files
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import load_data, prepare_data, get_feature_columns
from isolation_forest import run_isolation_forest_analysis
from pca_distance import run_pca_distance_analysis
from stats import (compute_erp_statistics, test_erp_differences, 
                  compute_erp_statistics_by_percentiles)
from plots import (plot_extremeness_vs_erp_combined, plot_extremeness_histogram_combined, 
                  plot_erp_by_state_all_models)


def main():
    """Main execution function."""
    print("="*70)
    print("Extremeness Models Analysis")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    erp_df, macro_df, _ = load_data()
    
    print(f"   ERP data: {len(erp_df)} observations")
    print(f"   Macro data: {len(macro_df)} observations")
    
    # Prepare data
    print("\n2. Preparing data...")
    df = prepare_data(erp_df, macro_df, None, include_sentiment=False)
    print(f"   Merged data: {len(df)} observations")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Create output directory (in parent directory's results folder)
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature columns
    macro_cols = get_feature_columns(include_sentiment=False)
    
    # Run Isolation Forest model
    print("\n3. Running Isolation Forest model...")
    if_results = run_isolation_forest_analysis(
        df, macro_cols, contamination=0.1, threshold_percentile=90
    )
    print(f"   Extreme states: {np.sum(if_results['is_extreme'])} ({np.mean(if_results['is_extreme'])*100:.1f}%)")
    
    # Run PCA Distance model
    print("\n4. Running PCA Distance model...")
    pca_results = run_pca_distance_analysis(
        df, macro_cols, variance_threshold=0.85, 
        distance_method='euclidean', threshold_percentile=90
    )
    print(f"   Selected components: {pca_results['n_components']}")
    print(f"   Variance explained: {pca_results['cumulative_variance'][-1]*100:.1f}%")
    print(f"   Extreme states: {np.sum(pca_results['is_extreme'])} ({np.mean(pca_results['is_extreme'])*100:.1f}%)")
    
    # Store all results
    all_results = {
        'Isolation Forest': if_results,
        'PCA Distance': pca_results
    }
    
    # Generate statistics
    print("\n5. Computing statistics...")
    
    # ERP statistics for each model
    erp_stats_all = {}
    test_results_all = {}
    
    for model_name, results in all_results.items():
        print(f"\n   {model_name}:")
        erp_stats = compute_erp_statistics(results['results_df'])
        erp_stats_all[model_name] = erp_stats
        
        test_results = test_erp_differences(results['results_df'])
        test_results_all[model_name] = test_results
        
        print(f"      Normal ERP mean: {erp_stats.loc[0, 'mean']:.4f}")
        print(f"      Extreme ERP mean: {erp_stats.loc[1, 'mean']:.4f}")
        print(f"      Mean difference: {test_results['t_test']['mean_diff']:.4f} (p={test_results['t_test']['pvalue']:.4f})")
    
    # Save ERP statistics
    for model_name, stats_df in erp_stats_all.items():
        filename = model_name.lower().replace(' ', '_')
        stats_path = output_dir / f"{filename}_erp_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"   Saved ERP statistics to {stats_path}")
    
    # Compute and save ERP statistics by percentile thresholds
    print("\n   Computing ERP statistics by percentile thresholds...")
    percentile_stats_all = {}
    for model_name, results in all_results.items():
        percentile_stats = compute_erp_statistics_by_percentiles(results['results_df'])
        percentile_stats_all[model_name] = percentile_stats
        
        filename = model_name.lower().replace(' ', '_')
        percentile_path = output_dir / f"{filename}_erp_statistics_by_percentiles.csv"
        percentile_stats.to_csv(percentile_path, index=False)
        print(f"   Saved percentile statistics to {percentile_path}")
    
    # Save statistical test results
    print("\n   Saving statistical test results...")
    test_results_list = []
    for model_name, test_results in test_results_all.items():
        test_results_list.append({
            'model': model_name,
            't_statistic': test_results['t_test']['statistic'],
            't_pvalue': test_results['t_test']['pvalue'],
            'normal_mean': test_results['t_test']['normal_mean'],
            'extreme_mean': test_results['t_test']['extreme_mean'],
            'mean_difference': test_results['t_test']['mean_diff'],
            'ks_statistic': test_results['ks_test']['statistic'],
            'ks_pvalue': test_results['ks_test']['pvalue'],
            'mannwhitney_statistic': test_results['mannwhitney_test']['statistic'],
            'mannwhitney_pvalue': test_results['mannwhitney_test']['pvalue'],
            'tail_diff_p5': test_results['tail_differences']['p5_diff'],
            'tail_diff_p1': test_results['tail_differences']['p1_diff']
        })
    
    test_results_df = pd.DataFrame(test_results_list)
    test_results_path = output_dir / "statistical_tests.csv"
    test_results_df.to_csv(test_results_path, index=False)
    print(f"   Saved statistical test results to {test_results_path}")
    
    # Generate essential plots
    print("\n6. Generating plots...")
    
    # Combined extremeness vs ERP scatter plots (stacked vertically)
    print("\n   Creating combined extremeness vs ERP plot...")
    plot_extremeness_vs_erp_combined(all_results, output_dir)
    
    # Combined histogram plots (stacked vertically)
    print("\n   Creating combined histogram plot...")
    plot_extremeness_histogram_combined(all_results, output_dir)
    
    # Combined ERP boxplot for all models
    print("\n   Creating combined ERP boxplot...")
    plot_erp_by_state_all_models(all_results, output_dir)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - ERP statistics CSV files (one per model)")
    print("  - ERP statistics by percentiles CSV files (one per model)")
    print("  - Statistical test results CSV (all models)")
    print("  - Combined extremeness vs ERP plot (all models, stacked)")
    print("  - Combined extremeness histogram plot (all models, stacked)")
    print("  - Combined ERP boxplot (all models)")


if __name__ == "__main__":
    main()

