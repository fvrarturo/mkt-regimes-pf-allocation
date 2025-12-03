"""
Main script for extremeness models analysis.

Now compares macro-only vs macro+sentiment feature sets across Isolation Forest
and PCA-distance extremeness definitions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow relative imports for helper modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import load_data, prepare_data, get_feature_columns
from isolation_forest import run_isolation_forest_analysis
from pca_distance import run_pca_distance_analysis
from stats import (
    compute_erp_statistics,
    compute_erp_statistics_by_percentiles,
    test_erp_differences,
)
from plots import (
    plot_extremeness_histogram_combined,
    plot_extremeness_vs_erp_combined,
    plot_erp_by_state_all_models,
)


def slugify(text: str) -> str:
    """Filesystem-safe slug."""
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Extremeness Models Analysis")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    erp_df, macro_df, sentiment_df = load_data()
    print(f"   ERP data: {len(erp_df)} observations")
    print(f"   Macro data: {len(macro_df)} observations")

    # Prepare output directory
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_versions = [
        {"name": "Macro Only", "include_sentiment": False},
        {"name": "Macro + Sentiment", "include_sentiment": True},
    ]

    model_runners = [
        (
            "Isolation Forest",
            lambda data, cols: run_isolation_forest_analysis(
                data, cols, contamination=0.1, threshold_percentile=90
            ),
        ),
        (
            "PCA Distance",
            lambda data, cols: run_pca_distance_analysis(
                data,
                cols,
                variance_threshold=0.85,
                distance_method="euclidean",
                threshold_percentile=90,
            ),
        ),
    ]

    all_results = {}
    summary_rows = []
    test_rows = []

    # Iterate through feature sets
    for version in data_versions:
        version_name = version["name"]
        include_sentiment = version["include_sentiment"]
        version_slug = slugify(version_name)

        if include_sentiment and sentiment_df is None:
            print(f"\n⚠️  Skipping '{version_name}' (sentiment data not available).")
            continue

        print(f"\n2. Preparing data ({version_name})...")
        df = prepare_data(
            erp_df,
            macro_df,
            sentiment_df,
            include_sentiment=include_sentiment,
        )
        if df.empty:
            print(f"   Warning: No overlapping observations for {version_name}.")
            continue

        print(f"   Merged data: {len(df)} observations")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")

        feature_cols = get_feature_columns(include_sentiment=include_sentiment)

        for model_name, runner in model_runners:
            print(f"\n3. Running {model_name} ({version_name})...")
            results = runner(df, feature_cols)
            label = f"{version_name} - {model_name}"
            results.update(
                {
                    "version": version_name,
                    "model_name": model_name,
                    "feature_set": version_slug,
                }
            )
            print(
                f"   Extreme states: {np.sum(results['is_extreme'])} "
                f"({np.mean(results['is_extreme']) * 100:.1f}%)"
            )
            all_results[label] = results

    if not all_results:
        print("No models executed. Exiting.")
        return

    # Statistics per configuration
    print("\n5. Computing statistics...")
    for model_label, results in all_results.items():
        safe_name = slugify(model_label)
        print(f"\n   {model_label}:")
        stats_df = compute_erp_statistics(results["results_df"])
        test_results = test_erp_differences(results["results_df"])

        print(f"      Normal ERP mean: {stats_df.loc[0, 'mean']:.4f}")
        print(f"      Extreme ERP mean: {stats_df.loc[1, 'mean']:.4f}")
        print(
            f"      Mean difference: {test_results['t_test']['mean_diff']:.4f} "
            f"(p={test_results['t_test']['pvalue']:.4f})"
        )

        stats_df.insert(0, "model_label", model_label)
        stats_path = output_dir / f"{safe_name}_erp_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"   Saved ERP statistics to {stats_path}")

        percentile_stats = compute_erp_statistics_by_percentiles(results["results_df"])
        percentile_stats.insert(0, "model_label", model_label)
        percentile_path = output_dir / f"{safe_name}_erp_statistics_by_percentiles.csv"
        percentile_stats.to_csv(percentile_path, index=False)
        print(f"   Saved percentile statistics to {percentile_path}")

        summary_rows.append(
            {
                "model_label": model_label,
                "version": results["version"],
                "model_name": results["model_name"],
                "normal_mean": stats_df.loc[0, "mean"],
                "extreme_mean": stats_df.loc[1, "mean"],
                "mean_difference": test_results["t_test"]["mean_diff"],
                "t_statistic": test_results["t_test"]["statistic"],
                "t_pvalue": test_results["t_test"]["pvalue"],
            }
        )

        test_rows.append(
            {
                "model_label": model_label,
                "version": results["version"],
                "model_name": results["model_name"],
                "t_statistic": test_results["t_test"]["statistic"],
                "t_pvalue": test_results["t_test"]["pvalue"],
                "normal_mean": test_results["t_test"]["normal_mean"],
                "extreme_mean": test_results["t_test"]["extreme_mean"],
                "mean_difference": test_results["t_test"]["mean_diff"],
                "ks_statistic": test_results["ks_test"]["statistic"],
                "ks_pvalue": test_results["ks_test"]["pvalue"],
                "mannwhitney_statistic": test_results["mannwhitney_test"]["statistic"],
                "mannwhitney_pvalue": test_results["mannwhitney_test"]["pvalue"],
                "tail_diff_p5": test_results["tail_differences"]["p5_diff"],
                "tail_diff_p1": test_results["tail_differences"]["p1_diff"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "extremeness_model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Saved summary table to {summary_path}")

    test_df = pd.DataFrame(test_rows)
    test_path = output_dir / "statistical_tests.csv"
    test_df.to_csv(test_path, index=False)
    print(f"   Saved statistical test results to {test_path}")

    # Plots
    print("\n6. Generating plots...")
    print("\n   Creating combined extremeness vs ERP plot...")
    plot_extremeness_vs_erp_combined(all_results, output_dir)

    print("\n   Creating combined histogram plot...")
    plot_extremeness_histogram_combined(all_results, output_dir)

    print("\n   Creating combined ERP boxplot...")
    plot_erp_by_state_all_models(all_results, output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - ERP statistics CSV files (per version + model)")
    print("  - ERP statistics by percentiles CSV files")
    print("  - extremeness_model_summary.csv")
    print("  - statistical_tests.csv")
    print("  - Combined extremeness vs ERP plot")
    print("  - Combined extremeness histogram plot")
    print("  - Combined ERP boxplot")


if __name__ == "__main__":
    main()
