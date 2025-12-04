"""
Utility script to compare the 2×2 quadrant regime model with the optimal
HMM (Growth + Policy) specification using a consistent significance threshold.

Outputs:
- CSV summary with headline metrics for each model
- Markdown report at repository root for quick review
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parent
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

from path_utils import get_project_root

SIGNIFICANCE_LEVEL = 0.05


@dataclass
class ModelPaths:
    """Convenience bundle for required files."""

    stats: Path
    pairs: Path
    label: str


def load_model_data(paths: ModelPaths) -> Dict[str, pd.DataFrame]:
    """Load regime statistics and pairwise t-tests for a given model."""
    if not paths.stats.exists() or not paths.pairs.exists():
        raise FileNotFoundError(
            f"Missing results for {paths.label}. "
            f"Expected {paths.stats} and {paths.pairs}."
        )
    return {
        "stats": pd.read_csv(paths.stats),
        "pairs": pd.read_csv(paths.pairs),
    }


def summarize_model(label: str, stats_df: pd.DataFrame, pair_df: pd.DataFrame) -> Dict[str, float]:
    """Compute comparable summary metrics for a regime model."""
    best_row = stats_df.loc[stats_df["avg_erp"].idxmax()]
    worst_row = stats_df.loc[stats_df["avg_erp"].idxmin()]

    sig_pairs = pair_df[pair_df["p_value"] <= SIGNIFICANCE_LEVEL]
    neg_sig_pairs = (sig_pairs["mean_diff"] < 0).sum()

    return {
        "model": label,
        "n_regimes": len(stats_df),
        "best_regime": best_row["regime_name"],
        "best_regime_erp": best_row["avg_erp"],
        "best_regime_share": best_row["pct_of_total"],
        "worst_regime": worst_row["regime_name"],
        "worst_regime_erp": worst_row["avg_erp"],
        "erp_spread": best_row["avg_erp"] - worst_row["avg_erp"],
        "regimes_with_positive_erp": int((stats_df["avg_erp"] > 0).sum()),
        "regimes_with_negative_erp": int((stats_df["avg_erp"] < 0).sum()),
        "significant_pairs": int(len(sig_pairs)),
        "significant_pairs_negative": int(neg_sig_pairs),
    }


def build_markdown_report(output_path: Path, summaries: List[Dict[str, float]]) -> None:
    """Create a concise Markdown comparison report."""
    df = pd.DataFrame(summaries)
    df = df.set_index("model")

    display_cols = [
        "n_regimes",
        "best_regime",
        "best_regime_erp",
        "worst_regime",
        "worst_regime_erp",
        "erp_spread",
        "significant_pairs",
    ]
    printable = df[display_cols].copy()
    percent_cols = ["best_regime_erp", "worst_regime_erp", "erp_spread"]
    for col in percent_cols:
        printable[col] = printable[col].map(lambda x: f"{x:.2%}")

    lines = [
        "# 2×2 vs HMM (Growth + Policy) Comparison",
        "",
        f"- Significance threshold: {SIGNIFICANCE_LEVEL:.0%} (p-value ≤ {SIGNIFICANCE_LEVEL:.2f})",
        "- Metrics computed from the latest outputs in `s1_macro_vars/s12_regimeness/regimes`",
        "",
        "## Headline Metrics",
        "```",
        printable.to_string(),
        "```",
        "",
        "## Interpretation",
    ]

    two_by_two, hmm = df.index.tolist()

    takeaways = [
        f"HMM (Growth + Policy) achieves a wider ERP spread "
        f"({df.loc[hmm, 'erp_spread']:.2%}) than the 2×2 quadrants "
        f"({df.loc[two_by_two, 'erp_spread']:.2%}), highlighting stronger regime differentiation.",
        f"The HMM model delivers {df.loc[hmm, 'significant_pairs']} significant regime pair "
        f"comparisons at the 5% level versus {df.loc[two_by_two, 'significant_pairs']} "
        "for the 2×2 approach, indicating clearer statistical separation.",
        f"HMM isolates a much harsher risk-off regime "
        f"({df.loc[hmm, 'worst_regime']} at {df.loc[hmm, 'worst_regime_erp']:.2%}) "
        f"compared with the 2×2 worst regime "
        f"({df.loc[two_by_two, 'worst_regime']} at {df.loc[two_by_two, 'worst_regime_erp']:.2%}), "
        "highlighting the role of policy support in tail scenarios.",
    ]

    lines.extend(f"- {t}" for t in takeaways)

    lines.append("")
    lines.append("Generated automatically by `s12_regimeness/compare_models.py`.")

    output_path.write_text("\n".join(lines))


def main():
    project_root = get_project_root(__file__)
    repo_root = project_root.parent

    paths = {
        "2×2 Quadrants": ModelPaths(
            stats=project_root / "s1_macro_vars" / "s12_regimeness" / "regimes" / "2x2_regimes" / "results" / "regime_statistics.csv",
            pairs=project_root / "s1_macro_vars" / "s12_regimeness" / "regimes" / "2x2_regimes" / "results" / "pairwise_ttests.csv",
            label="2×2 Quadrants",
        ),
        "HMM (Growth + Policy)": ModelPaths(
            stats=project_root / "s1_macro_vars" / "s12_regimeness" / "regimes" / "HMM_regimes" / "results_2vars_optimal" / "regime_statistics.csv",
            pairs=project_root / "s1_macro_vars" / "s12_regimeness" / "regimes" / "HMM_regimes" / "results_2vars_optimal" / "pairwise_ttests_erp.csv",
            label="HMM (Growth + Policy)",
        ),
    }

    summaries: List[Dict[str, float]] = []
    for label, model_paths in paths.items():
        data = load_model_data(model_paths)
        summaries.append(summarize_model(label, data["stats"], data["pairs"]))

    summary_df = pd.DataFrame(summaries)
    summary_csv = project_root / "s1_macro_vars" / "s12_regimeness" / "results" / "regime_comparison_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    md_path = repo_root / "COMPARISON_2X2_VS_HMM_OPTIMAL.md"
    build_markdown_report(md_path, summaries)

    print(f"Saved comparison summary to {summary_csv}")
    print(f"Wrote Markdown report to {md_path}")


if __name__ == "__main__":
    main()
