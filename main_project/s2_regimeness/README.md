# Regime-Conditional Regression Analysis

This folder contains the Section 1 workflow for discovering macro regimes and running conditional ERP regressions that feed the forecasting/trading pipeline described in `codex_instructions.md`.

## 📁 Folder Organization (current)

```
s12_regimeness/
├── regimes/                    # Regime definitions + statistics
│   ├── 2x2_regimes/            # Growth × Inflation baseline
│   └── HMM_regimes/            # Gaussian HMMs (Growth+Policy optimal)
├── compare_models.py           # Generates 2×2 vs HMM comparison tables
├── results/                    # CSV summary from compare_models.py
└── regressions_expanding_window/
    ├── regimes/                # Expanding-window regime detection
    ├── regression/             # Probability-weighted regressions
    └── results/                # Look-ahead-free regression outputs
```

Legacy `results_full_sample/` assets were removed to keep the focus on out-of-sample, pipeline-ready artifacts.

## 🔑 What matters for the main goal

1. **Select the informative regime lens**  
   - Run `python s12_regimeness/compare_models.py`.  
   - Output: `results/regime_comparison_summary.csv` and the repo-level `COMPARISON_2X2_VS_HMM_OPTIMAL.md`.  
   - Interpretation: Growth + Policy HMM delivers the widest ERP spread (4.15 ppts) and twice as many 5 %‑significant regime differences as the 2×2 baseline. This becomes the default regime signal for forecasting and strategies.

2. **Inspect regime details**  
   - 2×2 documentation: `regimes/2x2_regimes/README.md` + `results/RESULTS_SUMMARY.md` (serves as the intuitive baseline).  
   - HMM documentation: `regimes/HMM_regimes/README.md`, `SUMMARY.md`, `ECONOMIC_ANALYSIS.md`.  
   - For quantified characteristics use each folder’s `regime_statistics.csv` / `pairwise_ttests_erp.csv`.

3. **Run probability-weighted regressions without look-ahead bias**  
   - Entry point: `regressions_expanding_window/main.py`.  
   - Produces expanding-window regime probabilities and conditional regressions (horizon-by-regime coefficients and their significance) aligned with the requirements in `codex_instructions.md`.

## 📊 Current Findings (5 % threshold)

- **Best regime model**: Growth + Policy HMM (see comparison summary).  
- **Baseline intuition**: 2×2 Growth × Inflation regimes still provide a pedagogy-friendly interpretation layer; Stagflation remains the only negative-ERP quadrant.  
- **Conditional regressions**: The expanding-window outputs highlight Unemployment, Discount Rate, Financial Conditions, and Industrial Production as repeatedly significant predictors in risk-off regimes—those are the features prioritized when moving into forecasting (Step 2).

## Quick Start

1. `python s12_regimeness/compare_models.py`
2. `python s12_regimeness/regimes/HMM_regimes/run_growth_policy_model.py`
3. `python s12_regimeness/regressions_expanding_window/main.py`

Use the generated CSVs/plots directly when building forecasting features (Step 2) and strategy signals (Step 3). Everything else has been trimmed to keep the Section 1 documentation aligned with the project’s main goal.
