# Section 1: Macro Variables Analysis

This directory contains the complete analysis pipeline to identify which macro variables have the most predictability on Equity Risk Premium (ERP).

## Quick Start

Run the master orchestration script to execute the complete analysis:

```bash
python s1_macro_vars/main.py
```

This single command runs:
1. Full-sample regression analysis (baseline)
2. Regime detection (2x2 Growth×Inflation and HMM Growth+Policy)
3. Regime model comparison
4. Expanding-window conditional regressions (no look-ahead bias)
5. Extremeness analysis (Isolation Forest and PCA Distance)

## Directory Structure

```
s1_macro_vars/
├── main.py                    # ⭐ Master orchestration script (run this)
├── path_utils.py              # Path utilities
│
├── s11_full_sample/           # Full-sample regression analysis
│   ├── main.py
│   ├── regression.py
│   ├── plotting.py
│   └── results/               # Output: coefficient tables, R², variable importance
│
├── s12_regimeness/           # Regime-dependent conditional regressions
│   ├── compare_models.py     # Compare 2x2 vs HMM models
│   ├── regimes/
│   │   ├── 2x2_regimes/      # Growth × Inflation quadrants
│   │   └── HMM_regimes/       # HMM optimal (Growth + Policy)
│   └── regressions_expanding_window/  # Expanding-window conditional regressions
│       └── results/           # Output: regime-specific coefficients
│
└── s13_extremeness/           # Extremeness analysis
    ├── initial_relevance/     # Initial extremeness analysis
    │   └── main.py
    ├── isolation_forest.py
    ├── pca_distance.py
    └── results/               # Output: extremeness statistics, tests
```

## Outputs Generated

### 1. Full-Sample Analysis (`s11_full_sample/results/`)
- `regression_results_all_horizons.csv` - Coefficients, t-stats, R² for all horizons
- `regression_summary.csv` - Summary statistics
- `variable_importance_ranking.csv` - Variable importance based on |t-stat|
- `variable_importance_ranking.png` - Visualization
- `coefficient_comparison.png` - Coefficient comparison across horizons
- `r_squared_by_horizon.png` - R² by forecast horizon

### 2. Regime Analysis (`s12_regimeness/`)

**Regime Detection:**
- `regimes/2x2_regimes/results/regime_statistics.csv` - 2x2 regime statistics
- `regimes/HMM_regimes/results_2vars_optimal/regime_statistics.csv` - HMM optimal statistics

**Model Comparison:**
- `results/regime_comparison_summary.csv` - Comparison of 2x2 vs HMM
- `../COMPARISON_2X2_VS_HMM_OPTIMAL.md` - Markdown comparison report

**Conditional Regressions:**
- `regressions_expanding_window/results/.../regression_conditional/hmm_optimal/` - HMM conditional coefficients
- `regressions_expanding_window/results/.../regression_conditional/2x2/` - 2x2 conditional coefficients
- Includes coefficient tables, significance tests, and visualizations for each regime

### 3. Extremeness Analysis (`s13_extremeness/results/`)
- `statistical_tests.csv` - T-tests, KS-tests, Mann-Whitney tests
- `extremeness_model_summary.csv` - Summary of all extremeness models
- `*_erp_statistics.csv` - ERP statistics for normal vs extreme states
- `*_erp_statistics_by_percentiles.csv` - Statistics across percentile thresholds
- `extremeness_vs_erp_combined.png` - Scatter plots
- `extremeness_histogram_combined.png` - Distribution plots
- `erp_boxplot_all_models.png` - ERP distributions by extremeness state

## Key Findings

### Best Regime Model
- **HMM Growth + Policy** is optimal (4.15% ERP spread, 4 significant regime pairs)
- See `COMPARISON_2X2_VS_HMM_OPTIMAL.md` for details

### Best Extremeness Model
- **Macro-only features** (not macro+sentiment) deliver significant results
- Isolation Forest and PCA Distance both show 1.5-2x wider ERP volatility in extreme states
- See `s13_extremeness/initial_relevance/SUMMARY.md` for details

### Most Predictive Variables
- Expanding-window conditional regressions identify:
  - Unemployment
  - Discount Rate
  - Financial Conditions
  - Industrial Production
- These are repeatedly significant in risk-off regimes

## Individual Component Execution

If you need to run components individually:

```bash
# Full-sample analysis only
python s11_full_sample/main.py

# Regime detection
python s12_regimeness/regimes/2x2_regimes/main.py
python s12_regimeness/regimes/HMM_regimes/run_growth_policy_model.py

# Compare regime models
python s12_regimeness/compare_models.py

# Conditional regressions
python s12_regimeness/regressions_expanding_window/main.py

# Extremeness analysis
python s13_extremeness/initial_relevance/main.py
```

## Requirements

All scripts require:
- Python 3.7+
- Standard data science libraries (pandas, numpy, scipy, sklearn, matplotlib, seaborn)
- hmmlearn (for HMM models)
- Data files in `main_project/data/`:
  - `macro_processed/equity_risk_pr.csv`
  - `macro_final/final_macro.csv`
  - `macro_processed/3m_yield_processed.csv`
  - `macro_processed/sp500_processed.csv`

## Next Steps

After running the complete analysis:
1. Review `regime_comparison_summary.csv` to confirm HMM Growth+Policy is optimal
2. Review `statistical_tests.csv` to confirm macro-only extremeness is best
3. Use expanding-window conditional regression results for forecasting (Part 2)
4. Use regime probabilities and extremeness scores for trading strategies (Part 3)

## Alignment with Codex Instructions

This analysis directly addresses Part 1 of `codex_instructions.md`:
> "Analysis of which macro variables have the most predictability on Equity Risk Premium"
> 
> "We're looking for the regime/extremeness definition that shows the clearest patterns 
> in terms of significance of linear regression coef for macro variables"

The outputs provide:
- ✅ Baseline variable importance (full-sample)
- ✅ Regime-specific coefficients (conditional regressions)
- ✅ Extremeness-based patterns (volatility and tail risk)
- ✅ Model comparison (2x2 vs HMM, macro-only vs macro+sentiment)

All results feed into Part 2 (forecasting) and Part 3 (trading strategies).

