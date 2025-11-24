# Folder Structure

## Overview

This folder contains the analysis of predictive significance of macro variables for VIX, conditional on market regimes.

## Directory Structure

```
test_cond_significance_VIX/
│
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
│
├── Scripts (Main Analysis)
│   ├── vix_macro_relevance.py        # Main analyzer class
│   ├── regime_conditional_predictors.py  # Lagged predictor analysis
│   └── summarize_predictive_significance.py  # Summary script
│
├── docs/                              # Documentation
│   ├── BIAS_ACCOUNTING_STATUS.md     # Look-ahead bias status
│   ├── HOW_PREDICTIVE_POWER_IS_TESTED.md
│   ├── LOOKAHEAD_BIAS_ISSUES.md
│   └── PVALUES_AND_COMPOSITE_SCORE_EXPLAINED.md
│
└── results/                           # All analysis results
    ├── tables/                       # CSV data tables
    │   ├── vix_macro_relevance_summary.csv
    │   ├── vix_correlations_by_regime.csv
    │   ├── vix_regressions_by_regime.csv
    │   ├── vix_feature_importance_by_regime.csv
    │   ├── predictive_significance_summary.csv
    │   └── category_summary_by_regime.csv
    │
    ├── plots/                        # Visualizations (PNG)
    │   ├── relevance_heatmap.png
    │   ├── correlation_heatmap.png
    │   └── top_variables_by_regime.png
    │
    └── detailed_by_regime/          # Detailed results by regime
        ├── lagged_correlations_regime_0.csv
        ├── lagged_correlations_regime_1.csv
        ├── lagged_correlations_regime_2.csv
        ├── lagged_correlations_regime_3.csv
        ├── regressions_regime_0.csv
        ├── regressions_regime_1.csv
        ├── regressions_regime_2.csv
        ├── regressions_regime_3.csv
        └── summary_by_category.csv
```

## Key Files

### Main Scripts

1. **`vix_macro_relevance.py`**
   - Main analyzer class
   - Runs correlation, regression, and Random Forest analysis
   - Creates composite relevance scores
   - Usage: `python vix_macro_relevance.py`

2. **`regime_conditional_predictors.py`**
   - Analyzes lagged predictors (t-1 predicts t)
   - Detailed regime-by-regime analysis
   - Usage: `python regime_conditional_predictors.py`

3. **`summarize_predictive_significance.py`**
   - Creates summary of significant predictors
   - Combines correlation and regression results
   - Usage: `python summarize_predictive_significance.py`

### Results Tables

**Main Summary Tables** (`results/tables/`):
- `vix_macro_relevance_summary.csv` - Complete summary with all metrics
- `predictive_significance_summary.csv` - Significant predictors only
- `category_summary_by_regime.csv` - Summary by variable category

**Detailed Analysis Tables** (`results/tables/`):
- `vix_correlations_by_regime.csv` - Correlation results
- `vix_regressions_by_regime.csv` - Regression results
- `vix_feature_importance_by_regime.csv` - Random Forest importance

**Regime-Specific Tables** (`results/detailed_by_regime/`):
- `lagged_correlations_regime_X.csv` - Detailed correlations by regime
- `regressions_regime_X.csv` - Detailed regressions by regime
- `summary_by_category.csv` - Category-level summary

### Visualizations

**Plots** (`results/plots/`):
- `relevance_heatmap.png` - Heatmap of relevance scores by regime
- `correlation_heatmap.png` - Correlation heatmap by regime
- `top_variables_by_regime.png` - Bar charts of top 10 variables per regime

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run main analysis:**
   ```bash
   python vix_macro_relevance.py
   ```

3. **Run lagged predictor analysis:**
   ```bash
   python regime_conditional_predictors.py
   ```

4. **Generate summary:**
   ```bash
   python summarize_predictive_significance.py
   ```

## Results Location

- **Tables**: `results/tables/`
- **Plots**: `results/plots/`
- **Detailed by regime**: `results/detailed_by_regime/`


