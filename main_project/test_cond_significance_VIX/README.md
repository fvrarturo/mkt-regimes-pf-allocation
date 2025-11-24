# VIX Macro Variable Relevance Analysis

This module analyzes which macro variables are most relevant for VIX (Volatility Index) conditional on different market regimes identified by the HMM regime detection system.

## Overview

The analysis identifies the most important macro variables that drive VIX in each of the 4 economic regimes:
1. **High Growth / High Inflation** - Economic expansion with rising prices
2. **High Growth / Low Inflation** - Goldilocks economy (ideal conditions)
3. **Low Growth / High Inflation** - Stagflation
4. **Low Growth / Low Inflation** - Recession/deflation

## Methodology

The analysis uses three complementary approaches:

1. **Correlation Analysis**: Pearson correlations between VIX and each macro variable within each regime
2. **Regression Analysis**: Linear regressions of VIX on individual macro variables, measuring R² and statistical significance
3. **Feature Importance**: Random Forest regressor to identify non-linear relationships and variable interactions

These metrics are combined into a composite **Relevance Score** that ranks macro variables by their importance for VIX in each regime.

## Data Requirements

### Input Files:
- `regime_assignments.csv` - Regime assignments from HMM model (in `test_4regimes_HMM/results/`)
- `vix_processed.csv` - VIX data (in `data/macro_processed/selection/`)
- Macro variables from `data/macro_processed/`:
  - Economic Growth: GDP, unemployment, industrial production, retail sales, etc.
  - Inflation: CPI, PCE, PPI
  - Market Volatility: Nasdaq volatility, SP500 volatility indices
  - Monetary Policy: Fed funds rate, discount rate, money supply
  - Other: Treasury yields, credit spreads, SP500

### Output Files:
All results are organized in `results/` directory:

**Tables** (`results/tables/`):
- `comprehensive_summary_by_regime.csv` - **Main summary table** with all metrics (p-values, R², correlations, coefficients, t-statistics)
- `top_5_predictors_by_regime.csv` - Top 5 predictors per regime ranked by R²
- `r2_pivot_by_regime.csv` - R² values in pivot format (variables × regimes)
- `pvalue_pivot_by_regime.csv` - P-values in pivot format (variables × regimes)
- `vix_macro_relevance_summary.csv` - Complete summary with relevance scores
- `vix_correlations_by_regime.csv` - Correlation analysis results
- `vix_regressions_by_regime.csv` - Regression analysis results
- `vix_feature_importance_by_regime.csv` - Random Forest feature importance
- `predictive_significance_summary.csv` - Summary of significant predictors
- `category_summary_by_regime.csv` - Summary by variable category

**Plots** (`results/plots/`):
- `relevance_heatmap.png` - Heatmap of top variables by regime
- `correlation_heatmap.png` - Correlation heatmap by regime
- `top_variables_by_regime.png` - Bar charts of top variables per regime

**Detailed Results** (`results/detailed_by_regime/`):
- `lagged_correlations_regime_X.csv` - Detailed lagged correlations by regime
- `regressions_regime_X.csv` - Detailed regression results by regime
- `summary_by_category.csv` - Category-level summary

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pathlib import Path
from vix_macro_relevance import VIXMacroRelevanceAnalyzer

# Set up paths
project_root = Path(__file__).parent.parent
regime_assignments_path = project_root / 'test_4regimes_HMM' / 'results' / 'regime_assignments.csv'
macro_data_dir = project_root / 'data' / 'macro_processed'
vix_path = project_root / 'data' / 'macro_processed' / 'selection' / 'vix_processed.csv'
output_dir = Path(__file__).parent / 'results'

# Initialize analyzer
analyzer = VIXMacroRelevanceAnalyzer(
    regime_assignments_path=regime_assignments_path,
    macro_data_dir=macro_data_dir,
    vix_path=vix_path,
    output_dir=output_dir
)

# Run full analysis
analyzer.run_full_analysis()
```

### Command Line

```bash
python vix_macro_relevance.py
```

### Regime-Conditional Predictors (Lagged Analysis)

To analyze which lagged macro variables predict VIX in each regime:

```bash
python regime_conditional_predictors.py
```

This script uses **lagged** macro variables (t-1) to predict VIX at time t, addressing the question of what predicts volatility in each regime.

### Create Comprehensive Summary Table

To create a detailed summary table with p-values, R², correlations, and coefficients for each regime:

```bash
python create_summary_table.py
```

This creates:
- `comprehensive_summary_by_regime.csv` - Full table with all metrics
- `top_5_predictors_by_regime.csv` - Top 5 predictors per regime
- `r2_pivot_by_regime.csv` - R² pivot table
- `pvalue_pivot_by_regime.csv` - P-value pivot table

## Key Metrics

### Relevance Score
Composite score combining:
- **40%** Normalized absolute correlation
- **30%** Normalized regression R²
- **30%** Normalized Random Forest importance

Higher scores indicate stronger relevance for VIX prediction in that regime.

### Statistical Significance
- Correlation p-values: Test if correlation is significantly different from zero
- Regression p-values: Test if regression coefficient is significant
- Both use standard statistical tests (t-tests)

## Interpretation

### High Relevance Variables
Variables with high relevance scores in a regime are the most important drivers of VIX in that economic environment. These can be used for:
- **Forecasting**: Predict VIX changes based on macro conditions
- **Portfolio Allocation**: Adjust allocations when these variables change
- **Risk Management**: Monitor these variables for regime-specific risks

### Regime-Specific Insights
Different regimes may have different key drivers:
- **High Growth / Low Inflation**: Typically low volatility, may be driven by growth indicators
- **Low Growth / High Inflation**: High volatility, may be driven by inflation and policy variables
- **High Growth / High Inflation**: May be driven by policy responses and inflation expectations
- **Low Growth / Low Inflation**: May be driven by monetary policy and deflation concerns

## Example Output

```
ANALYSIS COMPLETE
================================================================================

Results saved to: results/

Key findings:

  Regime 0: Low Growth / High Inflation
    bofa_highyield_spread        (score: 0.856)
    fedfunds                     (score: 0.743)
    nat_fin_condition_indx       (score: 0.692)
    ...

  Regime 1: High Growth / High Inflation
    gdp                          (score: 0.821)
    PCE_price_index              (score: 0.789)
    ...
```

## Technical Details

### Data Alignment
- All data is aligned to monthly frequency (end of month)
- Macro variables are forward-filled to use most recent available values
- Missing data is handled appropriately for each analysis method

### Statistical Methods
- **Correlation**: Pearson correlation with significance testing
- **Regression**: OLS regression with t-statistics and p-values
- **Random Forest**: 100 trees, max depth 10, minimum 5 samples per split

### Robustness
- Minimum observation requirements (10+ for correlations, 20+ for RF)
- Handles missing data gracefully
- Standardizes features for regression and RF to ensure fair comparison
- Uses regime probabilities for weighted analysis when available

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## Folder Structure

```
test_cond_significance_VIX/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── vix_macro_relevance.py            # Main analyzer class
├── regime_conditional_predictors.py  # Lagged predictor analysis
├── summarize_predictive_significance.py  # Summary script
├── docs/                              # Additional documentation
│   ├── BIAS_ACCOUNTING_STATUS.md     # Look-ahead bias status
│   ├── HOW_PREDICTIVE_POWER_IS_TESTED.md
│   ├── LOOKAHEAD_BIAS_ISSUES.md
│   └── PVALUES_AND_COMPOSITE_SCORE_EXPLAINED.md
└── results/                           # All results
    ├── tables/                       # CSV tables
    ├── plots/                        # PNG visualizations
    └── detailed_by_regime/          # Detailed results by regime
```

## Important Notes

⚠️ **Limitations**: See `docs/BIAS_ACCOUNTING_STATUS.md` for details on look-ahead bias and other limitations.

## References

- VIX and volatility literature
- Regime-dependent asset pricing models
- Macro-finance relationships
- Feature importance methods in machine learning

