# ERP Macro Variable Relevance Analysis

This module analyzes which macro variables are most relevant for Equity Risk Premium (ERP = SP500 - 3m yield) conditional on different market regimes identified by the HMM regime detection system.

## Overview

The analysis identifies the most important macro variables that drive Equity Risk Premium in each of the 4 economic regimes:
1. **High Growth / High Inflation** - Economic expansion with rising prices
2. **High Growth / Low Inflation** - Goldilocks economy (ideal conditions)
3. **Low Growth / High Inflation** - Stagflation
4. **Low Growth / Low Inflation** - Recession/deflation

## Methodology

The analysis uses three complementary approaches:

1. **Correlation Analysis**: Pearson correlations between ERP and each macro variable within each regime
2. **Regression Analysis**: Linear regressions of ERP on individual macro variables, measuring R² and statistical significance
3. **Feature Importance**: Random Forest regressor to identify non-linear relationships and variable interactions

These metrics are combined into a composite **Relevance Score** that ranks macro variables by their importance for ERP in each regime.

## Data Requirements

### Input Files:
- `regime_assignments.csv` - Regime assignments from HMM model (in `test_regimes_matrix/results/`)
- `sp500_processed.csv` - SP500 index data (in `data/macro_processed/other/`)
- `3m_yield_processed.csv` - 3-month Treasury yield (in `data/macro_processed/other/`)
- Macro variables from `data/macro_processed/`:
  - Economic Growth: GDP, unemployment, industrial production, retail sales, etc.
  - Inflation: CPI, PCE, PPI
  - Market Volatility: VIX, spreads, volatility indices
  - Monetary Policy: Fed funds rate, discount rate, money supply
  - Other: Treasury yields, credit spreads

### Output Files:
All results are saved to `results/` directory:
- `erp_macro_relevance_summary.csv` - Complete summary with relevance scores
- `erp_correlations_by_regime.csv` - Correlation analysis results
- `erp_regressions_by_regime.csv` - Regression analysis results
- `erp_feature_importance_by_regime.csv` - Random Forest feature importance
- `relevance_heatmap.png` - Heatmap of top variables by regime
- `correlation_heatmap.png` - Correlation heatmap by regime
- `top_variables_by_regime.png` - Bar charts of top variables per regime

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pathlib import Path
from erp_macro_relevance import ERPMacroRelevanceAnalyzer

# Set up paths
project_root = Path(__file__).parent.parent
regime_assignments_path = project_root / 'test_regimes_matrix' / 'results' / 'regime_assignments.csv'
macro_data_dir = project_root / 'data' / 'macro_processed'
sp500_path = project_root / 'data' / 'macro_processed' / 'other' / 'sp500_processed.csv'
yield_3m_path = project_root / 'data' / 'macro_processed' / 'other' / '3m_yield_processed.csv'
output_dir = Path(__file__).parent / 'results'

# Initialize analyzer
analyzer = ERPMacroRelevanceAnalyzer(
    regime_assignments_path=regime_assignments_path,
    macro_data_dir=macro_data_dir,
    sp500_path=sp500_path,
    yield_3m_path=yield_3m_path,
    output_dir=output_dir
)

# Run full analysis
analyzer.run_full_analysis()
```

### Command Line

```bash
python erp_macro_relevance.py
```

## Key Metrics

### Relevance Score
Composite score combining:
- **40%** Normalized absolute correlation
- **30%** Normalized regression R²
- **30%** Normalized Random Forest importance

Higher scores indicate stronger relevance for ERP prediction in that regime.

### Statistical Significance
- Correlation p-values: Test if correlation is significantly different from zero
- Regression p-values: Test if regression coefficient is significant
- Both use standard statistical tests (t-tests)

## Interpretation

### High Relevance Variables
Variables with high relevance scores in a regime are the most important drivers of ERP in that economic environment. These can be used for:
- **Forecasting**: Predict ERP changes based on macro conditions
- **Portfolio Allocation**: Adjust allocations when these variables change
- **Risk Management**: Monitor these variables for regime-specific risks

### Regime-Specific Insights
Different regimes may have different key drivers:
- **High Growth / Low Inflation**: Typically favorable for equities, may be driven by growth indicators
- **Low Growth / High Inflation**: Challenging environment, may be driven by inflation and policy variables
- **High Growth / High Inflation**: May be driven by policy responses and inflation expectations
- **Low Growth / Low Inflation**: May be driven by monetary policy and deflation concerns

## Example Output

```
ANALYSIS COMPLETE
================================================================================

Results saved to: results/

Key findings:

  Regime 0: Low Growth / High Inflation
    vix                          (score: 0.856)
    bofa_highyield_spread        (score: 0.743)
    fedfunds                     (score: 0.692)
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

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## References

- Equity Risk Premium literature
- Regime-dependent asset pricing models
- Macro-finance relationships
- Feature importance methods in machine learning

