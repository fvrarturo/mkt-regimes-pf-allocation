# 2x2 Growth × Inflation Regime Classification

This module implements a simple 2x2 regime classification system based on Growth and Inflation factors.

## Structure

```
2x2_regimes/
├── main.py                 # Main analysis script
├── regime_definitions.py   # Regime classification logic
├── plotting.py             # Plotting functions
├── results/               # Output directory (generated)
│   ├── regime_assignments.csv
│   ├── regime_statistics.csv
│   ├── pairwise_ttests.csv
│   ├── anova_results.txt
│   ├── regime_scatter_plot.png
│   ├── erp_boxplots_by_regime.png
│   └── regime_time_series.png
└── README.md              # This file
```

## Regimes

The system classifies periods into 4 regimes based on Growth and Inflation thresholds:

1. **Goldilocks (High G / Low I)** - High growth with low inflation
2. **Overheating (High G / High I)** - High growth with high inflation
3. **Stagflation (Low G / High I)** - Low growth with high inflation
4. **Slowdown / Disinflation (Low G / Low I)** - Low growth with low inflation

## Threshold Methods

Thresholds can be determined using:
- `median`: Use median value (default)
- `zero`: Use zero (for standardized data)
- `mean`: Use mean value

## Usage

```python
from pathlib import Path
from main import TwoByTwoRegimeAnalyzer

# Set up paths
data_dir = Path('path/to/data')
output_dir = Path('results')

# Initialize analyzer
analyzer = TwoByTwoRegimeAnalyzer(
    data_dir=data_dir,
    output_dir=output_dir,
    threshold_method='median'
)

# Run full analysis
analyzer.run_full_analysis()
```

Or run from command line:

```bash
python main.py
```

## Outputs

### CSV Files

1. **regime_assignments.csv**: Date, regime ID, regime name, growth, inflation, ERP
2. **regime_statistics.csv**: Summary statistics for each regime:
   - Macro averages (growth, inflation, policy, volatility)
   - ERP statistics (mean, std, volatility, skew, kurtosis)
   - Tail statistics (min, max, 5th/95th percentiles)
3. **pairwise_ttests.csv**: T-test results comparing ERP means across regimes
4. **anova_results.txt**: ANOVA F-test results for ERP variation across regimes

### Plots

1. **regime_scatter_plot.png**: Growth vs Inflation scatter plot colored by regime
2. **erp_boxplots_by_regime.png**: Boxplots of ERP and ERP volatility by regime
3. **regime_time_series.png**: Time series of regime assignments, factors, and ERP

## Statistical Tests

The analysis includes:

1. **Pairwise t-tests**: Compare ERP means between each pair of regimes
2. **ANOVA F-test**: Test for overall differences in ERP means across all regimes

## Data Requirements

The script expects the following data structure:

```
data/
├── macro_final/
│   └── final_macro.csv          # Must contain: date, growth_factor, inflation_factor, 
│                                #              monetary_policy_factor, market_volatility_factor
└── macro_processed/
    └── other/
        ├── sp500_processed.csv  # Must contain: date, pct_change_mom (or value for calculation)
        └── 3m_yield_processed.csv # Must contain: date, value (annual yield %)
```

## Notes

- ERP is calculated as: `SP500 Monthly Return - 3m Yield Monthly Rate`
- Monthly yield rate = Annual yield / 100 / 12
- Regime classification uses growth_factor and inflation_factor from final_macro.csv
- All dates are aligned to monthly frequency

