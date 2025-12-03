# Systematic HMM Regime Detection

This module implements **systematic testing** of Gaussian Hidden Markov Models (HMM) for detecting macroeconomic regimes using different combinations of macro variables.

## Overview

The code systematically tests:
- **Variable combinations**: All 4 variables + all 6 combinations of 2 variables
- **Regime numbers**: K = 2, 3, 4, 5, 6
- **Comparison criteria**: AIC and BIC

## Variable Combinations Tested

### All 4 Variables
- `growth_factor`, `inflation_factor`, `monetary_policy_factor`, `market_volatility_factor`

### All 2-Variable Combinations (6 total)
1. `growth_factor` + `inflation_factor`
2. `growth_factor` + `monetary_policy_factor`
3. `growth_factor` + `market_volatility_factor`
4. `inflation_factor` + `monetary_policy_factor`
5. `inflation_factor` + `market_volatility_factor`
6. `monetary_policy_factor` + `market_volatility_factor`

## Quick Start

### Run Systematic Testing

```bash
python main.py
```

This will:
1. Load macro data from `final_macro.csv`
2. Test all 7 variable combinations (1 with 4 vars + 6 with 2 vars)
3. For each combination, test K = 2, 3, 4, 5, 6
4. Compare all models using AIC and BIC
5. Save results to `results_systematic/`

## Output Files

### `results_systematic/all_model_results.csv`
Complete results for all models tested:
- Combination name
- Variables used
- K (number of regimes)
- AIC, BIC, log-likelihood
- Number of parameters
- Flags for best AIC/BIC within each combination

### `results_systematic/model_comparison_summary.csv`
Summary table showing best K for each variable combination:
- Best K by AIC
- Best K by BIC
- Corresponding AIC/BIC values

### `results_systematic/best_models.csv`
Overall best models:
- Best model by AIC (across all combinations and K)
- Best model by BIC (across all combinations and K)

## Model Selection

Models are compared using:
- **AIC (Akaike Information Criterion)**: Penalizes complexity less than BIC
- **BIC (Bayesian Information Criterion)**: Stronger penalty for complexity, prefers simpler models

Lower values indicate better models.

## File Structure

```
HMM_regimes/
├── main.py              # Systematic testing script
├── hmm_model.py         # HMM model class (flexible variable support)
├── plotting.py          # Visualization functions
├── results.py           # Statistical tests and results processing
├── README.md            # This file
│
└── results_systematic/  # Output directory
    ├── all_model_results.csv
    ├── model_comparison_summary.csv
    └── best_models.csv
```

## Old Code

Previous model-specific scripts and results have been moved to:
- `s12_regimeness_old/regimes/HMM_regimes/`

## Requirements

- pandas, numpy, scikit-learn, hmmlearn, scipy
