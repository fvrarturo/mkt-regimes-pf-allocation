# Conditional Regression Analysis for All HMM Regimes

This module performs conditional regression analysis to identify which macro variables predict ERP (Equity Risk Premium) in different economic regimes defined by HMM models.

## Overview

The analysis:
1. **Loads all macro variables** from `macro_processed_full` directories:
   - Economic Growth (`ec_growth/`)
   - Inflation (`inflation/`)
   - Market Volatility (`mkt_vol/`)
   - Monetary Policy (`mon_policy/`)

2. **Loads all HMM regime assignments** from systematic testing:
   - All variable combinations (all 4 vars + all 6 pairs of 2 vars)
   - All K values (2-10 regimes)

3. **Runs conditional regressions** for each regime:
   - Regresses ERP on all macro variables
   - Extracts coefficients, t-statistics, p-values
   - Calculates R² and RMSE

4. **Creates visualizations**:
   - Coefficient heatmaps by regime
   - Significance heatmaps (p-values)
   - Top predictors by regime
   - Overall variable importance

## Files

### Core Scripts

- **`main.py`**: Main orchestration script
- **`conditional_regression_all_regimes.py`**: Core regression analysis logic
- **`visualize_regression_results.py`**: Visualization functions

### Data Preparation

- **`../../data/macro_processed_full/aggregate_to_monthly.py`**: Aggregates high-frequency market volatility data (VIX, NFCI, yield spread) to monthly frequency

## Usage

### Step 1: Aggregate Market Volatility Data (if needed)

```bash
cd main_project/data/macro_processed_full
python aggregate_to_monthly.py
```

This creates monthly versions of:
- `vix_processed_monthly.csv`
- `nat_fin_condition_indx_processed_monthly.csv`
- `10y_2y_spread_processed_monthly.csv`

### Step 2: Run Conditional Regression Analysis

```bash
cd main_project/s1_macro_vars/s12_regimeness/regressions
python main.py
```

## Output

Results are saved to `regressions/results/`:

### CSV Files

- **`conditional_regression_results_all.csv`**: Complete regression results for all regimes
  - Columns: combination, K, regime, variable, coefficient, t_statistic, p_value, n_observations, r_squared, rmse

- **`significant_variables_summary.csv`**: Summary of statistically significant variables (p < 0.05)

### Visualization Files

For each combination and K value:
- **`coefficient_heatmap_{combination}_K{k}.png`**: Coefficient heatmap (significant only)
- **`significance_heatmap_{combination}_K{k}.png`**: P-value heatmap
- **`top_predictors_{combination}_K{k}.png`**: Top 10 predictors by regime

Overall:
- **`variable_importance_overall.png`**: Variable importance across all regimes

## Methodology

### Regime Detection

Regimes are detected using HMM models fitted on:
- Variable combinations: All 4 variables + all 6 pairs of 2 variables
- K values: 2, 3, 4, 5, 6, 7, 8, 9, 10

### Conditional Regressions

For each regime specification (combination, K):
1. Fit HMM model on macro factors
2. Assign regime to each time period
3. For each regime:
   - Filter observations belonging to that regime
   - Run OLS regression: ERP ~ all macro variables
   - Extract coefficients, t-stats, p-values
   - Calculate R² and RMSE

### Statistical Significance

Variables are considered significant if p-value < 0.05.

## Interpretation

- **Coefficient heatmaps**: Show which variables predict ERP in each regime
- **Significance heatmaps**: Highlight statistically significant relationships
- **Top predictors**: Identify the most important variables for each regime
- **Variable importance**: Overall ranking of macro variables across all regimes

## Notes

- Minimum 10 observations required per regime to run regression
- Missing values are handled by dropping rows with any NaN
- Features are standardized before regression
- Results are aggregated across time (no expanding window - uses full sample for regime detection)

