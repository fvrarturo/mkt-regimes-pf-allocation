# Regime Probability Handling

## Overview

The HMM regime detection model provides **soft probabilities** for each regime at each time point, not just hard assignments. This document explains how these probabilities are incorporated into the ERP macro variable relevance analysis.

## Regime Probability Data Structure

The `regime_assignments.csv` file contains:
- **Hard assignment**: The most likely regime (`regime` column: 0, 1, 2, or 3)
- **Soft probabilities**: Probability of being in each regime:
  - `prob_R0_Low_Growth___High_Inflation`
  - `prob_R1_High_Growth___High_Inflation`
  - `prob_R2_High_Growth___Low_Inflation`
  - `prob_R3_Low_Growth___Low_Inflation`

These probabilities sum to 1.0 for each observation.

## Why Use Probabilities?

### Problem with Hard Assignments Only

If we only use hard regime assignments:
- Observations near regime boundaries are assigned to one regime only
- Uncertainty about regime classification is ignored
- Observations with mixed probabilities (e.g., 40% R0, 35% R1, 15% R2, 10% R3) are treated the same as observations with high confidence (e.g., 95% R0)

### Benefits of Using Probabilities

1. **Accounts for uncertainty**: Observations with mixed probabilities contribute less to each regime
2. **Smoother transitions**: Regime boundaries are handled more gracefully
3. **More robust statistics**: Weighted methods provide better estimates
4. **Better use of data**: All observations contribute to all relevant regimes with appropriate weights

## How Probabilities Are Used

### 1. Weighted Correlation Analysis

Instead of calculating correlations only for observations with hard regime assignments, we:

1. Include all observations with non-zero probability (>1%) for the regime
2. Weight each observation by its regime probability
3. Calculate weighted correlation:
   ```
   Weighted correlation = weighted_covariance(X, Y) / sqrt(weighted_var(X) * weighted_var(Y))
   ```
4. Use effective sample size for significance testing:
   ```
   n_effective = (sum(weights))² / sum(weights²)
   ```

**Example**: An observation with 60% probability of being in Regime 0 contributes 0.6 weight to Regime 0 correlations, but also contributes to other regimes with their respective probabilities.

### 2. Weighted Regression Analysis

Linear regressions use sample weights based on regime probabilities:

1. Observations are weighted by their probability of being in the regime
2. Standard OLS regression with `sample_weight` parameter
3. Standard errors and p-values adjusted for effective sample size
4. R² calculated using weighted residuals

**Implementation**:
```python
model.fit(X, y, sample_weight=regime_probabilities)
```

### 3. Weighted Random Forest

Feature importance uses Random Forest with sample weights:

1. Each observation's contribution to tree splits is weighted by regime probability
2. Feature importance reflects weighted contribution to predictions
3. More important features are those that help predict ERP when weighted by regime probability

**Implementation**:
```python
rf.fit(X, y, sample_weight=regime_probabilities)
```

## Example: How an Observation Contributes

Consider an observation on 1991-02-28:
- Hard assignment: Regime 0 (Low Growth / High Inflation)
- Probabilities:
  - R0: 39.5%
  - R1: 18.6%
  - R2: 21.0%
  - R3: 21.0%

**With hard assignments only**: This observation contributes 100% to Regime 0 analysis only.

**With probabilities**: 
- Contributes 39.5% weight to Regime 0 analysis
- Contributes 18.6% weight to Regime 1 analysis
- Contributes 21.0% weight to Regime 2 analysis
- Contributes 21.0% weight to Regime 3 analysis

This provides a more nuanced view of how macro variables relate to ERP in each regime.

## Threshold for Inclusion

Observations with very low probability (<1%) for a regime are excluded from that regime's analysis to:
- Avoid noise from observations that are clearly in other regimes
- Improve computational efficiency
- Focus on observations that meaningfully contribute to each regime

## Effective Sample Size

When using weights, the effective sample size is:
```
n_effective = (Σw_i)² / Σw_i²
```

This accounts for the fact that weighted observations provide less information than unweighted ones. For example:
- 100 observations with equal weights: n_effective = 100
- 100 observations with weights [0.5, 0.5, 0.5, ...]: n_effective = 50
- 100 observations with weights [0.1, 0.1, 0.1, ...]: n_effective = 10

Statistical tests use n_effective rather than the raw number of observations.

## Comparison: With vs. Without Probabilities

The analysis can be run with or without probabilities:

```python
# With probabilities (default, recommended)
analyzer.analyze_correlations_by_regime(use_probabilities=True)
analyzer.analyze_regressions_by_regime(use_probabilities=True)
analyzer.analyze_feature_importance_by_regime(use_probabilities=True)

# Without probabilities (hard assignments only)
analyzer.analyze_correlations_by_regime(use_probabilities=False)
analyzer.analyze_regressions_by_regime(use_probabilities=False)
analyzer.analyze_feature_importance_by_regime(use_probabilities=False)
```

**Default behavior**: Uses probabilities if available, falls back to hard assignments if not.

## Output Indicators

Results include indicators of whether probabilities were used:
- `uses_probabilities`: Boolean flag in correlation/regression results
- `weighted_n_observations`: Effective sample size (vs. raw count)
- `effective_n`: Effective sample size in regression results

## Interpretation

When interpreting results:

1. **Higher weighted_n_observations**: More data contributing to the analysis (good)
2. **Lower weighted_n_observations**: Fewer observations or more uncertainty (less reliable)
3. **Large difference between n_observations and weighted_n_observations**: High regime uncertainty in the data

## References

- Weighted correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation
- Weighted regression: https://en.wikipedia.org/wiki/Weighted_least_squares
- Effective sample size: Kish, L. (1965). Survey Sampling.

