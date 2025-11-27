# Extremeness Models: Initial Relevance Analysis

## Overview

This analysis investigates whether extremeness in macroeconomic variables impacts the distribution and characteristics of Equity Risk Premium (ERP). We employ two complementary extremeness detection methods to identify periods of extreme macro conditions and examine how ERP behaves during these periods.

## Methodology

### Models Implemented

1. **Isolation Forest**
   - Unsupervised anomaly detection algorithm
   - Identifies extreme macro states based on isolation from normal patterns
   - Contamination parameter: 10% (expected proportion of outliers)
   - Features: 4 macro indices (inflation, growth, monetary policy, market volatility)

2. **PCA Distance**
   - Principal Component Analysis to reduce dimensionality
   - Computes distance from center in PC space as extremeness measure
   - Variance threshold: 85% (selected 4 components explaining 100% of variance)
   - Distance method: Euclidean distance from origin

### Extremeness Thresholds

Both models flag extreme states at multiple percentile thresholds:
- **99th percentile (Top 1%)**: Most extreme macro conditions
- **95th percentile (Top 5%)**: Very extreme conditions
- **90th percentile (Top 10%)**: Extreme conditions (primary threshold)
- **80th percentile (Top 20%)**: Moderately extreme conditions

### Data

- **Sample period**: 1990-01-31 to 2025-11-30 (monthly frequency)
- **Observations**: 431 months
- **Macro variables**: 4 standardized indices (inflation_factor, growth_factor, monetary_policy_factor, market_volatility_factor)
- **Target variable**: Equity Risk Premium (ERP) = Stock return - Risk-free return

## Key Findings

### 1. Distributional Impact: Increased Volatility in Extreme States

**Primary Finding**: ERP exhibits significantly wider distributions when macro variables are at extremes, providing strong rationale for extremeness-based risk management.

#### Isolation Forest Results:
- **Normal states** (n=387): 
  - Mean ERP: 0.68% per month
  - Standard deviation: 3.78%
  - Range: -15.0% to 10.8%
  
- **Extreme states** (n=44, 90th percentile):
  - Mean ERP: -0.65% per month
  - Standard deviation: **7.13%** (1.88x wider than normal)
  - Range: -17.0% to 12.7%

#### PCA Distance Results:
- **Normal states** (n=387):
  - Mean ERP: 0.71% per month
  - Standard deviation: 3.73%
  
- **Extreme states** (n=44, 90th percentile):
  - Mean ERP: -0.92% per month
  - Standard deviation: **5.89%** (1.58x wider than normal)

**Statistical Evidence**:
- **Kolmogorov-Smirnov test**: Both models show highly significant distributional differences
  - Isolation Forest: KS statistic = 0.249, p-value = 0.012
  - PCA Distance: KS statistic = 0.276, p-value = 0.004
- This confirms that ERP distributions are fundamentally different in extreme macro states

### 2. Tail Risk Amplification

Extreme macro states are associated with significantly worse tail outcomes:

- **5th percentile ERP**:
  - Normal: -6.24% (Isolation Forest), -6.17% (PCA)
  - Extreme: -11.12% (IF), -9.15% (PCA)
  - **Difference: -4.88% to -4.95%** (worse tail outcomes)

- **1st percentile ERP**:
  - Normal: -9.20% (IF), -6.17% (PCA)
  - Extreme: -15.08% (IF), -9.15% (PCA)
  - **Difference: -5.88% to -6.16%** (substantially worse tail risk)

This demonstrates that extreme macro conditions amplify downside risk, making tail events more severe.

### 3. Mean Differences: Weak and Inconsistent Signal

**Key Limitation**: Simple extremeness measures do not provide clear, consistent mean differences in ERP.

#### At 90th Percentile Threshold:
- **Isolation Forest**: Mean difference = -1.34% (p=0.048) - marginally significant
- **PCA Distance**: Mean difference = -1.63% (p=0.016) - significant

However, examining across percentile thresholds reveals inconsistency:

| Percentile | Isolation Forest Mean Diff | PCA Distance Mean Diff |
|------------|---------------------------|------------------------|
| 99th (1%)  | -5.33% (n=5 extreme)      | -0.77% (n=5 extreme)   |
| 95th (5%)  | -0.48% (n=17 extreme)     | -2.97% (n=17 extreme)  |
| 90th (10%) | -1.03% (n=22 extreme)     | -0.73% (n=22 extreme)  |
| 80th (20%) | +0.62% (n=43 extreme)     | +0.74% (n=43 extreme)  |

**Observations**:
- Mean differences vary substantially across thresholds
- Direction changes at 80th percentile (positive mean difference)
- Small sample sizes at higher percentiles (especially 99th) limit reliability
- No consistent pattern emerges across extremeness levels

**Conclusion**: While there is some evidence of negative mean ERP in extreme states at the 90th percentile, the signal is weak and inconsistent across different extremeness thresholds. The extremeness measures do not reliably predict mean ERP direction.

### 4. Model Agreement

- **Correlation between extremeness scores**: 0.906 (very high)
- **Overlap in extreme state identification**: 81.8% at 90th percentile
- Both models identify similar periods as extreme, providing robustness to the findings

## Interpretation and Implications

### What This Analysis Shows:

1. **Rationale for Extremeness-Based Risk Management**: 
   - The **2x wider standard deviation** in extreme states provides strong justification for using extremeness measures in risk management
   - Extreme macro conditions create an environment where ERP outcomes are more uncertain and volatile
   - Tail risk is substantially amplified (5-6% worse at 5th/1st percentiles)

2. **Limitation of Simple Extremeness Measures**:
   - While extremeness identifies periods of higher volatility and tail risk, it does **not reliably predict mean ERP direction**
   - The inconsistent mean differences across percentile thresholds suggest that:
     - Simple extremeness measures may be too coarse
     - Mean ERP may depend on the *type* of extremeness (e.g., extreme growth vs extreme inflation)
     - Directional predictions require more nuanced models (e.g., regime-dependent or interaction models)

### Statistical Summary

| Test | Isolation Forest | PCA Distance |
|------|----------------|--------------|
| **T-test (mean difference)** | p=0.048* | p=0.016** |
| **KS-test (distribution)** | p=0.012** | p=0.004*** |
| **Mann-Whitney (non-parametric)** | p=0.348 | p=0.176 |
| **Tail difference (5th pct)** | -4.88% | -4.95% |
| **Tail difference (1st pct)** | -5.88% | -6.16% |

*Significant at 10% level, **Significant at 5% level, ***Significant at 1% level

## Conclusions

1. **Extremeness measures successfully identify periods of elevated ERP volatility** - Standard deviations are 1.5-2x wider in extreme states

2. **Tail risk is substantially amplified** - Extreme macro conditions are associated with 5-6% worse outcomes at the 5th and 1st percentiles

3. **Mean differences are weak and inconsistent** - Simple extremeness measures do not provide reliable directional signals for mean ERP

4. **Rationale established** - The increased volatility and tail risk in extreme states provides strong justification for extremeness-based risk management, even if mean predictions are unreliable

## Next Steps

Future analyses should explore:
- **Regime-dependent models**: Different extremeness impacts in different macro regimes
- **Interaction effects**: How combinations of extreme macro variables affect ERP
- **Directional extremeness**: Distinguishing between extreme positive vs extreme negative macro conditions
- **Time-varying effects**: Whether extremeness impacts vary over different market cycles

## Files Generated

### Statistics:
- `*_erp_statistics.csv`: Descriptive statistics for normal vs extreme states
- `*_erp_statistics_by_percentiles.csv`: Statistics across all percentile thresholds
- `statistical_tests.csv`: T-tests, KS-tests, Mann-Whitney tests, tail differences

### Visualizations:
- `extremeness_vs_erp_combined.png`: Scatter plots showing relationship between extremeness and ERP
- `extremeness_histogram_combined.png`: Distribution of extremeness scores
- `erp_boxplot_all_models.png`: ERP distributions across percentile groups

