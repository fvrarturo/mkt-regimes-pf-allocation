# Extremeness Models: Initial Relevance Analysis

## Overview

This analysis investigates whether extremeness in macroeconomic (and macro + sentiment) variables changes the distribution of the Equity Risk Premium (ERP). The goal is to understand when extremeness scores should be used as risk‑management inputs in the broader asset-allocation pipeline from `codex_instructions.md`.

## Methodology

### Models Implemented

We run two extremeness algorithms **across two feature sets**:

| Model | Description | Feature Sets |
|-------|-------------|--------------|
| **Isolation Forest** | Unsupervised anomaly detector with 10 % contamination | (i) Macro only (4 factors); (ii) Macro + sentiment indices |
| **PCA Distance** | PCA to 85 % variance then Euclidean distance from origin | Same two feature sets |

### Extremeness Thresholds

Both models flag extreme states at multiple percentile thresholds:
- **99th percentile (Top 1%)**: Most extreme macro conditions
- **95th percentile (Top 5%)**: Very extreme conditions
- **90th percentile (Top 10%)**: Extreme conditions (primary threshold)
- **80th percentile (Top 20%)**: Moderately extreme conditions

All variants use the same **1990‑01 to 2025‑11 monthly sample** (431 obs). Sentiment indices are the LLM-derived category scores aligned with the same month-end calendar.

## Key Findings

### 1. Macro factors alone are still the cleanest signal

| Configuration | Normal ERP mean | Extreme ERP mean | Mean diff | t‑pvalue | KS p‑value |
|---------------|-----------------|------------------|-----------|----------|------------|
| **Macro Only – Isolation Forest** | 0.68 % | -0.65 % | **-1.34 ppt** | 0.048 | 0.012 |
| **Macro Only – PCA Distance** | 0.71 % | -0.92 % | **-1.63 ppt** | 0.016 | 0.004 |
| Macro + Sentiment – Isolation Forest | 0.62 % | -0.13 % | -0.76 ppt | 0.264 | 0.153 |
| Macro + Sentiment – PCA Distance | 0.62 % | -0.07 % | -0.69 ppt | 0.308 | 0.082 |

- Macro-only features deliver 5 %‑significant mean differences and distributional shifts (KS p ≤ 0.012).  
- Adding sentiment **dampens** the extremeness contrast and loses statistical significance—the anomaly detectors treat the noisier 8‑feature space as more “normal.”

### 2. Distributional impact: still wider volatility in extreme states

- For macro-only runs, extreme buckets retain the earlier finding of **~2× higher ERP standard deviation** and wider ranges.  
- The macro + sentiment runs still show wider dispersion, but the KS p‑values > 0.08 highlight that sentiment noise obscures the contrast.

### 3. Tail risk amplification survives, regardless of feature set

- All configurations show **5–6 ppt** worse ERP at the 5th and 1st percentiles (see `statistical_tests.csv`, columns `tail_diff_p5` / `tail_diff_p1`).  
- Sentiment does *not* improve tail isolation: differences stay negative but without stronger significance.

### 4. Model agreement remains high

- Correlation between the two extremeness scores (per feature set) is ≥ 0.90.  
- At the 90th percentile the overlap in flagged observations stays above 80 %, so downstream strategies can treat the two detectors as interchangeable score sources.

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

### 5. Mean differences remain a secondary tool

- Macro-only scores provide marginal to moderate significance (p ≈ 0.016–0.048) at the 90th percentile.  
- Sentiment-enriched scores fail to produce meaningful mean separation (p > 0.26).  
- The feature-set comparison highlights that extremeness is more about identifying **risk regimes** (volatility/tails) than predicting ERP direction; regime context is still required.

## Conclusions

1. **Use extremeness as a tail-risk overlay**: The volatility/tail widening is robust and doesn’t rely on noisy sentiment inputs.
2. **Macro-only features are the default**: They keep the signal tight and statistically significant; sentiment scores should enter through other channels (e.g., forecasting).
3. **Directional ERP views still require regimes/forecasts**: Extremeness on its own has weak and unstable mean signals, so it should be combined with regime probabilities before driving allocations.

## Next Steps

Future analyses should explore:
- Regime × extremeness interactions (e.g., only cut risk when risk-off regime **and** high extremeness)
- Directional extremeness (positive vs negative macro shocks)
- Whether tuning sentiment inputs (different horizons/providers) can improve anomaly detection rather than dilute it

## Files Generated

### Statistics:
- `*_erp_statistics.csv`: Descriptive statistics for normal vs extreme states
- `*_erp_statistics_by_percentiles.csv`: Statistics across all percentile thresholds
- `statistical_tests.csv`: T-tests, KS-tests, Mann-Whitney tests, tail differences

### Visualizations:
- `extremeness_vs_erp_combined.png`: Scatter plots showing relationship between extremeness and ERP
- `extremeness_histogram_combined.png`: Distribution of extremeness scores
- `erp_boxplot_all_models.png`: ERP distributions across percentile groups
