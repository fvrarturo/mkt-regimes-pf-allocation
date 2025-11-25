# Regime-Conditional Predictors of ERP Returns

## Analysis Methodology

**Using LAGGED variables**: Macro variables at time t-1 predict ERP returns at time t
- This identifies truly **predictive** relationships, not just contemporaneous correlations
- Excludes volatility variables (they're descriptive, not predictive)

## Key Finding: Your Hypothesis is NOT Supported

**Hypothesis**: "Inflation might drive returns in high-inflation regimes"

**Result**: Inflation variables (CPI, PCE, PPI) are **NOT** strong predictors in any regime:
- **Regime 0** (Low Growth / High Inflation): CPI rank 10/17, PCE rank 9/17
- **Regime 1** (High Growth / High Inflation): CPI rank 8/17, PCE rank 6/17 (best, but still weak)
- **Regime 2** (High Growth / Low Inflation): CPI rank 10/17, PCE rank 8/17
- **Regime 3** (Low Growth / Low Inflation): CPI rank 13/17, PCE rank 12/17

**Correlations are weak** (0.019 to 0.042) and **not statistically significant** (p-values 0.4-0.7).

## What Actually Predicts Returns by Regime

### Consistent Across All Regimes

1. **Unemployment** - Rank 1-2 in ALL regimes
   - Regime 0: corr = 0.079 (rank 1)
   - Regime 1: corr = 0.080 (rank 1)
   - Regime 2: corr = 0.093 (rank 1)
   - Regime 3: corr = 0.114* (rank 2, p < 0.05 - statistically significant!)

2. **National Financial Conditions Index** - Rank 2-3 in most regimes
   - Regime 0: corr = -0.052 (rank 3)
   - Regime 1: corr = -0.063 (rank 2)
   - Regime 2: corr = -0.065 (rank 2)
   - Regime 3: corr = -0.144** (rank 1, p < 0.01 - highly significant!)

3. **M2 Money Supply** - Rank 2-6 across regimes
   - Regime 0: corr = 0.061 (rank 2)
   - Regime 1: corr = 0.045 (rank 3)
   - Regime 2: corr = 0.034 (rank 6)
   - Regime 3: corr = 0.063 (rank 3)

### Regime-Specific Patterns

**Regime 0: Low Growth / High Inflation (Stagflation)**
- Top predictors: Unemployment (0.079), M2 (0.061), Financial Conditions (-0.052)
- Inflation variables rank 9-13 (weak)
- **Insight**: Even in high-inflation periods, inflation doesn't predict returns

**Regime 1: High Growth / High Inflation**
- Top predictors: Unemployment (0.080), Financial Conditions (-0.063), M2 (0.045)
- PCE ranks 6th (best inflation performance, but still weak at 0.034)
- **Insight**: Growth matters more than inflation even when both are high

**Regime 2: High Growth / Low Inflation (Goldilocks)**
- Top predictors: Unemployment (0.093), Financial Conditions (-0.065), Industrial Production (-0.040)
- Inflation variables rank 8-16 (very weak)
- **Insight**: In ideal conditions, unemployment and financial conditions matter most

**Regime 3: Low Growth / Low Inflation (Recession/Deflation)**
- Top predictors: Financial Conditions (-0.144**, p<0.01), Unemployment (0.114*, p<0.05)
- **Only regime with statistically significant predictors!**
- Inflation variables rank 12-16 (weakest)
- **Insight**: Financial stress and unemployment predict returns in recessions

## Important Caveats

1. **All correlations are weak** (0.02 to 0.14)
   - Even the "best" predictors explain <2% of variance (R² < 0.02)
   - This suggests macro variables have limited predictive power for monthly returns

2. **Most relationships are not statistically significant**
   - Only Regime 3 shows significant relationships (p < 0.05)
   - This could be due to small sample size in that regime (23 observations)

3. **Monthly horizon may be too short**
   - Macro variables might matter more at quarterly/annual horizons
   - Monthly returns are dominated by noise

## Conclusions

### Your Hypothesis: NOT Supported
- **Inflation does NOT drive returns in high-inflation regimes**
- Inflation variables rank 6-13 across all regimes
- Correlations are weak (0.02-0.04) and not significant

### What Actually Matters

1. **Unemployment** - Consistently the strongest predictor across all regimes
   - Higher unemployment → Higher future returns (counter-cyclical)
   - Makes sense: bad economic news → lower prices → higher future returns

2. **Financial Conditions** - Strong predictor, especially in Regime 3
   - Tighter financial conditions → Lower future returns
   - Most significant in recessionary periods

3. **Money Supply (M2)** - Moderate predictor
   - More money supply → Higher future returns
   - Consistent with monetary policy transmission

4. **Monetary Policy (Interest Rates)** - Weak but consistent
   - 10y Treasury shows negative correlation (higher rates → lower returns)
   - Fed funds rate shows no predictive power

### Regime Differences

- **Regime 3 (Recession)** shows the strongest relationships (statistically significant)
- **Regime 2 (Goldilocks)** shows weakest relationships (everything is good)
- **High-inflation regimes (0, 1)** don't show inflation as a key driver

## Recommendations

1. **Focus on unemployment and financial conditions** for prediction
2. **Don't rely on inflation** as a predictor, even in high-inflation regimes
3. **Consider longer horizons** (quarterly/annual) where macro variables might matter more
4. **Regime 3 is most predictable** - financial stress and unemployment matter most in recessions

## Files Generated

- `lagged_correlations_regime_X.csv` - Detailed correlations by regime
- `regressions_regime_X.csv` - Regression results by regime
- `summary_by_category.csv` - Summary by variable category

All results saved in: `results/regime_predictors/`

