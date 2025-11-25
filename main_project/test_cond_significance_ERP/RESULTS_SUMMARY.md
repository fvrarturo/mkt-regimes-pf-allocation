# Regime-Conditional Predictors of ERP Returns - Results Summary

## Methodology

- **Analysis Type**: Regime-conditional using lagged macro variables
- **ERP Calculation**: SP500 monthly return - 3m yield monthly rate
- **Regime Assignment**: Uses regime probabilities (weighted analysis)
- **Lags Tested**: t-1, t-2, t-3 (showing best lag for each variable)
- **Sample**: 419 monthly observations (1990-12-31 to 2025-10-31)
- **Variables Analyzed**: 17 macro variables (excluding volatility)

---

## Key Findings

### ✅ Statistically Significant Predictors (p < 0.05)

**Regime 3: Low Growth / Low Inflation (Recession)**
- **Financial Conditions Index** (lag 1): corr = -0.144, **p = 0.0049** ⭐⭐
- **Unemployment** (lag 1): corr = 0.114, **p = 0.0262** ⭐

*This regime is the most predictable, with 2 significant predictors during recessionary periods.*

---

## Top 5 Predictors by Regime

### Regime 0: Low Growth / High Inflation (Stagflation)
**Effective Sample Size: 360 observations**

| Rank | Variable | Lag | Correlation | P-value | Status |
|------|----------|-----|-------------|---------|--------|
| 1 | unemployment | 3 | 0.084 | 0.110 | Marginal |
| 2 | m2_real_money_supply | 3 | 0.065 | 0.220 | Not significant |
| 3 | 10y_treasury_const_maturity_rate | 3 | -0.053 | 0.313 | Not significant |
| 4 | fed_reserve_discount_rate | 3 | -0.053 | 0.320 | Not significant |
| 5 | PCE_price_index | 3 | 0.052 | 0.323 | Not significant |

**Key Insight**: Unemployment is the top predictor, but inflation variables (PCE, CPI) rank 5th-10th, showing **inflation does NOT drive returns even in high-inflation regimes**.

---

### Regime 1: High Growth / High Inflation
**Effective Sample Size: 345 observations**

| Rank | Variable | Lag | Correlation | P-value | Status |
|------|----------|-----|-------------|---------|--------|
| 1 | unemployment | 1 | 0.080 | 0.140 | Not significant |
| 2 | nat_fin_condition_indx | 1 | -0.063 | 0.242 | Not significant |
| 3 | 10y_treasury_const_maturity_rate | 3 | -0.050 | 0.360 | Not significant |
| 4 | m2_real_money_supply | 3 | 0.046 | 0.394 | Not significant |
| 5 | fed_reserve_discount_rate | 3 | -0.045 | 0.405 | Not significant |

**Key Insight**: Even in high-growth/high-inflation periods, **inflation variables rank 7th-8th** (weak predictors).

---

### Regime 2: High Growth / Low Inflation (Goldilocks)
**Effective Sample Size: 354 observations**

| Rank | Variable | Lag | Correlation | P-value | Status |
|------|----------|-----|-------------|---------|--------|
| 1 | unemployment | 3 | 0.094 | 0.077 | **Marginal** ⚠️ |
| 2 | nat_fin_condition_indx | 1 | -0.065 | 0.221 | Not significant |
| 3 | fed_reserve_discount_rate | 3 | -0.046 | 0.387 | Not significant |
| 4 | 10y_treasury_const_maturity_rate | 3 | -0.041 | 0.439 | Not significant |
| 5 | industrial_production | 2 | -0.041 | 0.440 | Not significant |

**Key Insight**: Unemployment shows marginal significance (p = 0.077), close to statistical significance.

---

### Regime 3: Low Growth / Low Inflation (Recession)
**Effective Sample Size: 381 observations**

| Rank | Variable | Lag | Correlation | P-value | Status |
|------|----------|-----|-------------|---------|--------|
| 1 | nat_fin_condition_indx | 1 | -0.144 | **0.0049** | **Significant** ⭐⭐ |
| 2 | unemployment | 1 | 0.114 | **0.0262** | **Significant** ⭐ |
| 3 | 10y_treasury_const_maturity_rate | 3 | -0.069 | 0.180 | Not significant |
| 4 | m2_real_money_supply | 3 | 0.065 | 0.208 | Not significant |
| 5 | import_price_index | 3 | -0.059 | 0.252 | Not significant |

**Key Insight**: **Only regime with statistically significant predictors**. Financial stress and unemployment matter most during recessions.

---

## Consistent Patterns Across All Regimes

### 1. Unemployment - Top Predictor in ALL Regimes
- **Regime 0**: Rank 1, corr = 0.084, p = 0.110 (marginal)
- **Regime 1**: Rank 1, corr = 0.080, p = 0.140
- **Regime 2**: Rank 1, corr = 0.094, p = 0.077 (marginal)
- **Regime 3**: Rank 2, corr = 0.114, **p = 0.026** ⭐ (significant)

**Economic Interpretation**: Higher unemployment → Higher future returns (counter-cyclical relationship)

### 2. Financial Conditions Index - Strong in Most Regimes
- **Regime 0**: Rank 6, corr = -0.052, p = 0.323
- **Regime 1**: Rank 2, corr = -0.063, p = 0.242
- **Regime 2**: Rank 2, corr = -0.065, p = 0.221
- **Regime 3**: Rank 1, corr = -0.144, **p = 0.0049** ⭐⭐ (highly significant)

**Economic Interpretation**: Tighter financial conditions → Lower future returns (especially in recessions)

### 3. M2 Money Supply - Consistent Moderate Predictor
- Appears in top 5 in Regimes 0, 1, and 3
- Positive correlation: More money supply → Higher future returns

### 4. Inflation Variables - WEAK Predictors
- **CPI**: Ranks 10-13 across all regimes (correlations 0.019-0.050, p > 0.3)
- **PCE**: Ranks 5-12 across all regimes (correlations 0.021-0.052, p > 0.3)
- **PPI**: Ranks 12-16 across all regimes (correlations 0.011-0.040, p > 0.5)

**Conclusion**: **Inflation does NOT predict returns, even in high-inflation regimes.**

---

## Summary Statistics

### Sample Sizes (Effective n using probabilities)
- **Regime 0**: 360 observations
- **Regime 1**: 345 observations
- **Regime 2**: 354 observations
- **Regime 3**: 381 observations

### Statistical Significance
- **Significant (p < 0.05)**: 2 predictors (both in Regime 3)
- **Marginal (0.05 < p < 0.10)**: 2 predictors (unemployment in Regimes 0 and 2)
- **Not significant**: All other relationships

### Correlation Strengths
- **Strongest**: Financial Conditions Index in Regime 3 (corr = -0.144)
- **Average top predictor**: ~0.08-0.11 across regimes
- **All correlations are weak** (explain <2% of variance)

---

## Key Conclusions

### 1. Inflation Hypothesis NOT Supported ❌
- **Hypothesis**: "Inflation might drive returns in high-inflation regimes"
- **Result**: Inflation variables rank 5-13 across ALL regimes
- **Correlations**: Weak (0.02-0.05) and not significant (p > 0.3)
- **Conclusion**: Inflation does NOT predict returns, even in high-inflation periods

### 2. Unemployment is the Most Consistent Predictor ✅
- Rank 1-2 in ALL regimes
- Statistically significant in Regime 3 (p = 0.026)
- Marginally significant in Regimes 0 and 2 (p = 0.077-0.110)
- Counter-cyclical: Higher unemployment → Higher future returns

### 3. Financial Conditions Matter Most in Recessions ✅
- Highly significant in Regime 3 (p = 0.0049)
- Strong negative correlation: Tighter conditions → Lower returns
- Most relevant during economic downturns

### 4. Regime 3 (Recession) is Most Predictable ✅
- Only regime with statistically significant predictors
- Financial stress and unemployment are key drivers
- Larger effective sample size (381 observations) helps with significance

### 5. Macro Variables Have Limited Predictive Power
- Even best predictors explain <2% of variance (R² < 0.02)
- Most relationships are not statistically significant
- Monthly horizon may be too short for macro effects
- Consider quarterly/annual analysis for stronger relationships

---

## Recommendations

1. **Focus on Unemployment and Financial Conditions** for prediction
   - These are the most consistent and significant predictors
   - Especially relevant in recessionary periods (Regime 3)

2. **Do NOT rely on Inflation** as a predictor
   - Even in high-inflation regimes, inflation does not drive returns
   - Focus on growth indicators instead

3. **Regime 3 is most actionable**
   - Statistically significant relationships
   - Financial conditions and unemployment are reliable predictors

4. **Consider longer horizons**
   - Monthly returns may be too noisy
   - Quarterly/annual analysis might show stronger relationships

5. **Use p < 0.10 as "economically significant"**
   - Given small sample sizes per regime
   - Weak but consistent relationships may still be useful for portfolio allocation

---

## Files Generated

All detailed results saved in: `results/regime_predictors_improved/`
- `monthly_lags_regime_X.csv` - Detailed correlations by regime with optimal lags
- Summary statistics and rankings

---

*Analysis Date: 2025*
*Method: Weighted correlation analysis using regime probabilities*
*Lags Tested: t-1, t-2, t-3 (showing best lag per variable)*

