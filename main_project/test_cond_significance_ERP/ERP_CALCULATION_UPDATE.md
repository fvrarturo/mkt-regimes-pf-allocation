# ERP Calculation Update

## Changes Made

### Previous Calculation (INCORRECT)
- **ERP = SP500 Level - 3m Yield Level**
- This was economically meaningless because:
  - SP500 is in absolute values (hundreds/thousands)
  - 3m yield is in percentage points (0-20)
  - The subtraction doesn't represent a meaningful economic relationship

### New Calculation (CORRECT)
- **ERP = SP500 Monthly Return - 3m Yield Monthly Rate**
- This represents the **excess return** of equities over the risk-free rate
- Both are now in the same units (monthly returns/rates)
- Economically meaningful: measures risk premium

## Frequency Alignment

### SP500 Data
- **Original frequency**: Monthly (end of month)
- **Processing**: Already monthly, no resampling needed

### 3m Yield Data
- **Original frequency**: Daily
- **Processing**: Resampled to monthly (end of month) using `.resample('M').last()`

### Yield Conversion
- 3m yield is annualized percentage (e.g., 5.0 = 5% per year)
- Converted to monthly decimal: `monthly_rate = annual_yield / 100 / 12`
- This allows direct comparison with monthly SP500 returns

## New Results Summary

### Top Variables (Now Using Returns)

**All Regimes:**
1. **Nasdaq Volatility Index** - Rank 1 (relevance: 1.000)
2. **VIX** - Rank 2 (relevance: 0.74-0.82)
3. **3-month Volatility Index SP500** - Rank 3 (relevance: 0.58-0.60)
4. **BofA High Yield Spread** - Rank 4 (relevance: 0.32-0.36)
5. **National Financial Conditions Index** - Rank 5 (relevance: 0.29-0.37)

### Interest Rates - Updated Rankings

Interest rates are now more relevant when analyzing **returns**:

**Regime 1: High Growth / High Inflation**
- **10y Treasury**: Rank 7 (relevance: 0.095, correlation: -0.051)
- **2y Yield**: Rank 10 (relevance: 0.092, correlation: -0.037)
- **10y Yield**: Rank 11 (relevance: 0.091, correlation: -0.051)

**Regime 2: High Growth / Low Inflation**
- **Fed Reserve Discount Rate**: Rank 9 (relevance: 0.070, correlation: -0.029)
- **2y Yield**: Rank 10 (relevance: 0.070, correlation: -0.026)

**Regime 3: Low Growth / Low Inflation**
- **10y-2y Spread**: Rank 9 (relevance: 0.055, correlation: -0.052)
- **10y Yield**: Rank 10 (relevance: 0.048, correlation: -0.033)

## Key Insights

### 1. Volatility Dominates
When analyzing ERP **returns**, volatility measures (VIX, Nasdaq vol) are the strongest predictors. This makes economic sense:
- High volatility → Higher risk → Higher required return
- Volatility directly affects equity returns

### 2. Interest Rates Still Matter (But Less)
Interest rates show moderate relevance (ranks 7-13 in some regimes):
- **Correlations are weak** (-0.01 to -0.05) but statistically significant in some cases
- Interest rates affect equity returns over **longer horizons**, not necessarily month-to-month
- The relationship is **regime-dependent**

### 3. Inflation Variables Less Relevant for Returns
CPI and PCE are no longer top-ranked when analyzing returns:
- They were highly relevant for **levels** (because they affect valuations)
- But less relevant for **returns** (month-to-month changes)
- This is economically correct!

### 4. Credit Spreads Important
BofA High Yield Spread ranks 4th consistently:
- Measures credit risk premium
- Directly related to equity risk premium
- More relevant than interest rates for short-term returns

## Economic Interpretation

### Why Volatility is #1
- **Risk-return relationship**: Higher volatility → Higher required return
- **Market sentiment**: Volatility captures fear/uncertainty
- **Direct impact**: Volatility directly affects realized returns

### Why Interest Rates Rank Lower
1. **Longer-term effects**: Interest rates affect equity valuations over quarters/years, not months
2. **Indirect transmission**: Rates work through:
   - Economic growth (captured by GDP, unemployment)
   - Inflation expectations (captured by CPI, PCE)
   - Credit conditions (captured by spreads)
3. **Already embedded**: Rate effects are reflected in other variables

### Why This Makes Sense
- **Levels analysis**: Inflation/GDP explain ERP levels (valuations)
- **Returns analysis**: Volatility/spreads explain ERP returns (monthly changes)
- Both are correct, but answer different questions!

## Recommendations

### For Portfolio Allocation
1. **Monitor volatility first**: VIX and volatility indices are key indicators
2. **Watch credit spreads**: High yield spreads provide early warning
3. **Use interest rates for context**: Important but less predictive for monthly returns
4. **Consider regime**: Interest rates matter more in some regimes (High Growth / High Inflation)

### For Further Analysis
1. **Longer horizons**: Test if interest rates matter more for quarterly/annual returns
2. **Lagged effects**: Test if lagged interest rates are more predictive
3. **Interaction terms**: Test interactions between rates and volatility
4. **Term structure**: Analyze yield curve effects (spreads) more deeply

## Conclusion

The updated calculation using **returns instead of levels** provides:
- ✅ **Economically meaningful** results
- ✅ **Correct frequency alignment** (monthly)
- ✅ **Proper yield conversion** (annual to monthly)
- ✅ **More relevant variable rankings** (volatility dominates, as expected)

Interest rates are still relevant but rank lower because they affect equity returns over longer horizons, not month-to-month. This is consistent with financial theory and empirical evidence.

