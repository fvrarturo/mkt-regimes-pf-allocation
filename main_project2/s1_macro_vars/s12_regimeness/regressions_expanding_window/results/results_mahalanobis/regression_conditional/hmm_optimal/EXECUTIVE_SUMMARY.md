# Executive Summary: Regime-Conditional Regression Analysis

**Model**: HMM_OPTIMAL

## Key Takeaways

### Overall Performance

- **Total regressions**: 240
- **Significant results (p<0.05)**: 15 (6.2%)
- **Highly significant (p<0.01)**: 0 (0.0%)

### Top 5 Overall Predictors

| Variable | Avg |t-stat| | Significant Results | Avg R² |
|----------|----------------|---------------------|--------|
| unemployment | 1.62 | 7 | 0.0052 |
| fed_reserve_discount_rate | 1.45 | 4 | 0.0047 |
| gdp | 1.13 | 1 | 0.0081 |
| fedfunds | 0.98 | 0 | 0.0010 |
| m2_real_money_supply | 0.97 | 1 | 0.0007 |

### Best Predictors by Regime

**Regime 0**: unemployment (avg |t-stat| = 1.86)

**Regime 1**: gdp (avg |t-stat| = 1.49)

**Regime 2**: unemployment (avg |t-stat| = 1.89)

**Regime 3**: unemployment (avg |t-stat| = 1.60)

### Forecast Horizon Performance

| Horizon | Significant Results | Avg R² |
|---------|---------------------|--------|
| 1 month(s) | 4 | 0.0006 |
| 3 month(s) | 3 | 0.0010 |
| 6 month(s) | 4 | 0.0022 |
| 12 month(s) | 4 | 0.0021 |

## Recommendations

1. **Focus on top predictors** identified above for each regime
2. **Use regime-specific models** - coefficients differ significantly across regimes
3. **Consider forecast horizon** - predictive power varies by horizon
4. **Weight by probabilities** - Use Mahalanobis distance probabilities for soft regime assignments
