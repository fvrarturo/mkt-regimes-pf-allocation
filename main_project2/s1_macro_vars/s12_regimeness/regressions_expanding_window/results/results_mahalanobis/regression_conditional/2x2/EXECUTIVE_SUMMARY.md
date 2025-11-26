# Executive Summary: Regime-Conditional Regression Analysis

**Model**: 2X2

## Key Takeaways

### Overall Performance

- **Total regressions**: 240
- **Significant results (p<0.05)**: 27 (11.2%)
- **Highly significant (p<0.01)**: 8 (3.3%)

### Top 5 Overall Predictors

| Variable | Avg |t-stat| | Significant Results | Avg R² |
|----------|----------------|---------------------|--------|
| unemployment | 1.46 | 6 | -0.0011 |
| nat_fin_condition_indx | 1.33 | 3 | -0.0052 |
| fed_reserve_discount_rate | 1.30 | 4 | -0.0028 |
| gdp | 1.14 | 3 | -0.0035 |
| fedfunds | 1.07 | 1 | -0.0047 |

### Best Predictors by Regime

**Goldilocks**: gdp (avg |t-stat| = 1.36)

**Overheating**: 10y_2y_spread (avg |t-stat| = 1.21)

**Stagflation**: nat_fin_condition_indx (avg |t-stat| = 2.19)

**Slowdown**: fed_reserve_discount_rate (avg |t-stat| = 2.06)

### Regime-Dependent Relationships

- **29 significant coefficient differences** found across regimes
- Indicates that macro variables have **different predictive power** in different regimes

### Forecast Horizon Performance

| Horizon | Significant Results | Avg R² |
|---------|---------------------|--------|
| 1 month(s) | 9 | -0.0090 |
| 3 month(s) | 8 | -0.0045 |
| 6 month(s) | 5 | -0.0021 |
| 12 month(s) | 5 | -0.0015 |

## Recommendations

1. **Focus on top predictors** identified above for each regime
2. **Use regime-specific models** - coefficients differ significantly across regimes
3. **Consider forecast horizon** - predictive power varies by horizon
4. **Weight by probabilities** - Use Mahalanobis distance probabilities for soft regime assignments
