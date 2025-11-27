# Executive Summary: Regime-Conditional Regression Analysis

**Model**: HMM_OPTIMAL

## Key Takeaways

### Overall Performance

- **Total regressions**: 240
- **Significant results (p<0.05)**: 15 (6.2%)
- **Highly significant (p<0.01)**: 4 (1.7%)

### Top 5 Overall Predictors

| Variable | Avg |t-stat| | Significant Results | Avg R² |
|----------|----------------|---------------------|--------|
| fedfunds | 1.07 | 1 | 0.0163 |
| fed_reserve_discount_rate | 1.00 | 2 | 0.0204 |
| gdp | 0.98 | 2 | 0.0411 |
| nat_fin_condition_indx | 0.95 | 0 | 0.0106 |
| unemployment | 0.91 | 1 | 0.0112 |

### Best Predictors by Regime

**Regime 0**: nat_fin_condition_indx (avg |t-stat| = 1.17)

**Regime 1**: fedfunds (avg |t-stat| = 1.21)

**Regime 2**: unemployment (avg |t-stat| = 1.22)

**Regime 3**: fed_reserve_discount_rate (avg |t-stat| = 1.61)

### Regime-Dependent Relationships

- **18 significant coefficient differences** found across regimes
- Indicates that macro variables have **different predictive power** in different regimes

### Forecast Horizon Performance

| Horizon | Significant Results | Avg R² |
|---------|---------------------|--------|
| 1 month(s) | 3 | 0.0132 |
| 3 month(s) | 0 | 0.0119 |
| 6 month(s) | 11 | 0.0191 |
| 12 month(s) | 1 | 0.0104 |

## Recommendations

1. **Focus on top predictors** identified above for each regime
2. **Use regime-specific models** - coefficients differ significantly across regimes
3. **Consider forecast horizon** - predictive power varies by horizon
4. **Weight by probabilities** - Use Mahalanobis distance probabilities for soft regime assignments
