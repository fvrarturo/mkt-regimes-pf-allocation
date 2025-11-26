# Executive Summary: Regime-Conditional Regression Analysis

**Model**: 2X2

## Key Takeaways

### Overall Performance

- **Total regressions**: 240
- **Significant results (p<0.05)**: 22 (9.2%)
- **Highly significant (p<0.01)**: 3 (1.2%)

### Top 5 Overall Predictors

| Variable | Avg |t-stat| | Significant Results | Avg R² |
|----------|----------------|---------------------|--------|
| fed_reserve_discount_rate | 1.09 | 4 | 0.0232 |
| unemployment | 1.08 | 1 | 0.0144 |
| fedfunds | 1.06 | 3 | 0.0179 |
| 10y_treasury_const_maturity_rate | 0.98 | 2 | 0.0139 |
| m2_real_money_supply | 0.98 | 2 | 0.0134 |

### Best Predictors by Regime

**Goldilocks**: 10y_2y_spread (avg |t-stat| = 1.03)

**Overheating**: 10y_2y_spread (avg |t-stat| = 1.02)

**Stagflation**: unemployment (avg |t-stat| = 1.52)

**Slowdown**: m2_real_money_supply (avg |t-stat| = 1.99)

### Regime-Dependent Relationships

- **25 significant coefficient differences** found across regimes
- Indicates that macro variables have **different predictive power** in different regimes

### Forecast Horizon Performance

| Horizon | Significant Results | Avg R² |
|---------|---------------------|--------|
| 1 month(s) | 4 | 0.0145 |
| 3 month(s) | 6 | 0.0110 |
| 6 month(s) | 0 | 0.0093 |
| 12 month(s) | 12 | 0.0177 |

## Recommendations

1. **Focus on top predictors** identified above for each regime
2. **Use regime-specific models** - coefficients differ significantly across regimes
3. **Consider forecast horizon** - predictive power varies by horizon
4. **Weight by probabilities** - Use Mahalanobis distance probabilities for soft regime assignments
