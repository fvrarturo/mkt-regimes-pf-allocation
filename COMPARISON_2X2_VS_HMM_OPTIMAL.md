# 2×2 vs HMM (Growth + Policy) Comparison

- Significance threshold: 5% (p-value ≤ 0.05)
- Metrics computed from the latest outputs in `s1_macro_vars/s12_regimeness/regimes`

## Headline Metrics
```
                       n_regimes                 best_regime best_regime_erp                worst_regime worst_regime_erp erp_spread  significant_pairs
model                                                                                                                                                  
2×2 Quadrants                  4                    Slowdown           0.76%                 Stagflation            0.15%      0.61%                  0
HMM (Growth + Policy)          4  Low Growth / Low Inflation          12.67%  Low Growth / Low Inflation           -0.25%     12.92%                  4
```

## Interpretation
- HMM (Growth + Policy) achieves a wider ERP spread (12.92%) than the 2×2 quadrants (0.61%), highlighting stronger regime differentiation.
- The HMM model delivers 4 significant regime pair comparisons at the 5% level versus 0 for the 2×2 approach, indicating clearer statistical separation.
- HMM isolates a much harsher risk-off regime (Low Growth / Low Inflation at -0.25%) compared with the 2×2 worst regime (Stagflation at 0.15%), highlighting the role of policy support in tail scenarios.

Generated automatically by `s12_regimeness/compare_models.py`.