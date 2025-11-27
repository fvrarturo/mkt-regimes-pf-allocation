# HMM Regime Detection - Summary

## Folder Organization

The `HMM_regimes` folder is now cleanly organized with:

### Core Files
- `main.py` - 4 variables model (Growth + Inflation + Policy + Volatility)
- `run_growth_policy_model.py` - Growth + Policy model (optimal, best BIC)
- `run_growth_inflation_model.py` - Growth + Inflation model (aligns with 2×2)
- `hmm_model.py` - HMM model implementation
- `plotting.py` - Visualization functions (with interpretations)
- `results.py` - Statistical tests and results processing
- `ECONOMIC_ANALYSIS.md` - ⭐ **Comprehensive economic comparison**

### Results Folders
- `results_4vars/` - All 4 variables model results
- `results_2vars_optimal/` - Growth + Policy model (best statistical fit)
- `results_2vars_growth_inflation/` - Growth + Inflation model

## Key Findings

### Model Comparison

| Model | BIC | ERP Spread | Regimes | Recommendation |
|-------|-----|------------|---------|----------------|
| **4 Variables** | 3031.83 | ~2% | 4 (confusing labels) | Not recommended |
| **Growth + Policy** | **755.37** ⭐ | **4.16%** | 4 (clear) | **⭐ Recommended** |
| **Growth + Inflation** | 1732.42 | 2.27% | 3 (only 3 regimes) | For comparison only |

### Best Model: Growth + Policy

**Why it's best:**
1. **Best statistical fit** (BIC = 755.37, 75% lower than 4-var model)
2. **Strongest ERP differentiation** (4.16% spread between best and worst)
3. **Clear economic logic**: Growth (fundamental) + Policy (valuation)
4. **4 distinct regimes** with meaningful labels
5. **Policy as transmission mechanism** for inflation effects

**Key Economic Insight:**
- **Policy effectiveness matters more than growth alone**
- R1 (low growth + policy support) has **+1.40% ERP** (best)
- R0 (low growth + no policy support) has **-2.76% ERP** (worst)
- This suggests **liquidity-driven returns** can offset weak fundamentals

## Plot Interpretations

All plots now include **interpretation text boxes** explaining:
- **Regime probabilities plot**: "Higher probability = more confident regime assignment. Mixed colors indicate uncertain periods."
- **Transition matrix**: "Diagonal = persistence (staying in same regime). Off-diagonal = transition probability."
- **Regime interpretation plot**: "Points = observations, Stars = regime centroids, Dashed lines = medians"

## Link to 2×2 Quadrants

The **Growth + Policy model** reveals that:
- 2×2 found: Stagflation has negative ERP (-0.36%)
- HMM finds: Low growth with **policy support** has **positive ERP** (+1.40%)
- **Key insight**: Policy stance matters more than simple 2×2 classification suggests

## Next Steps

1. **Use Growth + Policy model** for production analysis
2. **Compare with 2×2 quadrants** to understand differences
3. **See ECONOMIC_ANALYSIS.md** for detailed discussion
4. **Check regime_statistics.csv** in each folder for complete macro statistics

## Files to Review

- **ECONOMIC_ANALYSIS.md** - Comprehensive economic comparison
- **results_2vars_optimal/regime_statistics.csv** - Best model statistics
- **results_2vars_optimal/regime_interpretation_plots.png** - How regimes are labeled
- **results_2vars_optimal/regime_probabilities_time_series.png** - Regime uncertainty over time

