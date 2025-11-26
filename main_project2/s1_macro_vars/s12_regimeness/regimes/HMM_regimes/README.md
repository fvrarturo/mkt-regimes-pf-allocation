# HMM Regime Detection with Macro Variables

This module implements **Gaussian Hidden Markov Models (HMM)** for detecting macroeconomic regimes using different combinations of macro variables.

## Overview

Three HMM models are implemented and compared:
1. **4 Variables**: Growth + Inflation + Policy + Volatility
2. **2 Variables (Optimal)**: Growth + Policy (best statistical fit)
3. **2 Variables (Growth+Inflation)**: Growth + Inflation (aligns with 2×2 quadrants)

## Quick Start

### Run All Models

```bash
# 4 Variables Model
python main.py

# Growth + Policy Model (Optimal)
python run_growth_policy_model.py

# Growth + Inflation Model  
python run_growth_inflation_model.py
```

## File Structure

```
HMM_regimes/
├── main.py                      # 4 variables model
├── run_growth_policy_model.py   # Growth + Policy model (optimal)
├── run_growth_inflation_model.py # Growth + Inflation model
├── hmm_model.py                 # HMM model class
├── plotting.py                  # Visualization functions (with interpretations)
├── results.py                   # Statistical tests and results processing
├── README.md                    # This file
├── ECONOMIC_ANALYSIS.md         # ⭐ Economic comparison of all models
│
└── Results folders:
    ├── results_4vars/           # All 4 variables model results
    ├── results_2vars_optimal/   # Growth + Policy model (best BIC)
    └── results_2vars_growth_inflation/  # Growth + Inflation model
```

## Model Comparison

| Model | Variables | Best K | BIC | Key Finding |
|-------|-----------|--------|-----|-------------|
| **4 Variables** | Growth + Inflation + Policy + Volatility | 4 | 3031.83 | Most comprehensive but worst fit |
| **2 Variables (Optimal)** | Growth + Policy | 4 | **755.37** ⭐ | **Best statistical fit** |
| **2 Variables (G+I)** | Growth + Inflation | 3 | 1732.42 | Aligns with 2×2 but only 3 regimes |

**Recommendation**: See `ECONOMIC_ANALYSIS.md` for detailed comparison and recommendation.

## Key Features

- **Automatic K Selection**: Tests K=2,3,4 and selects best by BIC
- **Regime Probabilities**: Soft assignments with uncertainty quantification
- **Transition Matrix**: Shows regime persistence
- **Statistical Tests**: t-tests and ANOVA for ERP differences
- **Plot Interpretations**: All plots include explanatory text
- **Detailed Statistics**: Growth and inflation statistics for each regime

## Outputs

Each results folder contains:
- `regime_statistics.csv` - Complete macro and ERP statistics
- `regime_assignments.csv` - Time series with regime assignments and probabilities
- `transition_matrix.csv` - Regime transition probabilities
- `regime_interpretation_plots.png` - How regimes are labeled (with interpretations)
- `regime_probabilities_time_series.png` - Regime probabilities over time (with interpretations)
- `transition_matrix_heatmap.png` - Transition matrix visualization (with interpretations)
- Statistical test results (t-tests, ANOVA)

## Interpretation Method

Regimes are labeled by:
1. Calculating average macro values for each regime
2. Comparing to overall median (High if ≥ median, Low otherwise)
3. Creating descriptive name: "{Growth Level} Growth / {Inflation Level} Inflation"

See `regime_interpretation_plots.png` in each results folder for visual explanation.

## Economic Analysis

See **`ECONOMIC_ANALYSIS.md`** for:
- Detailed comparison of all three models
- Economic interpretation of each model
- Link to 2×2 quadrant findings
- Recommendation on which model to use
- Discussion of which macro variables matter most for ERP

## Requirements

- pandas, numpy, scikit-learn, hmmlearn, scipy, matplotlib, seaborn

See `requirements.txt` if available, or install via:
```bash
pip install pandas numpy scikit-learn hmmlearn scipy matplotlib seaborn
```
