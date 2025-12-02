# MIDAS TVP-VAR Forecasting Model

## Overview

This folder contains the implementation of a **MIDAS (Mixed Data Sampling) Time-Varying Parameter VAR (TVP-VAR)** model that combines monthly macroeconomic factors with daily oil price data to forecast growth and inflation.

## Key Innovation

Traditional VAR models treat all variables as the same frequency. This model:
1. **Aggregates daily oil prices** into monthly MIDAS factors using exponential weighting
2. **Combines with 4 monthly macro factors** (growth, inflation, policy, volatility)
3. **Fits a TVP-VAR** with expanding windows to capture time-varying relationships

## Model Architecture

### Variables (5-dimensional VAR)

```
Y_t = [growth_factor_t, inflation_factor_t, monetary_policy_factor_t, 
       market_volatility_factor_t, oil_midas_t]
```

1. **Growth Factor** (monthly) - Industrial Production month-over-month % change
2. **Inflation Factor** (monthly) - CPI-based inflation measure
3. **Monetary Policy Factor** (monthly) - Federal funds rate proxy
4. **Market Volatility Factor** (monthly) - VIX-based volatility
5. **Oil MIDAS Factor** (monthly) - Daily oil prices aggregated via MIDAS

### MIDAS Aggregation

Daily oil prices are aggregated into monthly factors using exponential weights:

```
OIL_MIDAS_t = Σ(k=0 to K-1) w_k * oil_price_{t-k}

where w_k = exp(-θk) / Σ(j=0 to K-1) exp(-θj)
```

**Parameters:**
- `θ (theta)`: Decay parameter (default: 0.03) - controls weight decay speed
  - Higher θ → faster decay → recent prices matter more
  - Lower θ → slower decay → past prices weighted similarly
- `K`: Number of daily lags to aggregate (default: 60) - roughly 3 months of trading days

**Intuition:** Recent oil prices get exponentially higher weights than older prices, capturing momentum and recent shocks while avoiding the need for static lags.

## Methodology

### 1. Data Preparation

```
Raw Data
├── Macro Data (monthly, 1989-2025)
│   └── 4 factors: growth, inflation, policy, volatility
└── Oil Data (daily, 2000-2025)
    └── WTI crude oil daily prices

↓

MIDAS Aggregation
├── Daily oil prices → Monthly MIDAS factor (exponential weighting)
└── Result: 5-dimensional monthly time series (2000-2025, 302 observations)

↓

Train-Test Split (65%-35%)
├── Train: 2000-11 to 2017-02 (196 obs)
└── Test: 2017-03 to 2025-12, excluding last 6 months (100 forecast origins)
```

### 2. Lag Order Selection

Use Information Criteria (BIC/AIC) on the training data to select optimal VAR lag order.

```python
# Select lag order p ∈ {1, 2, 3}
# Criterion: BIC (more parsimonious)
optimal_lag = argmin{BIC(p) : p ∈ {1,2,3}}
```

**Result:** Typically p=1 or p=2 for this dataset.

### 3. Rolling/Expanding Window Forecasting

For each forecast origin date `t` in the test set:

```
At date t:
├── Get all data available up to t (from 2000 to t)
├── Determine window:
│   ├── Expanding: Use ALL available data
│   └── Rolling: Use last N months (if specified)
├── Fit VAR(p) model on window data
├── Generate h-step ahead forecasts for h ∈ {1, 3, 6} months
└── Store forecasts
```

**Why Expanding Window?**
- Prevents look-ahead bias: only uses past data
- Maximizes available training data as sample grows
- Allows coefficients to evolve over time (TVP effect)

### 4. Multi-Step Forecasting

The fitted VAR model generates multi-step ahead forecasts using:

```
y_t+h = A_1 * y_t+h-1 + A_2 * y_t+h-2 + ... + A_p * y_t+h-p + ε_t+h
```

Where:
- `A_i`: Time-varying coefficient matrices
- Forecasts are generated recursively (dynamic forecasting)
- Not static: uses predicted values as inputs for future steps

## Key Files

| File | Purpose |
|------|---------|
| `main_midas.py` | Main execution script - orchestrates pipeline |
| `midas_preprocessing.py` | Data loading, MIDAS aggregation, train-test split, lag selection |
| `midas_tvpvar_model.py` | Core MidasTVPVAR class - VAR fitting and forecasting |
| `stats.py` | Forecast evaluation metrics (RMSE, MAE) |

## Usage

```bash
cd s23_Midas
python main_midas.py
```

### Output

Results saved to `results_midas/`:
- `inflation_forecast_metrics.csv` - Inflation forecast performance (RMSE, MAE by horizon)
- `growth_forecast_metrics.csv` - Growth forecast performance
- `inflation_forecasts.csv` - Raw inflation forecasts (100 × 3 horizons)
- `growth_forecasts.csv` - Raw growth forecasts (100 × 3 horizons)

### Example Output

```
Inflation Forecast Metrics:
  horizon  rmse   mae    n_forecasts
1        1   0.224  0.169      100
2        3   0.324  0.206      100
3        6   0.296  0.200      100
```

## Performance Comparison

### MIDAS TVP-VAR vs TVP-VAR (4 macro only)

**MIDAS Results (5 variables including oil):**
- Inflation h=1: RMSE = 0.224, MAE = 0.169
- Inflation h=3: RMSE = 0.324, MAE = 0.206
- Inflation h=6: RMSE = 0.296, MAE = 0.200

**TVP-VAR Results (4 macro only):**
- Inflation h=1: RMSE = 0.178, MAE = 0.131
- Inflation h=3: RMSE = 0.222, MAE = 0.153
- Inflation h=6: RMSE = 0.224, MAE = 0.156

**Observations:**
- MIDAS has higher RMSE at h=1 and h=3
- Possible causes:
  1. **Data limitation**: MIDAS uses only 2000-2025 data (302 obs) vs TVP-VAR's 1989-2025 (444 obs)
  2. **Oil noise**: Oil prices are volatile and may add noise
  3. **Sample size**: Only 100 test forecasts vs TVP-VAR's 150
  4. **Multicollinearity**: Oil might be correlated with existing variables

## Data Availability Constraint

**Critical Issue:** Oil data starts from **2000-08**, while macro data starts from **1989-01**.

```
Timeline:
1989-01 ←─ Macro only ─→ 2000-08 ←─ MIDAS (macro + oil) ─→ 2025-12
         155 observations           302 observations

Result:
- MIDAS can only use 302 observations (2000-2025)
- TVP-VAR can use 444 observations (1989-2025)
- Test forecasts: 100 (MIDAS) vs 150 (TVP-VAR)
```

To get more test samples, would need earlier oil price data (e.g., from FRED or academic sources with reconstructed pre-2000 data).

## Economic Interpretation

### Why Oil Matters for Forecasting

1. **Inflation**: Oil is a major input cost. Oil price ↑ → inflation ↑
2. **Growth**: Oil shocks affect production costs and economic activity
3. **Monetary Policy**: Central banks respond to oil-driven inflation
4. **Volatility**: Oil price volatility is a risk indicator

### Time-Varying Parameters

The model captures evolving relationships:
- 2000s: Oil-growth relationship stronger (commodity super-cycle)
- 2008: Oil shock and financial crisis interaction
- 2010s: Shale revolution changes oil-economy nexus
- 2020: COVID oil collapse
- 2022: Ukraine-driven oil supply shock

## Comparison to Simple Approaches

### Without MIDAS (4-variable VAR):
- Uses monthly growth directly
- Ignores daily oil dynamics
- Misses high-frequency information

### With MIDAS (5-variable VAR):
- Captures daily oil momentum via exponential weighting
- Incorporates high-frequency data into low-frequency model
- More flexible parameterization (θ, K) for aggregation

## Limitations & Future Improvements

1. **Limited Historical Data**: Oil data starts 2000, constrains sample
   - **Fix**: Find reconstructed pre-2000 oil data from academic sources

2. **Oil Noise**: Daily volatility may add forecast noise
   - **Fix**: Try alternative aggregation schemes (MA filter, volatility weighting)

3. **Fixed MIDAS Parameters**: θ=0.03, K=60 chosen ad-hoc
   - **Fix**: Cross-validate θ and K over forecasting performance

4. **Linear VAR Assumption**: Relationships may be nonlinear
   - **Fix**: Consider threshold VAR or machine learning approaches

5. **Single Oil Metric**: WTI crude only
   - **Fix**: Add other commodities or broader energy indices

## Technical Details

### Implementation Notes

- **Language**: Python 3.x (requires statsmodels, pandas, numpy)
- **Time Series Validation**: Walk-forward expanding window prevents look-ahead bias
- **Scalers**: Fit separately on each training window to avoid data leakage
- **Missing Data**: Forward-fill used for oil prices, dropped for macro factors

### Expanding Window vs Rolling Window

**Expanding (Used):**
```
Window at t1: [start_date, ..., t1]
Window at t2: [start_date, ..., t2]
Benefits: More training data as time progresses, captures long-term evolution
```

**Rolling (Alternative):**
```
Window at t1: [t1-N, ..., t1]
Window at t2: [t2-N, ..., t2]
Benefits: Constant training size, captures recent regime only
```

## References & Related Work

**MIDAS Literature:**
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2007): "There is a risk-return tradeoff after all"
- Ghysels, E., Sinko, A., & Valkanov, R. (2007): "MIDAS regressions: Further results and new directions"

**TVP-VAR:**
- Cogley, T., & Sargent, T. J. (2005): "Drifts and volatilities: monetary policies and outcomes"
- Primiceri, G. E. (2005): "Time varying structural vector autoregressions and monetary policy"

**High-Frequency Data in Forecasting:**
- Andreou, E., Ghysels, E., & Kourtellos, A. (2013): "Should macroeconomic forecasters use daily financial data?"

## Contact & Questions

For questions about implementation or results, refer to:
- s21_macro/: TVP-VAR baseline (4 variables)
- This folder (s23_Midas): MIDAS-augmented VAR (5 variables with oil)
