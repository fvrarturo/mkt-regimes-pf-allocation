# Cross-Model Forecast Comparison Report

## Executive Summary

This report compares forecast performance across four models:
- **TVP-VAR**: Time-Varying Parameter VAR
- **XGBoost (Macro)**: Gradient boosting with macro features only
- **XGBoost (Macro+Sent)**: Gradient boosting with macro and sentiment features
- **LSTM**: Long Short-Term Memory neural network

## Key Findings

### Best Performing Models by Horizon

#### Horizon h=1 months

- **Growth**:
  - Best RMSE: TVP-VAR (0.7940)
  - Best MAE: TVP-VAR (0.2738)
- **Inflation**:
  - Best RMSE: TVP-VAR (0.1780)
  - Best MAE: TVP-VAR (0.1313)

#### Horizon h=3 months

- **Growth**:
  - Best RMSE: XGBoost (Macro+Sent) (1.2988)
  - Best MAE: XGBoost (Macro) (0.5688)
- **Inflation**:
  - Best RMSE: XGBoost (Macro) (0.1951)
  - Best MAE: XGBoost (Macro+Sent) (0.1464)

#### Horizon h=6 months

- **Growth**:
  - Best RMSE: XGBoost (Macro) (1.4087)
  - Best MAE: LSTM (0.6418)
- **Inflation**:
  - Best RMSE: XGBoost (Macro+Sent) (0.2194)
  - Best MAE: TVP-VAR (0.1556)

### Relative Improvements vs TVP-VAR

| Model | Variable | Horizon | RMSE Improvement (%) | MAE Improvement (%) |
|-------|----------|---------|---------------------|-------------------|
| XGBoost (Macro) | Growth | h=1m | -21.12 | -11.33 |
| XGBoost (Macro+Sent) | Growth | h=1m | -23.17 | -15.23 |
| LSTM | Growth | h=1m | -84.35 | -153.33 |
| XGBoost (Macro) | Growth | h=3m | 33.78 | 18.24 |
| XGBoost (Macro+Sent) | Growth | h=3m | 33.89 | 17.84 |
| LSTM | Growth | h=3m | 27.08 | 4.10 |
| XGBoost (Macro) | Growth | h=6m | 22.74 | 11.75 |
| XGBoost (Macro+Sent) | Growth | h=6m | 22.17 | 11.56 |
| LSTM | Growth | h=6m | 22.38 | 14.88 |
| XGBoost (Macro) | Inflation | h=1m | -6.46 | -3.66 |
| XGBoost (Macro+Sent) | Inflation | h=1m | -3.69 | -2.41 |
| XGBoost (Macro) | Inflation | h=3m | 12.02 | 4.28 |
| XGBoost (Macro+Sent) | Inflation | h=3m | 10.46 | 4.35 |
| XGBoost (Macro) | Inflation | h=6m | -9.94 | -8.48 |
| XGBoost (Macro+Sent) | Inflation | h=6m | 1.99 | -1.87 |

### Performance Table

| Model | Variable | Horizon | RMSE | MAE |
|-------|----------|---------|------|-----|
| TVP-VAR | Growth | h=1m | 0.7940 | 0.2738 |
| TVP-VAR | Growth | h=3m | 1.9646 | 0.6957 |
| TVP-VAR | Growth | h=6m | 1.8234 | 0.7539 |
| TVP-VAR | Inflation | h=1m | 0.1780 | 0.1313 |
| TVP-VAR | Inflation | h=3m | 0.2218 | 0.1530 |
| TVP-VAR | Inflation | h=6m | 0.2238 | 0.1556 |
| XGBoost (Macro) | Growth | h=1m | 0.9617 | 0.3049 |
| XGBoost (Macro) | Growth | h=3m | 1.3011 | 0.5688 |
| XGBoost (Macro) | Growth | h=6m | 1.4087 | 0.6653 |
| XGBoost (Macro) | Inflation | h=1m | 0.1895 | 0.1362 |
| XGBoost (Macro) | Inflation | h=3m | 0.1951 | 0.1465 |
| XGBoost (Macro) | Inflation | h=6m | 0.2461 | 0.1688 |
| XGBoost (Macro+Sent) | Growth | h=1m | 0.9780 | 0.3155 |
| XGBoost (Macro+Sent) | Growth | h=3m | 1.2988 | 0.5715 |
| XGBoost (Macro+Sent) | Growth | h=6m | 1.4191 | 0.6667 |
| XGBoost (Macro+Sent) | Inflation | h=1m | 0.1845 | 0.1345 |
| XGBoost (Macro+Sent) | Inflation | h=3m | 0.1986 | 0.1464 |
| XGBoost (Macro+Sent) | Inflation | h=6m | 0.2194 | 0.1585 |
| LSTM | Growth | h=1m | 1.4637 | 0.6937 |
| LSTM | Growth | h=3m | 1.4327 | 0.6671 |
| LSTM | Growth | h=6m | 1.4152 | 0.6418 |


## Conclusions

1. **Short-term forecasts (h=1)**: TVP-VAR performs best for both GDP and inflation.
2. **Medium-term forecasts (h=3,6)**: XGBoost models show improvements over TVP-VAR.
3. **Sentiment impact**: Adding sentiment features provides marginal improvements.
4. **LSTM performance**: LSTM shows competitive performance at longer horizons.
