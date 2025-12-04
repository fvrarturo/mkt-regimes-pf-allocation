# Section 3: ERP Forecasting and Trading

This section implements non-linear models (XGBoost, LSTM) to forecast ERP (Equity Risk Premium) and evaluates trading strategies based on these forecasts.

## Structure

- `data_loader.py`: Loads ERP, macro, and sentiment data
- `models/`: Model implementations
  - `xgboost_model.py`: XGBoost model for ERP forecasting
  - `lstm_model.py`: LSTM model for ERP forecasting
- `trading.py`: Trading strategy implementation (z-score based weights)
- `performance.py`: Performance metrics computation
- `main.py`: Main orchestration script

## Models

1. **XGBoost**: Gradient boosting model using macro features only
2. **LSTM**: Long Short-Term Memory neural network
3. **XGBoost + Groq Sentiment**: XGBoost with Groq sentiment features
4. **XGBoost + OpenAI Sentiment**: XGBoost with OpenAI sentiment features

## Training Strategy

- Initial training: Data up to 2002-03-31
- Retraining: Annually (every year start)
- Forecasts: Generated for all dates from 2002-03-31 onwards

## Trading Strategy

Uses z-score based weights:
- `weight = 0.5 + 0.25 * zscore(forecast)`
- Clipped between 10% and 90%
- Portfolio: `weight * equity_return + (1 - weight) * bond_return`

## Running

```bash
python main.py
```

## Outputs

- `results/strategy_performance_summary.csv`: Performance metrics for all strategies
- `results/{model}_returns.csv`: Time series of returns, weights, and forecasts for each model
