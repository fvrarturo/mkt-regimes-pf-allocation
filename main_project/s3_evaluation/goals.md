# Step 3 – Trading Strategy Evaluation

## Objectives
1. Translate every ERP forecast (regime, extremeness, baseline regression) into the **same** trading rule.
2. Forecasts update each month: train on history through month *t-1*, predict ERP\_{t+1}.
3. Run accuracy scenarios (40 %, 60 %, 80 %, 100 %) for macro-driven signals to mimic imperfect macro predictions.
4. Trade a two-asset portfolio: **S&P 500 vs 3M T-bills** with a fixed mapping from forecast → equity weight.

## Workflow implemented in `main.py`
1. **Load data**
   - Equity & bond monthly returns, ERP = equity – bond
   - Macro factors (`growth`, `inflation`, `policy`, `vol`)
   - HMM regime probabilities (Growth + Policy model)

2. **Generate ERP forecasts (rolling)**
   - `full_regression`: expanding-window OLS of ERP\_{t+1} on macro factors.
   - `regime_hmm`: probability-weighted conditional means per regime.
   - `extreme_isolation` / `extreme_pca`: classify current macro state via Isolation Forest / PCA distance and use conditional ERP means for extreme vs normal states.

3. **Accuracy scenarios**
   - For regime/extremeness forecasts, flip the forecast sign with probability (1 – accuracy) to simulate 40 %, 60 %, 80 % directional accuracy.
   - Full-sample regression serves as the clean benchmark (100 % scenario).

4. **Trading rule (same for every model)**
   - Equity weight = `clip(0.5 + 0.25 * zscore(forecast), 10 %, 90 %)`
   - Portfolio return = `w_t * r_equity + (1 – w_t) * r_bond`
   - Benchmark = 50/50 static allocation.

5. **Evaluation outputs**
   - `results/strategy_performance_summary.csv` with Sharpe, vol, drawdown, hit rate, turnover.
   - Strategy-specific CSVs (`*_returns.csv`) plus plots (`cumulative_returns_*.png`, `performance_comparison.png`).

## Extensibility
- Drop new forecast series into `base_forecasts` (must be a pd.Series of ERP estimates).
- Adjust accuracy scenarios in `accuracy_levels`.
- Modify the weight rule in `trading.py` to test alternative allocation formulas.
