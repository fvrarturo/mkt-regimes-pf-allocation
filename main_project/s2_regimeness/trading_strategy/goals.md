# Step 3 – Trading Strategy Evaluation

## Objectives
1. Evaluate HMM-based trading strategies using growth + inflation regimes (K=4).
2. Compare different regime determination approaches:
   - Forecast-based: Uses forecasts at T to determine regime mix at T+1
   - Actual-based: Uses actual values at T to determine regime mix
   - Fixed 50/50 benchmark
3. Trade a two-asset portfolio: **S&P 500 vs 3M T-bills** with a Z-score based trading rule.

## Workflow implemented in `main.py`
1. **Load data**
   - Equity & bond monthly returns, ERP = equity – bond
   - All macro variables from `macro_processed_full`
   - Macro forecasts for growth and inflation

2. **Generate HMM-based ERP forecasts**
   - Load HMM model (growth + inflation, K=4) and regression coefficients
   - For each strategy, determine regime probabilities and compute weighted ERP forecasts
   - Apply macro variables at time T to get ERP forecast

3. **Trading rule**
   - Equity weight = `clip(0.5 + 0.25 * zscore(forecast), 10%, 90%)`
   - Portfolio return = `w_t * r_equity + (1 – w_t) * r_bond`
   - Benchmark = 50/50 static allocation

4. **Evaluation outputs**
   - `results/strategy_performance_summary.csv` with Sharpe, vol, drawdown, hit rate, turnover
   - Strategy-specific CSVs (`*_returns.csv`)
   - Cumulative returns plot (`cumulative_returns_all_strategies.png`)
   - Performance comparison plot (`performance_comparison_all_strategies.png`)

## Strategy Descriptions

### hmm_forecast_based
Uses forecasts of growth and inflation at time T to determine regime probabilities at T+1, then applies weighted regression coefficients to macro variables at time T.

### hmm_actual_based
Uses actual values of growth and inflation at time T to determine regime probabilities, then applies weighted regression coefficients to macro variables at time T.

### fixed_50_50_benchmark
Static 50/50 allocation between equity and bonds.
