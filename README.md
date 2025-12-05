# Dynamic Macro Forecasting for Active Allocation

This project asks a central question:

> **Can macroeconomic data and news-based sentiment be used to forecast the U.S. equity risk premium (ERP) and improve a dynamic allocation between equities and bonds?** :contentReference[oaicite:0]{index=0}  

We study this through three complementary empirical blocks:

1. **Full-Sample Linear Models** – classic macro predictive regressions and their use in a simple timing strategy.  
2. **Regime-Based Models** – conditioning ERP forecasts on macro regimes estimated either with simple thresholds or Hidden Markov Models.  
3. **Agentic AI + Machine Learning** – building macro sentiment with LLM “agents” and feeding it into LSTM/XGBoost models for ERP forecasting and allocation.

---

## Data & Setup

- **Target variable – Equity Risk Premium (ERP)**  
  - Monthly excess return of the S&P 500 over the 3-month U.S. T-bill.  
  - S&P 500: geometric monthly return from month-end index levels.  
  - T-bill: convert daily annualized yields into daily simple returns and compound within each month to get the risk-free return. :contentReference[oaicite:1]{index=1}  

- **Macro predictors**  
  - 15+ monthly macro variables across: inflation, real activity, labor market, money and credit, term structure, and financial conditions (e.g., industrial production, inventories, unemployment, M2, term spread, NFCI).  

- **Sample**  
  - 1992-03 to 2025-08, monthly frequency (~355 observations).  

- **Evaluation framework**  
  - Out-of-sample expanding window.  
  - Minimum 120 months of history before the first forecast.  
  - All predictors standardized *within the training window* to avoid look-ahead bias.  

- **Allocation rule (used throughout the project)**  
  1. Forecast next-month ERP: $\widehat{ERP}_{t+1}$.  
  2. Compute a **real-time Z-score** vs historical ERPs up to $t-1$:  
     $$
     Z_t = \frac{\widehat{ERP}_{t+1} - \mu_{t-1}}{\sigma_{t-1}}
     $$
  3. Map $Z_t$ into an equity weight using a **banded rule**, with weights bounded between 25% and 75% equity (symmetric around ~50% benchmark).  
  4. Benchmark: a constant-mix portfolio holding the same average equity weight as the strategy. :contentReference[oaicite:2]{index=2}  

---

## Block 1 – Full-Sample Linear Models

### Goal

Use standard macro predictive regressions to forecast the ERP and see whether they justify dynamic allocation versus a constant-mix portfolio.

### Methods

1. **Full OLS model**

   $$
   ERP_{t+1} = \beta_0 + \beta_1 X_{1,t} + \dots + \beta_p X_{p,t} + \varepsilon_{t+1}
   $$

   - Uses **all** macro predictors simultaneously.  

2. **LASSO + refit**

   - LASSO is used for **variable selection and shrinkage**:
     $$
     \min_\beta \sum_t \left( ERP_{t+1} - \beta_0 - \sum_j \beta_j X_{j,t} \right)^2
     + \lambda \sum_j |\beta_j|
     $$
   - For each window, we:
     1. Run LASSO to select variables.  
     2. Refit OLS on the selected subset for interpretability.  

   - Frequently selected predictors: industrial production, unemployment, inventories, real M2, 10y–2y term spread, NFCI. :contentReference[oaicite:3]{index=3}  

### Findings

- **In-sample**  
  - R² and adjusted R² are **very low**; ERP is hard to explain at the monthly horizon.  
  - Only a few variables (notably unemployment) show consistent significance.

- **Out-of-sample**  
  - Both full OLS and LASSO models **underperform a historical-mean benchmark** (negative OOS R²).  
  - Linear macro models **cannot reliably forecast monthly ERP**, but they still generate signals that can be used in allocation.

- **Allocation**  
  - When plugged into the banded allocation rule, both models achieve reasonable Sharpe ratios and risk-adjusted returns, but benefits are modest.  
  - This motivates **richer nonlinear and state-dependent approaches**.

---

## Block 2 – Regime Definition & Conditional Regressions

### Goal

Capture the idea that ERP is **state-dependent**: macro variables might only be predictive **within certain regimes** (e.g., stagflation vs Goldilocks). We want to:

- Define macro regimes.  
- Estimate ERP–macro relationships **within** each regime.  
- Use regime-conditional forecasts in a dynamic allocation strategy.

### 2.1 Regime Construction

We build two types of regime frameworks:

#### A. Simple 2×2 Growth/Inflation Matrix

- Define **growth** and **inflation** as “high” or “low” relative to rolling medians.  
- This yields four intuitive quadrants: :contentReference[oaicite:4]{index=4}  
  - **Goldilocks** – High growth / Low inflation  
  - **Overheating** – High growth / High inflation  
  - **Stagflation** – Low growth / High inflation  
  - **Slowdown** – Low growth / Low inflation  

**Pros**

- Transparent, easily explainable.  

**Cons**

- Hard thresholds → discrete jumps in regime labels.  
- Cannot represent ambiguous states (borderline periods).

#### B. Hidden Markov Model (HMM)

- We treat regimes as **latent states**.  
- Inputs: macro variables (focus on growth + inflation after model selection).  
- The HMM learns:
  - State-specific means and covariances.  
  - Transition probabilities between regimes.  
- Output: **soft probabilities** $P(\text{regime } r | \text{data}_t)$ for each date. :contentReference[oaicite:5]{index=5}  

**Model selection**

- We test many variable combinations and numbers of states $K$.  
- Use AIC and BIC to trade off fit vs overfitting.  
- Best configuration: **Growth + Inflation with 4 regimes**.

### 2.2 Regime-Conditional Regressions

For each regime $r$, we estimate:

$$
ERP_{t+1} = \alpha_r + \beta_r^\top X_t + \varepsilon_{t+1}
$$

- In the **2×2 model**, each observation belongs to a single regime.  
- In the **HMM**, each observation contributes to *all* regimes with a weight:
  $$
  w_{r,t} = P(\text{regime } r | \text{data}_t)
  $$
  so ambiguous periods are handled smoothly. :contentReference[oaicite:6]{index=6}  

We then:

- Compare which macro variables matter most **by regime**.  
- Use regime-specific coefficients to forecast ERP.

### 2.3 Trading Strategy

- Start with 10 years of history.  
- Each month:
  1. Estimate or update regime model (2×2 or HMM).  
  2. Estimate regime-conditional regressions using data up to $t-1$.  
  3. Forecast $ERP_{t+1}$ using:
     - **2×2**: coefficients of the active regime at time $t$.  
     - **HMM**: a **probability-weighted mixture** of all regime regressions.  
  4. Translate forecast into equity weight via the same Z-score banded rule. :contentReference[oaicite:7]{index=7}  

### Results

- Both **2×2 and HMM strategies outperform their constant-mix benchmarks** in risk-adjusted terms.  
- **2×2 regimes** achieve slightly **higher Sharpe ratios** but show higher volatility due to abrupt regime switches.  
- **HMM** produces smoother allocations and lower volatility but sometimes exhibits state “stickiness”.  

**Takeaway:**  
> **ERP is more predictable when conditional on macro regimes.** Simple, interpretable regimes already deliver meaningful improvements.

---

## Block 3 – Agentic AI & Machine Learning Models

### Goal

Augment macro variables with **news-based macro sentiment** and use **nonlinear ML models** to capture complex, persistent patterns in ERP. We:

1. Build macro sentiment from 100k+ Reuters articles using an **Agentic AI** framework.  
2. Feed sentiment + macro variables into **LSTM** and **XGBoost** forecasters.  
3. Compare allocation performance with and without sentiment.

### 3.1 Agentic AI Sentiment Pipeline

**Data**

- Source: Dow Jones Factiva, Reuters only.  
- Period: 1990–2025.  
- Filter: macro-relevant news (policy, inflation, growth, financial stability), exclude firm-specific headlines.  

**Three-Agent System** :contentReference[oaicite:8]{index=8}  

1. **News Analyst Agent (A1)**  
   - Reads and summarizes macro news each month.  
   - Assigns four sentiment scores in [-1, +1]:  
     - Growth  
     - Inflation  
     - Monetary policy stance  
     - Market volatility  
   - Aggregates article-level scores into daily and then monthly signals.

2. **Fact-Checker Agent (A2)**  
   - Reviews the time series of scores for:
     - Consistency  
     - Scale drift  
     - Bias  
   - Can approve, adjust, or reject A1's output.  

3. **Manager Agent (A3)**  
   - Resolves disagreements between A1 and A2.  
   - Ensures convergence and prevents infinite back-and-forth.  

The result is a **35-year time series of robust macro sentiment scores** aligned monthly with the macro dataset.

**Code Implementation** (`data/agenticAI/llm_text/`):

- **`sentiment_agents.py`**: 
  - `SentimentAnalyzerAgent`: Analyzes news and assigns sentiment scores
    - Reads weekly news articles
    - Produces four sentiment scores: Growth, Inflation, Monetary Policy, Market Volatility
    - Implements smoothing guidelines to prevent dramatic week-to-week jumps
  - `FactCheckerAgent`: Validates scores for consistency and smoothness
    - Checks week-to-week change magnitude (rejects jumps > 0.5 without strong justification)
    - Verifies economic logic and temporal consistency
    - Can approve, reject, or suggest corrections
  - `CoordinatorAgent`: Orchestrates the three-agent workflow
    - Manages iteration between analyzer and fact-checker
    - Ensures convergence and prevents infinite loops
    - Processes weeks chronologically
  - Implements iterative refinement until convergence
  - Handles LLM API calls (OpenAI, Groq) with error handling and retries
  - Uses Groq-compatible output schema for JSON parsing

- **`main.py`**: 
  - Main orchestration script for sentiment generation
  - Processes news articles week-by-week in chronological order
  - Aggregates daily scores to monthly frequency
  - Saves intermediate results and final sentiment scores
  - Handles API rate limits and key rotation

- **`llm_config.py`**: 
  - Configuration for LLM providers (API keys, models, parameters)
  - Supports both OpenAI and Groq backends
  - Implements API key rotation for Groq to handle rate limits

- **Output**: 
  - `sentiment_scores.csv`: Monthly sentiment scores for all four dimensions
  - Scores range from -1 (negative) to +1 (positive)
  - Aligned monthly with macro dataset dates

### 3.2 Machine Learning Models

We consider two model classes:

#### A. LSTM (Long Short-Term Memory Networks)

- Sequence model that uses the **history of macro + sentiment features** to predict next-month ERP.  
- Well suited because:
  - ERP reflects **persistent macro forces**.  
  - LSTM captures **long-range dependencies** and **nonlinear interactions**.  
  - Rolling retraining lets it adapt to new regimes over time. :contentReference[oaicite:9]{index=9}  

#### B. XGBoost (Gradient Boosted Trees)

- Ensemble of decision trees trained sequentially to minimize forecast error plus regularization.  
- Naturally captures:
  - Nonlinearities  
  - Interaction terms (e.g., Growth × Inflation, Policy × Volatility)  
  - Mixed feature types (macro + sentiment). :contentReference[oaicite:10]{index=10}  

**Feature sets for both models**

1. Macro-only baseline.  
2. Macro + sentiment (OpenAI-based).  
3. Macro + sentiment (Groq-based).  

This allows us to isolate the **incremental value of sentiment** and compare LLM providers.

### 3.3 Training & Allocation

- Rolling 10-year training window, monthly retraining.  
- Forecast horizon: 1 month ahead.  
- Out-of-sample evaluation over ~15 years.  
- Overfitting control:  
  - Regularization + early stopping (validation split).  
- Allocation: same Z-score banded rule as in previous blocks, benchmarked to matching constant-mix portfolio. :contentReference[oaicite:11]{index=11}  

### Results

- **All ML models beat the benchmark** in risk-adjusted terms.  
- Adding sentiment **improves performance**, particularly by:
  - Reducing drawdowns.  
  - Lowering volatility.  
- **Groq sentiment + XGBoost** delivers the **highest Sharpe ratio** and the best drawdown profile among ML models. :contentReference[oaicite:12]{index=12}  

**Takeaway:**  
> **Combining macro data with LLM-derived sentiment and nonlinear ML models provides a tangible edge for dynamic allocation.**

---

## Overall Conclusions

1. **Full-sample linear models**  
   - Macro variables have **weak but non-zero** predictive power for the monthly ERP.  
   - Standalone forecasts are fragile, but even weak signals can be useful when translated into **allocation rules**.

2. **Regime-based models**  
   - ERP predictability is **state-dependent**.  
   - Conditioning on macro regimes (either simple 2×2 or HMM) materially improves risk-adjusted performance.  
   - There is a trade-off between **interpretability and smoothness** (2×2 vs HMM).

3. **Agentic AI + ML**  
   - LLM-based macro sentiment is a **meaningful, additive signal**.  
   - LSTM and XGBoost capture **nonlinear, persistent dynamics** and outperform simpler models.  
   - The best strategies combine **macro data + robust sentiment + flexible ML architecture**.

4. **Portfolio implications**  
   - Even in a difficult forecasting problem like the ERP, carefully designed models and signals can improve dynamic allocation.  
   - Regimes and sentiment help **stabilize performance**, especially by reducing drawdowns.  

---

## Project Structure & Code Organization

### Directory Overview

The project is organized into three main empirical sections, each with its own directory:

```
main_project/
├── data/                          # All data sources
│   ├── macro_final/               # Final macro factors (growth, inflation, policy, volatility)
│   ├── macro_processed/           # Processed market data (SP500, T-bills, ERP)
│   ├── macro_processed_full/      # Full set of 15+ macro variables by category
│   ├── forecasting data/          # Additional forecasting datasets
│   └── agenticAI/                 # Agentic AI sentiment pipeline code and outputs
│
├── s1_regression/                 # Block 1: Full-Sample Linear Models
│   ├── main.py                    # Main orchestration script
│   ├── regression.py              # OLS and LASSO regression implementations
│   ├── plotting.py                # Visualization functions
│   └── results/                   # Regression outputs (coefficients, R², rankings)
│
├── s2_regimeness/                 # Block 2: Regime-Based Models
│   ├── regimes/                   # Regime detection implementations
│   │   ├── 2x2_regimes/           # Hard-threshold 2×2 Growth×Inflation regimes
│   │   └── HMM_regimes/           # Hidden Markov Model implementations
│   ├── regressions/               # Conditional regression analysis
│   │   ├── main.py                # Main regression orchestration
│   │   ├── conditional_regression_all_regimes.py  # Core regression logic
│   │   └── create_coefficient_table.py            # Visualization
│   └── trading_strategy/          # Trading strategy implementation
│       ├── main.py                # Main trading strategy script
│       ├── main_lasso.py          # LASSO-based conditional regression strategy
│       ├── hmm_forecasts.py       # HMM-based ERP forecasting
│       ├── two_by_two_forecasts.py # 2×2 regime-based forecasting
│       ├── lasso_conditional_forecasts.py  # LASSO with monthly retraining
│       ├── trading.py             # Trading rule implementation (Z-score)
│       ├── performance.py         # Performance metrics
│       └── plotting.py            # Strategy visualization
│
├── s3_forecasting/                # Block 3: Agentic AI + Machine Learning
│   ├── main.py                    # Main orchestration script
│   ├── data_loader.py             # Data loading utilities
│   ├── models/                    # ML model implementations
│   │   ├── xgboost_model.py      # XGBoost forecaster with feature engineering
│   │   └── lstm_model.py          # LSTM sequence model
│   ├── trading.py                 # Trading strategy (Z-score based)
│   ├── performance.py             # Performance evaluation
│   ├── plotting.py                # Visualization functions
│   ├── news_data/                 # Sentiment scores (Groq, OpenAI)
│   └── results/                   # Model outputs and performance metrics
│
└── old_work/                      # Legacy code and exploratory work
    ├── hailey testing/            # Modular scripts from notebooks
    └── tables_slides/             # LaTeX-style table generation for presentations
```

### Code Organization by Block

#### Block 1: Full-Sample Linear Models (`s1_regression/`)

**Purpose**: Establish baseline predictive power of macro variables using standard linear methods.

**Key Modules**:
- **`regression.py`**: 
  - `run_full_sample_regressions()`: Runs OLS regressions across multiple horizons
  - `run_lasso_regression()`: LASSO variable selection with cross-validation
  - Handles standardization, expanding windows, and statistical inference
- **`plotting.py`**: 
  - Generates coefficient tables, R² plots, variable importance rankings
- **`main.py`**: 
  - Orchestrates the full analysis pipeline
  - Loads data, runs regressions, generates outputs

**Data Flow**:
1. Load ERP and macro factors from `data/macro_final/`
2. Align dates and standardize predictors
3. Run expanding-window regressions (minimum 120 months)
4. Extract coefficients, t-stats, R², variable importance
5. Save results to `results/`

**Outputs**:
- `regression_results_all_horizons.csv`: Complete regression results
- `variable_importance_ranking.csv`: Variables ranked by predictive power
- Coefficient comparison plots and R² visualizations

---

#### Block 2: Regime-Based Models (`s2_regimeness/`)

**Purpose**: Estimate ERP–macro relationships conditional on economic regimes.

**Subdirectories**:

**A. Regime Detection (`regimes/`)**

- **`2x2_regimes/`**:
  - `regime_definitions.py`: `RegimeDefinitions` class for hard-threshold classification
  - `main.py`: Generates regime assignments and statistics
  - Uses median thresholds for growth and inflation factors
  - Outputs: regime assignments, statistics, visualizations

- **`HMM_regimes/`**:
  - `hmm_model.py`: `HMMRegimeModel` class implementing Gaussian HMM
  - `main.py`: Systematic model selection across variable combinations and K values
  - `plot_3d_all_k.py`: AIC/BIC visualization across models
  - Features: regularization for regime separation, soft probability assignments
  - Outputs: Best model selection, regime probabilities, transition matrices

**B. Conditional Regressions (`regressions/`)**

- **`conditional_regression_all_regimes.py`**:
  - `ConditionalRegressionAnalyzer` class
  - Implements **weighted regressions** using regime probabilities as weights
  - For HMM: Each observation contributes to all regimes with probability weights
  - For 2×2: Hard assignments (weight = 1 for active regime, 0 otherwise)
  - Computes weighted R², RMSE, effective sample size
  - Handles 15 macro variables from `macro_processed_full/`

- **`create_coefficient_table.py`**:
  - Generates comprehensive coefficient tables (PNG + CSV)
  - Shows coefficients, t-stats, R² by regime
  - Highlights regimes with low average weight

- **`main.py`**: Orchestrates regression analysis and visualization

**C. Trading Strategies (`trading_strategy/`)**

- **`hmm_forecasts.py`**:
  - `load_hmm_model_and_coefficients()`: Loads trained HMM and regression coefficients
  - `get_regime_probabilities()`: Computes regime probabilities from macro factors
  - `compute_weighted_erp_forecast()`: Weighted ERP forecast using regime probabilities
  - `strategy_actual_based()`: Uses actual macro values at time T
  - Handles standardization and date alignment

- **`two_by_two_forecasts.py`**:
  - `load_2x2_regime_definitions_and_coefficients()`: Loads 2×2 thresholds and coefficients
  - `compute_erp_forecast_hard_regime()`: ERP forecast using active regime coefficients
  - `strategy_actual_based()`: 2×2 regime-based forecasting

- **`lasso_conditional_forecasts.py`**:
  - `LassoConditionalForecaster` class
  - Trains LASSO regressions conditional on regimes with monthly retraining
  - **HMM approach**: Weighted LASSO regressions using regime probabilities as sample weights
  - **2×2 approach**: Separate LASSO regressions for each hard regime assignment
  - Initial 10-year training period, then monthly retraining
  - Uses LassoCV for automatic alpha (regularization) selection via cross-validation
  - Tracks variable inclusion over time for visualization
  - Handles untrained regimes with constant forecasts (last known forecast)
  - Standardizes features within training windows to avoid look-ahead bias

- **`plot_lasso_variables.py`**:
  - `plot_hmm_variable_inclusion_weighted()`: Visualizes HMM variable inclusion weighted by regime probabilities (shades of blue)
  - `plot_variable_inclusion_over_time()`: Visualizes 2×2 variable inclusion (binary: light/dark blue)
  - Shows which macro variables are selected as predictors over time
  - Helps understand how variable importance evolves across regimes

- **`trading.py`**:
  - `forecast_to_weights()`: Converts ERP forecasts to equity weights using Z-score
  - `run_trading_strategy()`: Computes portfolio returns with configurable weight bounds

- **`performance.py`**: 
  - `compute_performance_metrics()`: Sharpe ratio, volatility, drawdown, etc.
  - `compute_turnover()`: Portfolio turnover calculation
  - `compute_hit_rate()`: Forecast accuracy metrics

- **`plotting.py`**:
  - `plot_cumulative_returns_all_strategies()`: Cumulative returns visualization
  - `plot_performance_comparison()`: Performance metrics bar charts
  - `plot_weights_over_time()`: Portfolio weights and regime probabilities over time

- **`main.py`**: 
  - Orchestrates HMM and 2×2 strategy evaluation
  - Selects best HMM model (by annualized return)
  - Generates performance summaries and visualizations

**Data Flow**:
1. Load macro factors and market data
2. Detect regimes (2×2 or HMM) using macro factors
3. Estimate regime-conditional regressions (weighted for HMM, hard for 2×2)
4. Generate ERP forecasts using regime probabilities/thresholds
5. Convert forecasts to portfolio weights via Z-score rule
6. Compute returns and performance metrics
7. Generate visualizations and save results

**Outputs**:
- `conditional_regression_results_all.csv`: All regression coefficients by regime
- `coefficient_table_comprehensive.png`: Visual coefficient table
- `strategy_performance_summary.csv`: Performance metrics for all strategies
- `cumulative_returns_all_strategies.png`: Cumulative returns plot
- `weights_over_time.png`: Portfolio weights and regime evolution
- `hmm_strategy_rankings.csv`: HMM model rankings

---

#### Block 3: Agentic AI + Machine Learning (`s3_forecasting/`)

**Purpose**: Forecast ERP using nonlinear ML models enhanced with LLM-derived sentiment.

**Key Modules**:

- **`data_loader.py`**:
  - `load_erp_data()`: Loads ERP from `equity_risk_pr.csv`
  - `load_market_data()`: Loads equity and bond returns
  - `load_macro_features()`: Loads 15 macro variables (same as conditional regressions)
  - `load_sentiment_groq()` / `load_sentiment_openai()`: Load sentiment scores

- **`models/xgboost_model.py`**:
  - `XGBoostERPForecaster` class
  - **Feature Engineering**:
    - Lagged features (1-6 months)
    - Rolling statistics (mean, std, z-score over multiple windows)
    - Momentum features (differences, percentage changes)
    - Forward-fills missing values before feature creation
  - **Training**:
    - Time-series cross-validation for hyperparameter tuning
    - Time-decay weights (exponentially favor recent data)
    - Early stopping with validation split
    - Monthly retraining (configurable)
  - **Forecasting**:
    - Rolling window forecasts with periodic retraining
    - Handles missing data gracefully

- **`models/lstm_model.py`**:
  - `LSTMerpForecaster` class
  - **Sequence Preparation**:
    - Creates sequences of historical features (default: 12 months)
    - Handles sentiment data integration
    - Properly filters data for training vs. prediction
  - **Architecture**:
    - LSTM layers with dropout regularization
    - Dense output layer
    - Early stopping and validation monitoring
  - **Training**:
    - Monthly retraining (configurable)
    - Validation split for early stopping

- **`trading.py`**:
  - Same Z-score based allocation rule as Block 2
  - `forecast_to_weights()`: Maps forecasts to equity weights (10%-90% bounds)
  - `run_trading_strategy()`: Computes portfolio returns

- **`performance.py`**: 
  - Same performance metrics as Block 2
  - Computes Sharpe, volatility, drawdown, hit rate, turnover

- **`plotting.py`**:
  - `plot_cumulative_returns_all_strategies()`: Includes fixed portfolio benchmarks
  - `plot_performance_comparison()`: Performance comparison with benchmarks
  - Saves performance data to CSV

- **`main.py`**:
  - Orchestrates all ML models (XGBoost, LSTM, XGBoost+Groq, XGBoost+OpenAI)
  - Handles monthly retraining
  - Generates forecasts, runs trading strategies, evaluates performance

**Data Flow**:
1. Load ERP, macro features, and sentiment data
2. Align all data to common monthly frequency
3. For each model:
   - Create features (with engineering for XGBoost)
   - Train initial model on 10 years of data
   - Generate rolling forecasts with monthly retraining
4. Convert forecasts to portfolio weights
5. Compute returns and performance metrics
6. Generate visualizations and save results

**Outputs**:
- `{model}_returns.csv`: Time series for each model (returns, weights, forecasts)
- `strategy_performance_summary.csv`: Performance metrics
- `cumulative_returns_all_strategies.png`: Cumulative returns with benchmarks
- `performance_comparison_all_strategies.png`: Performance bar charts

---

### Data Dependencies

**Core Data Files**:
- `data/macro_final/final_macro.csv`: Four macro factors (growth, inflation, monetary policy, market volatility)
- `data/macro_processed/sp500_processed.csv`: S&P 500 monthly returns
- `data/macro_processed/3m_yield_processed.csv`: 3-month T-bill yields
- `data/macro_processed/equity_risk_pr.csv`: Pre-computed ERP series
- `data/macro_processed_full/`: 15+ individual macro variables organized by category
  - `ec_growth/`: Industrial production, retail sales, inventories, unemployment, etc.
  - `inflation/`: CPI, PCE, PPI
  - `mkt_vol/`: NFCI, term spread, VIX (monthly aggregated)
  - `mon_policy/`: Treasury rates, Fed funds, discount rate, M2
- `data/agenticAI/llm_text/sentiment_scores.csv`: LLM-derived sentiment scores
- `s3_forecasting/news_data/sentiment_groq.csv`: Groq-based sentiment
- `s3_forecasting/news_data/sentiment_openai.csv`: OpenAI-based sentiment

**Data Processing**:
- All macro variables are resampled to monthly frequency (month-end)
- Missing values are forward-filled where appropriate
- Standardization is performed within training windows to avoid look-ahead bias
- Date alignment ensures consistent monthly frequency across all datasets

---

### Running the Code

#### Block 1: Full-Sample Linear Models
```bash
cd main_project/s1_regression
python main.py
```

#### Block 2: Regime-Based Models

**Step 1: Run Conditional Regressions**
```bash
cd main_project/s2_regimeness/regressions
python main.py
```

**Step 2: Evaluate Trading Strategies**
```bash
cd main_project/s2_regimeness/trading_strategy
python main.py                    # HMM and 2×2 strategies
python main_lasso.py              # LASSO conditional regression strategies
```

#### Block 3: ML Forecasting Models
```bash
cd main_project/s3_forecasting
python main.py
```

**Note**: Block 3 requires sentiment data. If sentiment files are missing, the script will run macro-only models.

---

### Key Design Patterns

1. **Expanding Window Training**: All models use expanding windows to avoid look-ahead bias
2. **Standardization**: Predictors are standardized within each training window
3. **Modular Design**: Each block is self-contained with clear interfaces
4. **Consistent Trading Rule**: All strategies use the same Z-score based allocation rule
5. **Benchmarking**: Each strategy is compared to its own constant-mix benchmark (average weight)
6. **Error Handling**: Graceful handling of missing data and edge cases
7. **Visualization**: Consistent plotting style across all blocks

---

### Code Quality & Maintenance

- **Type Hints**: Functions include type annotations for clarity
- **Docstrings**: All modules and key functions are documented
- **Path Management**: Uses `pathlib.Path` for cross-platform compatibility
- **Error Messages**: Descriptive error messages for debugging
- **Results Organization**: All outputs saved to `results/` subdirectories
- **Version Control**: Git-friendly structure with `.gitignore` for outputs

---

## Quick Reference: Running the Complete Analysis

### Prerequisites

1. **Data Setup**: Ensure all data files are in place:
   - Macro factors: `data/macro_final/final_macro.csv`
   - Market data: `data/macro_processed/sp500_processed.csv`, `3m_yield_processed.csv`
   - Macro variables: `data/macro_processed_full/` (all subdirectories)
   - Sentiment (optional): `s3_forecasting/news_data/sentiment_*.csv`

2. **Dependencies**: Install required packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn scipy hmmlearn yfinance
   ```

### Execution Order

**1. Block 1: Baseline Linear Models**
```bash
cd main_project/s1_regression
python main.py
```
**Outputs**: Regression coefficients, R², variable importance rankings

**2. Block 2: Regime-Based Analysis**

**Step 2a: Run Conditional Regressions**
```bash
cd main_project/s2_regimeness/regressions
python main.py
```
**Outputs**: Regime-specific coefficients, R² by regime, coefficient tables

**Step 2b: Evaluate Trading Strategies**
```bash
cd main_project/s2_regimeness/trading_strategy
python main.py                    # HMM and 2×2 strategies
python main_lasso.py              # LASSO conditional regression (optional)
```
**Outputs**: Strategy returns, performance metrics, cumulative returns plots

**3. Block 3: ML Forecasting**
```bash
cd main_project/s3_forecasting
python main.py
```
**Outputs**: ML model forecasts, strategy returns, performance comparisons

### Expected Runtime

- **Block 1**: ~1-2 minutes
- **Block 2 (Regressions)**: ~5-10 minutes (HMM fitting is computationally intensive)
- **Block 2 (Trading)**: ~2-3 minutes
- **Block 3**: ~30-60 minutes (ML model training with monthly retraining)

### Output Locations

All results are saved in `results/` subdirectories within each block:
- `s1_regression/results/`: Regression outputs
- `s2_regimeness/regressions/results/`: Conditional regression results
- `s2_regimeness/trading_strategy/results/`: Trading strategy outputs
- `s3_forecasting/results/`: ML model outputs and performance

---

## Suggested High-Level Narrative

When presenting the project, you can structure the story as:

1. **Motivation** – ERP is noisy and hard to forecast, but macro variables and news may contain weak predictive signals that matter for asset allocation.  
2. **Baseline** – Show that full-sample linear models struggle, motivating richer approaches.  
3. **Regimes** – Argue that macro states matter; show that regime-conditional models improve performance.  
4. **Agentic AI + ML** – Introduce the sentiment pipeline and ML models as a way to extract more signal from macro news and nonlinear patterns.  
5. **Conclusion** – Emphasize how combining economic structure (regimes) with modern tools (LLMs + ML) yields the best allocation performance.