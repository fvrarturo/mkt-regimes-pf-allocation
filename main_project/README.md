# Project Structure

This project is organized into three main sections:

## Section 1: Regression Analysis (`s1_regression/`)

**Purpose**: Full-sample regression analysis and testing of trading strategies based on ERP forecasts from full-sample regression.

**Contents**:
- Full-sample regression models (`regression.py`, `main.py`)
- Trading strategy implementation using full-sample forecasts (`trading_strategy_Full_Sample.ipynb`)
- Results and visualizations (`results/`)

**Key Files**:
- `regression.py`: OLS regression functions for ERP forecasting
- `main.py`: Main orchestration script
- `plotting.py`: Visualization functions
- `trading_strategy_Full_Sample.ipynb`: Trading strategy notebook

---

## Section 2: Regimeness Analysis (`s2_regimeness/`)

**Purpose**: All regimeness analysis and implementation of regimeness in trading strategies (HMM actual-based).

**Contents**:
- **`regimes/`**: Regime detection models
  - `2x2_regimes/`: 2x2 Growth x Inflation regime detection
  - `HMM_regimes/`: Hidden Markov Model regime detection
- **`regressions/`**: Conditional regressions by regime
  - `conditional_regression_all_regimes.py`: Main regression analysis
  - `create_coefficient_table.py`: Coefficient visualization
  - `results/`: Regression results and tables
- **`trading_strategy/`**: HMM-based trading strategy implementation
  - `hmm_forecasts.py`: HMM forecast generation
  - `main.py`: Strategy evaluation orchestration
  - `trading.py`: Trading rule implementation
  - `results/`: Strategy performance results

**Key Files**:
- `regimes/HMM_regimes/hmm_model.py`: HMM model implementation
- `regressions/conditional_regression_all_regimes.py`: Conditional regression analysis
- `trading_strategy/hmm_forecasts.py`: HMM-based ERP forecasting
- `trading_strategy/main.py`: Trading strategy evaluation

---

## Section 3: Forecasting Analysis (`s3_forecasting/`)

**Purpose**: All agentic AI and macro forecasting analysis.

**Contents**:
- **`s21_macro/`**: TVP-VAR econometric forecasting model
- **`s22_ml_based/`**: Machine learning forecasting models
  - `xgboost_model.py`: XGBoost forecasting
  - `lstm_model.py`: LSTM neural network forecasting
- **`s23_Midas/`**: MIDAS (Mixed Data Sampling) forecasting model
- **`cross_comparison/`**: Cross-model forecast comparison
- **`var_new/`**: VAR model experiments

**Key Files**:
- `main.py`: Master orchestration script for all forecasting models
- `s21_macro/tvpvar_model.py`: TVP-VAR implementation
- `s22_ml_based/xgboost_model.py`: XGBoost implementation
- `s22_ml_based/lstm_model.py`: LSTM implementation
- `s23_Midas/midas_tvpvar_model.py`: MIDAS implementation

---

## Old Work (`old_work/`)

Contains deprecated and experimental code:
- `s12_regimeness_old/`: Old regimeness analysis code
- `s13_extremeness/`: Extremeness analysis (deprecated)
- Other experimental and deprecated files

---

## Data (`data/`)

- `macro_final/`: Final processed macro factors
- `macro_processed/`: Processed macro variables
- `macro_processed_full/`: Full set of processed macro variables
- `forecasting data/`: Additional forecasting datasets

---

## Notes

- Each section is self-contained with its own `main.py` orchestration script
- Results are stored in `results/` subdirectories within each section
- Old and deprecated code has been moved to `old_work/` for reference

