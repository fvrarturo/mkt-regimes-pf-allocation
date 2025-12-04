# Section 2: Macro Forecasting - Master Pipeline

## Overview

This directory contains the master orchestration script (`main.py`) that runs all forecasting models to predict **Growth (Industrial Production)** and **Inflation** at horizons h ∈ {1, 3, 6} months.

## Execution

Run the master script to execute all forecasting models:

```bash
cd main_project/s2_forecasts
python main.py
```

Or from the project root:

```bash
python main_project/s2_forecasts/main.py
```

## Pipeline Components

The master script runs the following models in sequence:

### 1. TVP-VAR (Time-Varying Parameter VAR)
- **Location**: `s21_macro/main.py`
- **Description**: Baseline 4-variable TVP-VAR model
- **Variables**: Growth, Inflation, Policy (Fed Funds Rate), Volatility (VIX)
- **Outputs**: 
  - Forecast metrics (RMSE, MAE)
  - Time-varying coefficients
  - Impulse response functions

### 2. XGBoost (Gradient Boosting)
- **Location**: `s22_ml_based/main.py`
- **Description**: Gradient boosting models with macro features
- **Variants**:
  - Macro-only features
  - Macro + Sentiment features
- **Outputs**:
  - Forecast metrics (RMSE, MAE)
  - Feature importance plots
  - Forecast comparison plots

### 3. LSTM (Long Short-Term Memory)
- **Location**: `s22_ml_based/lstm_main.py`
- **Description**: Neural network sequence model
- **Features**: Multivariate LSTM with joint prediction
- **Outputs**:
  - Forecast metrics (RMSE, MAE)
  - Learning curves
  - Forecast vs realized plots

### 4. MIDAS TVP-VAR (Mixed Data Sampling)
- **Location**: `s23_Midas/main_midas.py`
- **Description**: TVP-VAR augmented with daily oil prices via MIDAS
- **Features**: Combines monthly macro factors with high-frequency oil data
- **Outputs**:
  - Forecast metrics (RMSE, MAE)
  - Forecast series

### 5. Cross-Model Comparison
- **Location**: `cross_comparison/main.py`
- **Description**: Compares all models and generates summary reports
- **Outputs**:
  - Performance comparison tables
  - Statistical tests (Diebold-Mariano)
  - Visualizations
  - Summary report

## Output Structure

After running the master script, results are organized as follows:

```
s2_forecasts/
├── s21_macro/results/
│   ├── growth_forecast_metrics.csv
│   ├── inflation_forecast_metrics.csv
│   └── forecast_performance_table.csv
├── s22_ml_based/results/
│   ├── xgboost/
│   │   ├── growth_factor_metrics_xgboost.csv
│   │   ├── inflation_factor_metrics_xgboost.csv
│   │   └── feature_importance_*.png
│   └── lstm/
│       ├── growth_factor_metrics_lstm.csv
│       ├── inflation_factor_metrics_lstm.csv
│       └── learning_curve_*.png
├── s23_Midas/results_midas/
│   ├── growth_forecast_metrics.csv
│   └── inflation_forecast_metrics.csv
└── cross_comparison/results/
    ├── performance_comparison_table.csv
    ├── model_comparison_report.md
    └── performance_*.png
```

## Key Metrics

All models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Diebold-Mariano tests** (for statistical significance)

Metrics are computed for:
- **Growth** (Industrial Production MoM % change)
- **Inflation** (Inflation factor)
- **Horizons**: 1, 3, 6 months ahead

## Dependencies

The master script requires:
- Python 3.x
- All dependencies from individual model scripts
- Data files:
  - `main_project/data/macro_final/final_macro.csv`
  - `main_project/data/macro_processed/equity_risk_pr.csv` (for ERP if needed)
  - `main_project/data/news_data/sentiment_scores.csv` (optional, for sentiment features)

## Error Handling

The master script:
- Continues execution even if individual models fail
- Reports success/failure status for each component
- Provides summary of outputs generated
- Returns exit code 0 if at least one model succeeds, 1 if all fail

## Usage Notes

1. **Execution Time**: Running all models may take significant time (30+ minutes depending on hardware)
2. **Memory**: LSTM and MIDAS models may require substantial memory
3. **Data Availability**: Ensure all required data files are present before running
4. **Virtual Environment**: Recommended to use the project's virtual environment

## Next Steps

After successful execution:
1. Review `cross_comparison/results/model_comparison_report.md` for best model identification
2. Check `cross_comparison/results/performance_comparison_table.csv` for detailed metrics
3. Use best-performing model forecasts for trading strategy evaluation (Section 3)

## Individual Model Execution

To run individual models separately:

```bash
# TVP-VAR only
python s21_macro/main.py

# XGBoost only
python s22_ml_based/main.py

# LSTM only
python s22_ml_based/lstm_main.py

# MIDAS only
python s23_Midas/main_midas.py

# Comparison only (requires other models to have run first)
python cross_comparison/main.py
```

## Troubleshooting

- **Import Errors**: Ensure PYTHONPATH includes the section directory
- **Missing Data**: Check that `final_macro.csv` exists and has required columns
- **Memory Issues**: Run models individually instead of using master script
- **Convergence Warnings**: Some models (especially HMM) may show convergence warnings - these are typically non-fatal

