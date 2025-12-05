# Hailey Testing - Modular Python Scripts

This directory contains modular Python scripts converted from Jupyter notebooks.

## Structure

### Data Modules
- **`data_download.py`**: Downloads market data from Yahoo Finance (VIX, Treasury yields, Dollar index futures)
- **`data_loader.py`**: Loads and prepares macro, stock, and bond data
- **`data_merger.py`**: Merges and aggregates data from multiple sources for forecasting

### Model Modules
- **`forecast_models.py`**: Trains Random Forest and XGBoost models for macro factor forecasting
- **`regime_identification.py`**: Identifies economic regimes based on growth and inflation factors
- **`strategy_construction.py`**: Constructs regime-based portfolio trading strategies
- **`performance_evaluation.py`**: Evaluates and visualizes strategy performance

### Main Scripts
- **`main_get_data.py`**: Downloads Yahoo Finance data (equivalent to `get data.ipynb`)
- **`main_clean_forecast.py`**: Data cleaning and forecasting pipeline (equivalent to `clean + forecast.ipynb`)
- **`main_basic_regime.py`**: Regime-based trading strategy (equivalent to `basic regime.ipynb`)

## Usage

### 1. Download Data
```bash
python main_get_data.py
```

### 2. Clean Data and Train Forecasting Models
```bash
python main_clean_forecast.py
```

### 3. Run Regime-Based Strategy
```bash
python main_basic_regime.py
```

## Dependencies

- pandas
- numpy
- yfinance
- scikit-learn
- xgboost
- matplotlib

## Notes

- All scripts use relative paths that assume they're run from the `hailey testing` directory
- Paths are constructed relative to `main_project` directory
- Data files should be in the expected locations under `main_project/data/`

