# Macro Data Engineering - Processed Data

This directory contains engineered macroeconomic data following academic and research best practices.

## Overview

All raw macro data has been transformed using appropriate statistical and econometric methods based on variable type and academic literature.

## Transformations Applied

### 1. **Level Variables** (GDP, CPI, Price Indices, Industrial Production, etc.)

These variables represent levels or indices that grow over time. Applied transformations:

- **Percentage Changes**:
  - `pct_change_mom`: Month-over-Month percentage change
  - `pct_change_yoy`: Year-over-Year percentage change (12-month lag)

- **Log Transformations**:
  - `log_value`: Natural logarithm of the level (for variables with exponential growth)
  - `log_diff`: First difference of log values (approximates percentage change)
  - `log_pct_change`: Percentage change in log values

- **Normalization** (using expanding windows to avoid look-ahead bias):
  - `zscore_*`: Z-score normalization (standardization) - mean=0, std=1
  - `minmax_*`: Min-Max normalization - scaled to [0, 1] range

**Rationale**: Level variables are typically non-stationary. Percentage changes and log transformations help achieve stationarity, which is crucial for time series analysis and econometric modeling (Hamilton, 1994; Stock & Watson, 2011).

### 2. **Rate Variables** (Interest Rates, Unemployment Rate, etc.)

These variables are already in rate/percentage form. Applied transformations:

- **First Differences**:
  - `first_diff`: First difference (Δx_t = x_t - x_{t-1})

- **Percentage Changes**:
  - `pct_change_mom`: Month-over-Month percentage change
  - `pct_change_yoy`: Year-over-Year percentage change

- **Normalization** (using expanding windows to avoid look-ahead bias):
  - `zscore_*`: Z-score normalization
  - `minmax_*`: Min-Max normalization

**Rationale**: Rates are often I(1) processes. First differencing helps achieve stationarity. Percentage changes capture relative movements.

### 3. **Spread Variables** (Yield Spreads, Credit Spreads)

Applied transformations:

- **First Differences**:
  - `first_diff`: First difference

- **Percentage Changes**:
  - `pct_change_mom`: Month-over-Month percentage change

- **Normalization** (using expanding windows to avoid look-ahead bias):
  - `zscore_*`: Z-score normalization
  - `minmax_*`: Min-Max normalization

**Rationale**: Spreads are often used as leading indicators. First differences capture changes in the spread, which are economically meaningful.

### 4. **Volatility Variables** (VIX, Volatility Indices)

Applied transformations:

- **Log Transformations**:
  - `log_value`: Natural logarithm (volatility is log-normally distributed)
  - `log_diff`: First difference of log values
  - `log_pct_change`: Percentage change in log values

- **Percentage Changes**:
  - `pct_change_mom`: Month-over-Month percentage change

- **Normalization** (using expanding windows to avoid look-ahead bias):
  - `zscore_*`: Z-score normalization
  - `minmax_*`: Min-Max normalization

**Rationale**: Volatility measures are typically log-normally distributed. Log transformation stabilizes variance and makes the distribution more symmetric (Diebold, 2017).

### 5. **Index Variables** (Financial Condition Indices)

Applied transformations:

- **Percentage Changes**:
  - `pct_change_mom`: Month-over-Month percentage change
  - `pct_change_yoy`: Year-over-Year percentage change

- **Normalization** (using expanding windows to avoid look-ahead bias):
  - `zscore_*`: Z-score normalization
  - `minmax_*`: Min-Max normalization

## File Structure

```
macro_processed/
├── ec_growth/
│   ├── export_price_index_processed.csv
│   ├── gdp_processed.csv
│   ├── import_price_index_processed.csv
│   ├── industrial_production_processed.csv
│   ├── real_gdp_processed.csv
│   ├── retail_sales_processed.csv
│   ├── tot_business_inventories_processed.csv
│   └── unemployment_processed.csv
├── inflation/
│   ├── cpi_processed.csv
│   ├── PCE_price_index_processed.csv
│   └── PPI_inflation_processed.csv
├── mkt_vol/
│   ├── 10y_2y_spread_processed.csv
│   ├── 3month_vol_index_sp500_processed.csv
│   ├── nasdaq_vol_indx_processed.csv
│   ├── nat_fin_condition_indx_processed.csv
│   └── vix_processed.csv
├── mon_policy/
│   ├── 10y_treasury_const_maturity_rate_processed.csv
│   ├── fed_reserve_discount_rate_processed.csv
│   ├── fedfunds_processed.csv
│   └── m2_real_money_supply_processed.csv
├── other/
│   ├── 10y_yield_processed.csv
│   ├── 2y_yield_processed.csv
│   ├── 3m_yield_processed.csv
│   ├── bofa_highyield_spread_processed.csv
│   └── sp500_processed.csv
├── processing_summary.txt
└── README.md
```

## Usage

Each processed CSV file contains:
- `date`: Observation date
- `value`: Original raw value
- Transformed columns based on variable category (see above)

### Example: Loading Processed Data

```python
import pandas as pd

# Load processed CPI data
cpi = pd.read_csv('inflation/cpi_processed.csv', parse_dates=['date'])

# Use percentage changes for analysis
cpi_yoy = cpi[['date', 'pct_change_yoy']].dropna()

# Use normalized values for machine learning
cpi_normalized = cpi[['date', 'zscore_value', 'zscore_pct_change_yoy']].dropna()
```

## Academic References

1. **Hamilton, J. D. (1994)**. *Time Series Analysis*. Princeton University Press.
   - Chapter 3: Stationary ARMA Processes
   - Chapter 15: Cointegration

2. **Stock, J. H., & Watson, M. W. (2011)**. *Introduction to Econometrics* (3rd ed.). Pearson.
   - Chapter 14: Introduction to Time Series Regression and Forecasting

3. **Diebold, F. X. (2017)**. *Forecasting: Principles and Practice* (2nd ed.).
   - Chapter 3: Time Series Decomposition
   - Chapter 8: ARIMA Models

4. **Tsay, R. S. (2010)**. *Analysis of Financial Time Series* (3rd ed.). Wiley.
   - Chapter 2: Linear Time Series Analysis and Its Applications

## Notes

- All transformations preserve the original `value` column for reference
- Missing values in transformed columns occur when:
  - First observation (no lag available for differences/percentage changes)
  - Year-over-year changes require 12+ months of data
- **Normalization uses expanding windows to avoid look-ahead bias**: For each time point, normalization statistics (mean, std, min, max) are computed using only historical data up to that point. This ensures that future information is never used to normalize past observations, which is critical for time series forecasting and backtesting.

## Regenerating Processed Data

To regenerate processed data after updating raw data:

```bash
cd main_project/data/macro
python engineer_macro_data.py
```

The script will automatically:
1. Load all CSV files from subdirectories
2. Apply appropriate transformations based on variable type
3. Save processed files to `macro_processed/` directory
4. Generate a summary report

