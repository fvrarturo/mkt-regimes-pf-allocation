# Final Macro Dataset Creation - Steps and Findings

## Overview

This document describes the process of creating the final macro dataset (`final_macro.csv`) containing four key macroeconomic factors for the period 1989-2025 (monthly frequency). The dataset follows best practices from empirical macro-finance research, using composite factors where appropriate to capture common trends across multiple indicators.

## Methodology

### 1. Inflation Factor (Composite - PC1)

**Approach**: Principal Component Analysis (PCA) on standardized inflation series

**Input Variables**:
- CPI (Consumer Price Index) - month-over-month % change
- PCE (Personal Consumption Expenditures Price Index) - month-over-month % change  
- PPI (Producer Price Index) - month-over-month % change

**Rationale**: 
- Inflation is multidimensional, capturing different channels (consumer, producer, pipeline)
- Using a single index (e.g., CPI alone) gives it undue weight due to volatility differences
- PCA extracts the common inflation trend across all three indices
- This aligns with Stock & Watson methodology and Fed/ECB nowcasting practices

**Processing Steps**:
1. Load CPI, PCE, and PPI processed data
2. Extract `pct_change_mom` for each series
3. Merge on date (outer join to preserve all dates)
4. Standardize each series using z-scores
5. Run PCA and extract PC1 (first principal component)
6. PC1 becomes the `inflation_factor`

**Results**:
- **Observations**: 441 monthly observations
- **Date Range**: 1989-01-01 to 2025-09-01
- **PCA Explained Variance**: 86.41%
- **Interpretation**: PC1 captures 86.4% of the common variation across CPI, PCE, and PPI, indicating a strong common inflation trend

**PCA Component Weights** (normalized):
- CPI: 0.596 (59.6%)
- PCE: 0.592 (59.2%)
- PPI: 0.542 (54.2%)
- All three series contribute positively and roughly equally to the factor, indicating they capture a common inflation trend

---

### 2. Economic Growth Factor (Industrial Production)

**Approach**: Direct extraction of Industrial Production month-over-month percentage change

**Input Variable**:
- Industrial Production Index (INDPRO) - month-over-month % change

**Rationale**:
- Industrial Production is a key real-time indicator of economic activity
- Available monthly with minimal lag (unlike GDP which is quarterly)
- Highly correlated with overall economic growth and business cycles
- Widely used in macro-finance research as a proxy for real activity
- Simpler and more interpretable than composite factors for regime analysis
- Aligns with the focus on identifying predictive patterns in macro variables

**Processing Steps**:
1. Load Industrial Production processed data (`ind_prod_processed.csv`)
2. Extract `pct_change_mom` (month-over-month percentage change)
3. Convert dates to month-start frequency to match other factors
4. Use `pct_change_mom` directly as `growth_factor`

**Results**:
- **Observations**: 428 monthly observations (1990-01-01 to 2025-08-01)
- **Date Range**: 1990-01-01 to 2025-08-01 (starts later than other factors due to data availability)
- **Method**: Industrial Production MoM % Change
- **Mean**: 0.12% per month
- **Std Dev**: 1.00% per month
- **Interpretation**: Positive values indicate month-over-month growth in industrial production, negative values indicate contraction. This provides a direct, interpretable measure of real economic activity growth.

---

### 3. Monetary Policy Factor (Single Series)

**Approach**: Direct extraction of Federal Funds Rate

**Input Variable**:
- Federal Funds Rate (effective rate)

**Rationale**:
- Monetary policy is largely one-dimensional (Fed controls short-term rate)
- Federal Funds Rate is the primary policy tool and most direct measure of policy stance
- Directly interpretable: higher rates indicate tighter policy, lower rates indicate easier policy
- Essential for regime classification and understanding policy-driven market dynamics

**Processing Steps**:
1. Load Federal Funds Rate processed data
2. Convert to monthly frequency (take first value of each month)
3. Use rate value directly as `monetary_policy_factor`

**Results**:
- **Observations**: 444 monthly observations
- **Date Range**: 1989-01-01 to 2025-12-01
- **Method**: Federal Funds Rate
- **Interpretation**: Values represent the effective Federal Funds Rate in percentage points. Higher values indicate tighter monetary policy, lower values indicate easier policy.

---

### 4. Market Volatility / Financial Conditions Factor (Single Series)

**Approach**: Direct extraction of VIX (CBOE Volatility Index)

**Input Variable**:
- VIX (CBOE Volatility Index)

**Rationale**:
- VIX is the most widely recognized measure of market volatility and fear
- Directly measures expected volatility of S&P 500 options
- Highly correlated with market stress and risk-off periods
- More readily available and interpretable than composite indices
- Essential for understanding market volatility regimes

**Processing Steps**:
1. Load VIX processed data
2. Convert to monthly frequency (take first value of each month)
3. Use VIX value directly as `market_volatility_factor`

**Results**:
- **Observations**: 444 monthly observations
- **Date Range**: 1989-01-01 to 2025-12-01
- **Method**: VIX (CBOE Volatility Index)
- **Interpretation**: Values represent the VIX index level. Higher values indicate higher expected volatility and market stress, lower values indicate calmer market conditions.

---

## Final Dataset Assembly

**Merging Process**:
1. Start with inflation factor dates (most complete)
2. Outer join with growth factor
3. Outer join with monetary policy factor
4. Outer join with market volatility factor
5. Create complete monthly date range (1989-01 to 2025-12)
6. Forward fill missing values to ensure continuity
7. Sort by date

**Final Dataset**:
- **File**: `final_macro.csv`
- **Observations**: 444 monthly observations
- **Date Range**: 1989-01-01 to 2025-12-01
- **Columns**:
  - `date`: Monthly date (first day of month)
  - `inflation_factor`: PC1 from CPI, PCE, PPI (standardized)
  - `growth_factor`: Industrial Production month-over-month % change (percentage points)
  - `monetary_policy_factor`: Federal Funds Rate (percentage points)
  - `market_volatility_factor`: VIX index value

## Summary Statistics

| Factor | Mean | Std Dev | Min | Max | Observations |
|--------|------|---------|-----|-----|--------------|
| Inflation Factor | 0.002 | 1.607 | -11.207 | 6.241 | 444 |
| Growth Factor | 0.120 | 1.004 | -7.88 | 7.76 | 428* |
| Monetary Policy Factor | 2.97 | 2.15 | 0.05 | 8.23 | 444 |
| Market Volatility Factor | 19.45 | 8.23 | 9.45 | 68.51 | 444 |

*Note: Growth Factor has 16 missing values (1989-01-01 to 1989-12-01) because Industrial Production data starts in 1990-01-31. Growth Factor is Industrial Production MoM % change (percentage points). Monetary Policy Factor is Federal Funds Rate (percentage points). Market Volatility Factor is VIX index value.*

## Key Findings

1. **Inflation Factor**: Strong common trend (86.4% explained variance) across CPI, PCE, and PPI, validating the composite approach.

2. **Growth Factor**: Industrial Production provides a direct, interpretable measure of real economic activity growth. Monthly frequency with minimal lag makes it ideal for regime analysis and forecasting applications.

3. **Monetary Policy**: Federal Funds Rate provides direct, interpretable measure of policy stance essential for regime classification.

4. **Market Volatility**: VIX provides direct measure of expected market volatility, essential for understanding volatility regimes and risk-off periods.

5. **Data Coverage**: Successfully created monthly dataset from 1989-2025, covering major economic cycles including:
   - Early 1990s recession
   - Dot-com bubble and bust
   - 2008 Financial Crisis
   - COVID-19 pandemic
   - Recent inflation surge

## Code Structure

The implementation is modular, with separate Python files for each factor:

- `src/inflation.py`: Inflation factor creation
- `src/ec_growth.py`: Economic growth factor creation
- `src/mon_policy.py`: Monetary policy factor extraction
- `src/mkt_volatility.py`: Market volatility factor extraction
- `main.py`: Orchestration script that runs all modules and creates final dataset

## Usage

To regenerate the dataset:

```bash
cd main_project/data/macro_final
python3 main.py
```

This will create/update `final_macro.csv` with the latest data.

## References

- Stock, J. H., & Watson, M. W. (1989). New indexes of coincident and leading economic indicators. *NBER Macroeconomics Annual*, 4, 351-394.
- Stock, J. H., & Watson, M. W. (2002). Macroeconomic forecasting using diffusion indexes. *Journal of Business & Economic Statistics*, 20(2), 147-162.
- Adrian, T., Crump, R. K., & Moench, E. (2013). Pricing the term structure with linear regressions. *Journal of Financial Economics*, 110(1), 110-138.
- Chicago Fed National Financial Conditions Index methodology

---

**Generated**: 2025-11-22
**Script Version**: 1.0

