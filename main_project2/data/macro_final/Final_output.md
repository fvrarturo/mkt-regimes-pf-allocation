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

### 2. Economic Growth Factor (Composite - PC1)

**Approach**: Principal Component Analysis (PCA) on standardized real-side indicators

**Input Variables**:
- Industrial Production - month-over-month % change
- Retail Sales - month-over-month % change
- Unemployment - inverted change (negative of month-over-month change)
- Real GDP - month-over-month % change (interpolated from quarterly to monthly)

**Rationale**:
- Economic growth is multifaceted, reacting to labor, production, and consumption
- A single indicator (e.g., GDP) is too infrequent (quarterly) and lagged
- PCA extracts the common "real activity" trend across multiple dimensions
- This aligns with Stock & Watson (1989) "real activity" index and Chicago Fed National Activity Index

**Processing Steps**:
1. Load Industrial Production, Retail Sales, Unemployment, and Real GDP data
2. Extract `pct_change_mom` for IP and Retail Sales
3. For Unemployment: calculate negative of month-over-month change (so lower unemployment = positive growth)
4. For GDP: interpolate quarterly data to monthly frequency using linear interpolation
5. Merge all series on date
6. Forward fill and backward fill missing values
7. Standardize each series using z-scores
8. Run PCA and extract PC1
9. PC1 becomes the `growth_factor`

**Results**:
- **Observations**: 440 monthly observations
- **Date Range**: 1989-01-01 to 2025-08-01
- **PCA Explained Variance**: 75.94%
- **Interpretation**: PC1 captures 75.9% of the common variation across growth indicators, indicating a strong common real activity trend

**PCA Component Weights** (normalized):
- Industrial Production: 0.597 (59.7%)
- Retail Sales: ~0.0 (minimal contribution in this dataset)
- Unemployment Change (inverted): 0.587 (58.7%)
- GDP: 0.547 (54.7%)
- Industrial Production, Unemployment Change, and GDP are the primary drivers of the growth factor

---

### 3. Monetary Policy Factor (Single Series)

**Approach**: Direct extraction of 10y-2y yield curve slope

**Input Variable**:
- 10y-2y Treasury Spread (already computed as 10-year minus 2-year yield)

**Rationale**:
- Monetary policy is largely one-dimensional (Fed controls short-term rate)
- The yield curve slope (10y-2y) incorporates:
  - Expectations of future policy
  - Macro outlook
  - Policy stance relative to the cycle
- This is the single most informative variable in macro-finance for policy stance
- Alternative considered: Fed Funds Rate (more direct but less forward-looking)

**Processing Steps**:
1. Load 10y-2y spread processed data
2. Convert to monthly frequency (take first value of each month)
3. Use spread value directly as `monetary_policy_factor`

**Results**:
- **Observations**: 443 monthly observations
- **Date Range**: 1989-01-01 to 2025-11-01
- **Method**: 10y-2y Yield Curve Slope
- **Interpretation**: Positive values indicate upward-sloping yield curve (normal/expansionary), negative values indicate inverted curve (recessionary signal)

---

### 4. Market Volatility / Financial Conditions Factor (Single Series)

**Approach**: Direct extraction of NFCI (National Financial Conditions Index)

**Input Variable**:
- NFCI (Chicago Fed National Financial Conditions Index)

**Rationale**:
- Financial stress is multidimensional (equity vol, credit spreads, liquidity, leverage, etc.)
- NFCI aggregates all these dimensions into a single index:
  - Credit conditions
  - Liquidity conditions
  - Leverage
  - Funding stress
  - Volatility
  - Spreads
- This is the canonical "single index" for financial conditions
- Alternative considered: VIX (equity volatility only, less comprehensive)

**Processing Steps**:
1. Load NFCI processed data
2. Convert to monthly frequency (take first value of each month)
3. Use NFCI value directly as `market_volatility_factor`

**Results**:
- **Observations**: 443 monthly observations
- **Date Range**: 1989-01-01 to 2025-11-01
- **Method**: NFCI (National Financial Conditions Index)
- **Interpretation**: Negative values indicate accommodative/easy financial conditions, positive values indicate tight/stressful conditions

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
  - `growth_factor`: PC1 from IP, Retail Sales, Unemployment, GDP (standardized)
  - `monetary_policy_factor`: 10y-2y yield curve slope (percentage points)
  - `market_volatility_factor`: NFCI index value

## Summary Statistics

| Factor | Mean | Std Dev | Min | Max | Observations |
|--------|------|---------|-----|-----|--------------|
| Inflation Factor | 0.002 | 1.607 | -11.207 | 6.241 | 444 |
| Growth Factor | 0.001 | 1.504 | -24.446 | 7.430 | 444 |
| Monetary Policy Factor | 0.977 | 0.922 | -1.080 | 2.870 | 444 |
| Market Volatility Factor | -0.384 | 0.500 | -1.101 | 3.045 | 444 |

*Note: PCA factors (inflation, growth) are standardized during PCA but may have slight deviations from mean=0, std=1 due to forward filling and merging. Monetary Policy Factor is in percentage points (10y-2y spread). Market Volatility Factor is the NFCI index value.*

## Key Findings

1. **Inflation Factor**: Strong common trend (86.4% explained variance) across CPI, PCE, and PPI, validating the composite approach.

2. **Growth Factor**: Strong common trend (75.9% explained variance) across real-side indicators, capturing the multi-dimensional nature of economic growth.

3. **Monetary Policy**: Using yield curve slope provides forward-looking policy stance information beyond just the current policy rate.

4. **Market Volatility**: NFCI provides comprehensive financial conditions measure, capturing stress across multiple dimensions.

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

