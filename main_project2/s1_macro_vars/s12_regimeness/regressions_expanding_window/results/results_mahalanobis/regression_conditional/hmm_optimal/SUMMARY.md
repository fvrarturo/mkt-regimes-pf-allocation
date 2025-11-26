# Regime-Conditional Regression Analysis Summary

**Model**: HMM_OPTIMAL

## Overview

This analysis identifies which macro variables are most significant for predicting ERP in different economic regimes.

## Key Findings

### Overall Statistics

- Total regressions run: 240
- Significant results (p < 0.05): 15 (6.2%)
- Highly significant (p < 0.01): 0 (0.0%)

### Forecast Horizon: 1 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0044** | 2.12 | 0.0342 | 0.0047 |
| nat_fin_condition_indx | -0.0043** | -2.08 | 0.0382 | 0.0012 |
| 10y_treasury_const_maturity_rate | -0.0023 | -1.13 | 0.2597 | 0.0010 |
| fed_reserve_discount_rate | -0.0022 | -1.00 | 0.3173 | 0.0014 |
| industrial_production | -0.0019 | -0.92 | 0.3588 | -0.0007 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0024 | 1.19 | 0.2365 | 0.0060 |
| m2_real_money_supply | 0.0022 | 1.09 | 0.2781 | 0.0026 |
| real_gdp | -0.0034 | -0.96 | 0.3392 | 0.0006 |
| 10y_treasury_const_maturity_rate | -0.0017 | -0.84 | 0.4012 | 0.0025 |
| gdp | -0.0029 | -0.82 | 0.4142 | 0.0013 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0049** | -2.38 | 0.0177 | 0.0011 |
| unemployment | 0.0046** | 2.23 | 0.0264 | 0.0060 |
| fed_reserve_discount_rate | -0.0034 | -1.56 | 0.1186 | 0.0014 |
| 10y_treasury_const_maturity_rate | -0.0029 | -1.40 | 0.1634 | 0.0025 |
| m2_real_money_supply | 0.0026 | 1.25 | 0.2128 | 0.0025 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0027 | 1.31 | 0.1919 | 0.0053 |
| m2_real_money_supply | 0.0024 | 1.18 | 0.2399 | 0.0008 |
| 10y_treasury_const_maturity_rate | -0.0023 | -1.14 | 0.2560 | 0.0010 |
| tot_business_inventories | 0.0021 | 1.00 | 0.3159 | -0.0008 |
| PCE_price_index | 0.0020 | 0.96 | 0.3362 | -0.0003 |

### Forecast Horizon: 3 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0046** | 2.21 | 0.0273 | 0.0056 |
| fed_reserve_discount_rate | -0.0038* | -1.74 | 0.0819 | 0.0026 |
| 10y_treasury_const_maturity_rate | -0.0031 | -1.47 | 0.1411 | 0.0015 |
| gdp | 0.0051 | 1.47 | 0.1443 | 0.0138 |
| real_gdp | 0.0044 | 1.28 | 0.2040 | 0.0084 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| gdp | 0.0064* | 1.83 | 0.0693 | 0.0114 |
| real_gdp | 0.0055 | 1.58 | 0.1157 | 0.0061 |
| nat_fin_condition_indx | -0.0032 | -1.55 | 0.1218 | -0.0030 |
| m2_real_money_supply | 0.0021 | 1.03 | 0.3016 | 0.0012 |
| tot_business_inventories | 0.0022 | 1.03 | 0.3025 | -0.0004 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| m2_real_money_supply | 0.0041** | 1.98 | 0.0482 | -0.0007 |
| 10y_treasury_const_maturity_rate | -0.0039* | -1.86 | 0.0633 | -0.0002 |
| gdp | 0.0059* | 1.69 | 0.0941 | 0.0120 |
| unemployment | 0.0034 | 1.65 | 0.1005 | 0.0062 |
| fed_reserve_discount_rate | -0.0033 | -1.50 | 0.1346 | 0.0033 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0046** | 2.21 | 0.0279 | 0.0049 |
| industrial_production | -0.0028 | -1.33 | 0.1847 | -0.0014 |
| fed_reserve_discount_rate | -0.0018 | -0.82 | 0.4137 | 0.0012 |
| PPI_inflation | -0.0009 | -0.42 | 0.6717 | -0.0023 |
| gdp | 0.0014 | 0.41 | 0.6807 | 0.0026 |

### Forecast Horizon: 6 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0045** | 2.15 | 0.0320 | 0.0092 |
| fed_reserve_discount_rate | -0.0042* | -1.93 | 0.0540 | 0.0062 |
| fedfunds | -0.0032 | -1.55 | 0.1229 | 0.0011 |
| gdp | 0.0051 | 1.48 | 0.1412 | 0.0127 |
| real_gdp | 0.0048 | 1.40 | 0.1651 | 0.0075 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0030 | 1.45 | 0.1482 | 0.0076 |
| gdp | 0.0043 | 1.25 | 0.2139 | 0.0119 |
| 10y_2y_spread | -0.0025 | -1.21 | 0.2281 | -0.0027 |
| industrial_production | -0.0022 | -1.07 | 0.2863 | 0.0002 |
| real_gdp | 0.0031 | 0.90 | 0.3700 | 0.0069 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0047** | 2.29 | 0.0227 | 0.0079 |
| fed_reserve_discount_rate | -0.0043** | -1.99 | 0.0474 | 0.0050 |
| fedfunds | -0.0026 | -1.27 | 0.2042 | 0.0004 |
| m2_real_money_supply | 0.0023 | 1.10 | 0.2715 | 0.0008 |
| industrial_production | -0.0021 | -0.99 | 0.3218 | 0.0001 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0046** | 2.22 | 0.0266 | 0.0092 |
| fed_reserve_discount_rate | -0.0042* | -1.95 | 0.0516 | 0.0061 |
| gdp | 0.0061* | 1.76 | 0.0804 | 0.0116 |
| real_gdp | 0.0053 | 1.54 | 0.1267 | 0.0067 |
| fedfunds | -0.0030 | -1.43 | 0.1539 | 0.0014 |

### Forecast Horizon: 12 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0051** | -2.30 | 0.0223 | 0.0100 |
| fedfunds | -0.0039* | -1.88 | 0.0613 | 0.0061 |
| 10y_treasury_const_maturity_rate | -0.0033 | -1.56 | 0.1201 | 0.0009 |
| PPI_inflation | 0.0031 | 1.47 | 0.1427 | 0.0018 |
| PCE_price_index | 0.0030 | 1.43 | 0.1523 | 0.0011 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0045** | -2.06 | 0.0397 | 0.0105 |
| gdp | 0.0072** | 2.05 | 0.0428 | 0.0075 |
| fedfunds | -0.0040* | -1.94 | 0.0528 | 0.0052 |
| real_gdp | 0.0063* | 1.81 | 0.0732 | 0.0026 |
| PPI_inflation | 0.0036* | 1.73 | 0.0852 | 0.0010 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0034 | -1.55 | 0.1215 | 0.0095 |
| unemployment | 0.0029 | 1.41 | 0.1597 | 0.0008 |
| industrial_production | -0.0024 | -1.15 | 0.2498 | -0.0015 |
| fedfunds | -0.0024 | -1.13 | 0.2579 | 0.0046 |
| gdp | 0.0027 | 0.78 | 0.4345 | 0.0093 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0045** | -2.05 | 0.0413 | 0.0106 |
| fedfunds | -0.0033 | -1.56 | 0.1193 | 0.0061 |
| gdp | 0.0049 | 1.40 | 0.1641 | 0.0126 |
| real_gdp | 0.0037 | 1.06 | 0.2907 | 0.0079 |
| industrial_production | -0.0021 | -0.99 | 0.3245 | -0.0010 |

## Coefficient Differences Across Regimes

No significant coefficient differences found.
