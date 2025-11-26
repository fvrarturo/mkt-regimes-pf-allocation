# Regime-Conditional Regression Analysis Summary

**Model**: 2X2

## Overview

This analysis identifies which macro variables are most significant for predicting ERP in different economic regimes.

## Key Findings

### Overall Statistics

- Total regressions run: 240
- Significant results (p < 0.05): 27 (11.2%)
- Highly significant (p < 0.01): 8 (3.3%)

### Forecast Horizon: 1 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| 10y_2y_spread | -0.0046** | -2.32 | 0.0208 | -0.0228 |
| tot_business_inventories | 0.0042** | 2.07 | 0.0396 | -0.0161 |
| retail_sales | 0.0041** | 2.02 | 0.0437 | -0.0160 |
| m2_real_money_supply | 0.0038* | 1.89 | 0.0589 | -0.0156 |
| nat_fin_condition_indx | 0.0033 | 1.63 | 0.1039 | -0.0262 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0048** | -2.37 | 0.0185 | -0.0067 |
| 10y_2y_spread | 0.0042** | 2.14 | 0.0331 | -0.0132 |
| unemployment | 0.0039** | 2.02 | 0.0439 | 0.0019 |
| fedfunds | -0.0030 | -1.56 | 0.1204 | -0.0047 |
| 10y_treasury_const_maturity_rate | -0.0026 | -1.35 | 0.1789 | 0.0024 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0054*** | 2.61 | 0.0093 | 0.0064 |
| industrial_production | -0.0040* | -1.95 | 0.0518 | -0.0061 |
| gdp | -0.0052 | -1.40 | 0.1632 | -0.0116 |
| real_gdp | -0.0052 | -1.40 | 0.1636 | -0.0115 |
| fed_reserve_discount_rate | -0.0025 | -1.14 | 0.2555 | 0.0008 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| m2_real_money_supply | 0.0051** | 2.45 | 0.0149 | -0.0059 |
| unemployment | 0.0044** | 2.13 | 0.0341 | 0.0021 |
| PCE_price_index | 0.0036* | 1.72 | 0.0869 | -0.0038 |
| nat_fin_condition_indx | -0.0035* | -1.71 | 0.0884 | 0.0028 |
| 10y_treasury_const_maturity_rate | -0.0035* | -1.70 | 0.0890 | -0.0022 |

### Forecast Horizon: 3 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| PPI_inflation | 0.0033* | 1.67 | 0.0951 | 0.0023 |
| PCE_price_index | 0.0027 | 1.36 | 0.1743 | 0.0015 |
| fed_reserve_discount_rate | -0.0028 | -1.36 | 0.1757 | -0.0016 |
| cpi | 0.0026 | 1.34 | 0.1824 | 0.0013 |
| m2_real_money_supply | 0.0025 | 1.27 | 0.2065 | 0.0021 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| gdp | 0.0040 | 1.15 | 0.2504 | -0.0043 |
| 10y_2y_spread | -0.0021 | -1.03 | 0.3050 | -0.0089 |
| real_gdp | 0.0035 | 1.03 | 0.3072 | -0.0106 |
| fed_reserve_discount_rate | -0.0016 | -0.73 | 0.4665 | -0.0097 |
| unemployment | 0.0014 | 0.70 | 0.4853 | -0.0021 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0070*** | -3.35 | 0.0009 | -0.0262 |
| industrial_production | -0.0055*** | -2.63 | 0.0088 | -0.0151 |
| unemployment | 0.0049** | 2.39 | 0.0172 | -0.0057 |
| PPI_inflation | -0.0028 | -1.36 | 0.1761 | -0.0144 |
| cpi | -0.0018 | -0.89 | 0.3760 | -0.0120 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0057*** | -2.60 | 0.0096 | -0.0022 |
| m2_real_money_supply | 0.0049** | 2.35 | 0.0191 | -0.0028 |
| 10y_treasury_const_maturity_rate | -0.0049** | -2.33 | 0.0201 | -0.0024 |
| gdp | 0.0076** | 2.20 | 0.0298 | 0.0078 |
| PCE_price_index | 0.0042** | 2.02 | 0.0442 | -0.0030 |

### Forecast Horizon: 6 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0052** | -2.45 | 0.0147 | 0.0147 |
| unemployment | 0.0032 | 1.50 | 0.1347 | 0.0032 |
| gdp | 0.0049 | 1.44 | 0.1519 | 0.0075 |
| real_gdp | 0.0038 | 1.09 | 0.2762 | 0.0040 |
| fed_reserve_discount_rate | -0.0019 | -0.84 | 0.4026 | 0.0031 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0033 | -1.56 | 0.1197 | 0.0178 |
| 10y_2y_spread | -0.0023 | -1.07 | 0.2839 | -0.0005 |
| fedfunds | 0.0014 | 0.67 | 0.5017 | -0.0032 |
| gdp | 0.0016 | 0.45 | 0.6531 | 0.0058 |
| fed_reserve_discount_rate | -0.0009 | -0.42 | 0.6718 | 0.0007 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0083*** | -3.89 | 0.0001 | -0.0346 |
| industrial_production | -0.0031 | -1.48 | 0.1402 | -0.0091 |
| 10y_2y_spread | -0.0021 | -0.99 | 0.3207 | -0.0111 |
| unemployment | 0.0019 | 0.92 | 0.3584 | -0.0048 |
| m2_real_money_supply | 0.0017 | 0.81 | 0.4188 | -0.0070 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0058*** | 2.78 | 0.0056 | 0.0050 |
| fed_reserve_discount_rate | -0.0055** | -2.53 | 0.0117 | 0.0015 |
| gdp | 0.0077** | 2.20 | 0.0296 | -0.0013 |
| fedfunds | -0.0041* | -1.95 | 0.0520 | -0.0037 |
| real_gdp | 0.0063* | 1.80 | 0.0740 | -0.0045 |

### Forecast Horizon: 12 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| gdp | 0.0094** | 2.55 | 0.0121 | 0.0023 |
| real_gdp | 0.0073* | 1.98 | 0.0502 | 0.0007 |
| m2_real_money_supply | 0.0025 | 1.12 | 0.2652 | -0.0026 |
| tot_business_inventories | 0.0024 | 1.03 | 0.3028 | -0.0018 |
| retail_sales | 0.0023 | 1.01 | 0.3145 | -0.0030 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0029 | -1.36 | 0.1731 | -0.0036 |
| industrial_production | -0.0028 | -1.31 | 0.1906 | -0.0032 |
| unemployment | 0.0025 | 1.14 | 0.2532 | -0.0001 |
| m2_real_money_supply | -0.0018 | -0.82 | 0.4111 | -0.0060 |
| retail_sales | -0.0014 | -0.65 | 0.5142 | -0.0048 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0088*** | -3.91 | 0.0001 | -0.0024 |
| fedfunds | -0.0075*** | -3.54 | 0.0005 | -0.0045 |
| 10y_2y_spread | 0.0053** | 2.47 | 0.0139 | -0.0084 |
| unemployment | 0.0043** | 2.02 | 0.0443 | 0.0002 |
| 10y_treasury_const_maturity_rate | -0.0038* | -1.78 | 0.0759 | -0.0015 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0042* | -1.90 | 0.0587 | 0.0086 |
| PPI_inflation | 0.0039* | 1.87 | 0.0628 | -0.0018 |
| gdp | 0.0064* | 1.83 | 0.0692 | 0.0067 |
| cpi | 0.0037* | 1.78 | 0.0751 | -0.0023 |
| PCE_price_index | 0.0037* | 1.78 | 0.0757 | -0.0022 |

## Coefficient Differences Across Regimes

Found 29 significant coefficient differences:

| Variable | Horizon | Regime1 | Regime2 | Difference | p-value |
|----------|---------|---------|---------|------------|----------|
| 10y_2y_spread | 1 | 0 | 1 | -0.0088 | 0.0016 |
| 10y_2y_spread | 1 | 0 | 2 | -0.0066 | 0.0219 |
| 10y_2y_spread | 1 | 1 | 3 | 0.0068 | 0.0175 |
| fed_reserve_discount_rate | 1 | 0 | 1 | 0.0065 | 0.0262 |
| industrial_production | 1 | 0 | 2 | 0.0064 | 0.0253 |
| nat_fin_condition_indx | 1 | 0 | 3 | 0.0068 | 0.0182 |
| unemployment | 1 | 0 | 1 | -0.0069 | 0.0131 |
| unemployment | 1 | 0 | 2 | -0.0084 | 0.0036 |
| unemployment | 1 | 0 | 3 | -0.0074 | 0.0102 |
| 10y_treasury_const_maturity_rate | 3 | 2 | 3 | 0.0064 | 0.0302 |
| PCE_price_index | 3 | 2 | 3 | -0.0059 | 0.0439 |
| PPI_inflation | 3 | 0 | 2 | 0.0061 | 0.0328 |
| PPI_inflation | 3 | 2 | 3 | -0.0058 | 0.0480 |
| cpi | 3 | 2 | 3 | -0.0059 | 0.0440 |
| fed_reserve_discount_rate | 3 | 2 | 3 | 0.0060 | 0.0499 |
| industrial_production | 3 | 0 | 2 | 0.0064 | 0.0251 |
| m2_real_money_supply | 3 | 2 | 3 | -0.0060 | 0.0415 |
| nat_fin_condition_indx | 3 | 0 | 2 | 0.0079 | 0.0057 |
| nat_fin_condition_indx | 3 | 1 | 2 | 0.0063 | 0.0311 |
| nat_fin_condition_indx | 3 | 2 | 3 | -0.0078 | 0.0082 |
