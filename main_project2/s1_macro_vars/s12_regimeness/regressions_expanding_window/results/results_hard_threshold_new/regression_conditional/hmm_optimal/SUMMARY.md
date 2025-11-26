# Regime-Conditional Regression Analysis Summary

**Model**: HMM_OPTIMAL

## Overview

This analysis identifies which macro variables are most significant for predicting ERP in different economic regimes.

## Key Findings

### Overall Statistics

- Total regressions run: 240
- Significant results (p < 0.05): 15 (6.2%)
- Highly significant (p < 0.01): 4 (1.7%)

### Forecast Horizon: 1 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| gdp | -0.0162** | -2.68 | 0.0113 | 0.1706 |
| real_gdp | -0.0153** | -2.51 | 0.0172 | 0.1525 |
| fedfunds | 0.0051 | 1.25 | 0.2123 | 0.0141 |
| 10y_2y_spread | -0.0049 | -1.22 | 0.2246 | 0.0134 |
| nat_fin_condition_indx | -0.0047 | -1.17 | 0.2434 | 0.0124 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0031 | -0.67 | 0.5033 | 0.0082 |
| nat_fin_condition_indx | 0.0023 | 0.56 | 0.5764 | 0.0036 |
| PCE_price_index | 0.0023 | 0.55 | 0.5859 | 0.0034 |
| real_gdp | -0.0040 | -0.53 | 0.6011 | 0.0092 |
| gdp | -0.0036 | -0.46 | 0.6462 | 0.0071 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0086** | 2.00 | 0.0483 | 0.0363 |
| gdp | 0.0126 | 1.66 | 0.1070 | 0.0794 |
| real_gdp | 0.0115 | 1.51 | 0.1407 | 0.0668 |
| fedfunds | -0.0056 | -1.28 | 0.2022 | 0.0153 |
| m2_real_money_supply | 0.0054 | 1.23 | 0.2205 | 0.0141 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0057 | -1.49 | 0.1388 | 0.0196 |
| unemployment | 0.0053 | 1.40 | 0.1639 | 0.0171 |
| fedfunds | -0.0047 | -1.23 | 0.2217 | 0.0132 |
| nat_fin_condition_indx | -0.0041 | -1.08 | 0.2810 | 0.0103 |
| 10y_treasury_const_maturity_rate | -0.0037 | -0.97 | 0.3365 | 0.0082 |

### Forecast Horizon: 3 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0061 | 1.59 | 0.1137 | 0.0226 |
| industrial_production | -0.0044 | -1.15 | 0.2522 | 0.0119 |
| nat_fin_condition_indx | -0.0040 | -1.03 | 0.3065 | 0.0095 |
| retail_sales | -0.0032 | -0.80 | 0.4238 | 0.0061 |
| 10y_2y_spread | 0.0025 | 0.66 | 0.5131 | 0.0039 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fedfunds | 0.0087* | 1.85 | 0.0674 | 0.0384 |
| gdp | 0.0117 | 1.64 | 0.1127 | 0.0821 |
| 10y_2y_spread | -0.0063 | -1.34 | 0.1840 | 0.0204 |
| real_gdp | 0.0095 | 1.30 | 0.2032 | 0.0536 |
| nat_fin_condition_indx | -0.0053 | -1.11 | 0.2685 | 0.0142 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0076* | -1.79 | 0.0761 | 0.0299 |
| fedfunds | -0.0075* | -1.76 | 0.0812 | 0.0284 |
| m2_real_money_supply | 0.0071 | 1.66 | 0.1005 | 0.0252 |
| gdp | 0.0106 | 1.54 | 0.1346 | 0.0688 |
| 10y_treasury_const_maturity_rate | -0.0064 | -1.51 | 0.1351 | 0.0209 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| 10y_treasury_const_maturity_rate | -0.0046 | -1.22 | 0.2237 | 0.0131 |
| nat_fin_condition_indx | 0.0038 | 1.03 | 0.3069 | 0.0092 |
| unemployment | 0.0029 | 0.77 | 0.4440 | 0.0052 |
| fedfunds | -0.0025 | -0.67 | 0.5011 | 0.0040 |
| 10y_2y_spread | -0.0024 | -0.64 | 0.5245 | 0.0036 |

### Forecast Horizon: 6 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | 0.0068* | 1.95 | 0.0537 | 0.0334 |
| retail_sales | -0.0046 | -1.29 | 0.2006 | 0.0156 |
| m2_real_money_supply | -0.0042 | -1.19 | 0.2380 | 0.0126 |
| tot_business_inventories | -0.0039 | -1.07 | 0.2877 | 0.0108 |
| PPI_inflation | -0.0028 | -0.78 | 0.4349 | 0.0056 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| tot_business_inventories | 0.0062 | 1.23 | 0.2236 | 0.0187 |
| PCE_price_index | 0.0057 | 1.17 | 0.2445 | 0.0163 |
| cpi | 0.0056 | 1.15 | 0.2539 | 0.0157 |
| 10y_2y_spread | -0.0055 | -1.13 | 0.2608 | 0.0152 |
| fedfunds | 0.0055 | 1.13 | 0.2637 | 0.0150 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0078* | 1.75 | 0.0830 | 0.0281 |
| nat_fin_condition_indx | -0.0075* | -1.66 | 0.0992 | 0.0254 |
| industrial_production | -0.0066 | -1.47 | 0.1445 | 0.0200 |
| gdp | 0.0074 | 0.83 | 0.4130 | 0.0211 |
| real_gdp | 0.0070 | 0.79 | 0.4373 | 0.0190 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fedfunds | -0.0113*** | -3.18 | 0.0019 | 0.0821 |
| fed_reserve_discount_rate | -0.0111*** | -3.14 | 0.0022 | 0.0816 |
| 10y_treasury_const_maturity_rate | -0.0109*** | -3.03 | 0.0030 | 0.0753 |
| PCE_price_index | 0.0091** | 2.52 | 0.0133 | 0.0530 |
| cpi | 0.0091** | 2.50 | 0.0140 | 0.0523 |

### Forecast Horizon: 12 month(s)

#### Regime 0

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| PPI_inflation | 0.0028 | 0.72 | 0.4756 | 0.0047 |
| unemployment | 0.0028 | 0.70 | 0.4880 | 0.0044 |
| tot_business_inventories | 0.0025 | 0.61 | 0.5446 | 0.0035 |
| retail_sales | 0.0024 | 0.59 | 0.5549 | 0.0034 |
| nat_fin_condition_indx | -0.0022 | -0.54 | 0.5871 | 0.0027 |

#### Regime 1

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0133*** | -2.91 | 0.0053 | 0.1334 |
| fedfunds | -0.0064 | -1.58 | 0.1179 | 0.0307 |
| gdp | 0.0091 | 1.44 | 0.1608 | 0.0693 |
| m2_real_money_supply | 0.0056 | 1.39 | 0.1698 | 0.0237 |
| real_gdp | 0.0088 | 1.38 | 0.1779 | 0.0641 |

#### Regime 2

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| PPI_inflation | 0.0041 | 0.95 | 0.3447 | 0.0085 |
| tot_business_inventories | 0.0039 | 0.89 | 0.3755 | 0.0077 |
| PCE_price_index | 0.0033 | 0.78 | 0.4388 | 0.0057 |
| cpi | 0.0032 | 0.75 | 0.4525 | 0.0054 |
| gdp | 0.0046 | 0.68 | 0.4995 | 0.0144 |

#### Regime 3

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0052 | -1.27 | 0.2080 | 0.0142 |
| unemployment | 0.0052 | 1.26 | 0.2110 | 0.0138 |
| fedfunds | -0.0048 | -1.17 | 0.2440 | 0.0120 |
| industrial_production | -0.0042 | -1.02 | 0.3097 | 0.0091 |
| gdp | 0.0050 | 0.77 | 0.4440 | 0.0144 |

## Coefficient Differences Across Regimes

Found 18 significant coefficient differences:

| Variable | Horizon | Regime1 | Regime2 | Difference | p-value |
|----------|---------|---------|---------|------------|----------|
| gdp | 1 | 0 | 2 | -0.0288 | 0.0029 |
| gdp | 1 | 0 | 3 | -0.0194 | 0.0301 |
| real_gdp | 1 | 0 | 2 | -0.0269 | 0.0059 |
| real_gdp | 1 | 0 | 3 | -0.0184 | 0.0406 |
| fedfunds | 3 | 1 | 2 | 0.0162 | 0.0106 |
| 10y_treasury_const_maturity_rate | 6 | 0 | 3 | 0.0129 | 0.0103 |
| 10y_treasury_const_maturity_rate | 6 | 2 | 3 | 0.0120 | 0.0383 |
| PCE_price_index | 6 | 0 | 3 | -0.0116 | 0.0221 |
| PPI_inflation | 6 | 0 | 3 | -0.0108 | 0.0328 |
| cpi | 6 | 0 | 3 | -0.0115 | 0.0229 |
| fed_reserve_discount_rate | 6 | 0 | 3 | 0.0106 | 0.0346 |
| fedfunds | 6 | 0 | 3 | 0.0131 | 0.0092 |
| fedfunds | 6 | 1 | 3 | 0.0168 | 0.0053 |
| m2_real_money_supply | 6 | 0 | 3 | -0.0125 | 0.0134 |
| nat_fin_condition_indx | 6 | 0 | 2 | 0.0143 | 0.0121 |
| retail_sales | 6 | 0 | 3 | -0.0122 | 0.0173 |
| tot_business_inventories | 6 | 0 | 3 | -0.0114 | 0.0267 |
| fed_reserve_discount_rate | 12 | 0 | 1 | 0.0132 | 0.0329 |
