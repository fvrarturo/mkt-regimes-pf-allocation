# Regime-Conditional Regression Analysis Summary

**Model**: 2X2

## Overview

This analysis identifies which macro variables are most significant for predicting ERP in different economic regimes.

## Key Findings

### Overall Statistics

- Total regressions run: 240
- Significant results (p < 0.05): 22 (9.2%)
- Highly significant (p < 0.01): 3 (1.2%)

### Forecast Horizon: 1 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| tot_business_inventories | 0.0055* | 1.74 | 0.0836 | 0.0245 |
| retail_sales | 0.0054* | 1.68 | 0.0946 | 0.0229 |
| 10y_2y_spread | -0.0049 | -1.55 | 0.1238 | 0.0187 |
| m2_real_money_supply | 0.0044 | 1.41 | 0.1616 | 0.0155 |
| nat_fin_condition_indx | 0.0037 | 1.16 | 0.2493 | 0.0105 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0116** | -2.43 | 0.0184 | 0.0880 |
| fedfunds | -0.0091** | -2.03 | 0.0462 | 0.0500 |
| 10y_treasury_const_maturity_rate | -0.0084* | -1.86 | 0.0663 | 0.0426 |
| unemployment | 0.0069 | 1.50 | 0.1386 | 0.0283 |
| 10y_2y_spread | 0.0060 | 1.31 | 0.1952 | 0.0214 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0091** | 2.28 | 0.0250 | 0.0474 |
| industrial_production | -0.0081** | -2.03 | 0.0454 | 0.0380 |
| 10y_2y_spread | 0.0069* | 1.72 | 0.0879 | 0.0278 |
| fed_reserve_discount_rate | -0.0053 | -1.27 | 0.2082 | 0.0179 |
| retail_sales | -0.0049 | -1.19 | 0.2368 | 0.0136 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| m2_real_money_supply | 0.0069 | 1.52 | 0.1322 | 0.0207 |
| 10y_2y_spread | -0.0054 | -1.18 | 0.2389 | 0.0127 |
| 10y_treasury_const_maturity_rate | -0.0050 | -1.10 | 0.2754 | 0.0109 |
| PCE_price_index | 0.0049 | 1.06 | 0.2900 | 0.0103 |
| cpi | 0.0046 | 1.00 | 0.3189 | 0.0091 |

### Forecast Horizon: 3 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| PPI_inflation | 0.0037 | 1.19 | 0.2349 | 0.0112 |
| nat_fin_condition_indx | 0.0029 | 0.93 | 0.3532 | 0.0068 |
| PCE_price_index | 0.0027 | 0.87 | 0.3846 | 0.0060 |
| tot_business_inventories | 0.0028 | 0.87 | 0.3847 | 0.0063 |
| cpi | 0.0027 | 0.86 | 0.3908 | 0.0059 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| 10y_2y_spread | -0.0031 | -0.74 | 0.4609 | 0.0072 |
| industrial_production | 0.0028 | 0.67 | 0.5075 | 0.0058 |
| unemployment | -0.0024 | -0.57 | 0.5703 | 0.0043 |
| PPI_inflation | -0.0020 | -0.46 | 0.6444 | 0.0028 |
| gdp | 0.0033 | 0.42 | 0.6760 | 0.0074 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| industrial_production | -0.0092* | -1.93 | 0.0569 | 0.0344 |
| nat_fin_condition_indx | -0.0082* | -1.70 | 0.0918 | 0.0271 |
| unemployment | 0.0066 | 1.37 | 0.1723 | 0.0178 |
| PPI_inflation | -0.0061 | -1.26 | 0.2123 | 0.0149 |
| cpi | -0.0045 | -0.92 | 0.3596 | 0.0081 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| m2_real_money_supply | 0.0104** | 2.50 | 0.0140 | 0.0542 |
| fed_reserve_discount_rate | -0.0102** | -2.37 | 0.0196 | 0.0513 |
| 10y_treasury_const_maturity_rate | -0.0090** | -2.16 | 0.0333 | 0.0409 |
| PCE_price_index | 0.0090** | 2.14 | 0.0343 | 0.0404 |
| cpi | 0.0087** | 2.08 | 0.0398 | 0.0382 |

### Forecast Horizon: 6 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0070* | 1.94 | 0.0552 | 0.0291 |
| 10y_2y_spread | 0.0036 | 1.00 | 0.3208 | 0.0079 |
| gdp | 0.0058 | 0.87 | 0.3880 | 0.0166 |
| industrial_production | -0.0027 | -0.74 | 0.4620 | 0.0043 |
| PPI_inflation | 0.0024 | 0.65 | 0.5138 | 0.0034 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| 10y_2y_spread | -0.0088* | -1.99 | 0.0500 | 0.0510 |
| industrial_production | 0.0048 | 1.06 | 0.2944 | 0.0149 |
| unemployment | -0.0039 | -0.87 | 0.3885 | 0.0101 |
| 10y_treasury_const_maturity_rate | -0.0038 | -0.84 | 0.4040 | 0.0094 |
| cpi | 0.0033 | 0.72 | 0.4731 | 0.0070 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| nat_fin_condition_indx | -0.0070* | -1.73 | 0.0874 | 0.0278 |
| unemployment | 0.0035 | 0.85 | 0.3947 | 0.0070 |
| fed_reserve_discount_rate | -0.0029 | -0.69 | 0.4914 | 0.0054 |
| industrial_production | -0.0026 | -0.64 | 0.5222 | 0.0039 |
| PPI_inflation | -0.0020 | -0.48 | 0.6347 | 0.0022 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| gdp | 0.0112* | 1.84 | 0.0732 | 0.0713 |
| fed_reserve_discount_rate | -0.0073* | -1.67 | 0.0971 | 0.0262 |
| m2_real_money_supply | 0.0070 | 1.63 | 0.1060 | 0.0238 |
| unemployment | 0.0069 | 1.63 | 0.1070 | 0.0237 |
| fedfunds | -0.0066 | -1.56 | 0.1228 | 0.0217 |

### Forecast Horizon: 12 month(s)

#### Goldilocks

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fedfunds | 0.0054 | 1.57 | 0.1183 | 0.0196 |
| 10y_2y_spread | -0.0051 | -1.47 | 0.1444 | 0.0171 |
| fed_reserve_discount_rate | 0.0051 | 1.40 | 0.1648 | 0.0167 |
| gdp | 0.0079 | 1.24 | 0.2212 | 0.0331 |
| 10y_treasury_const_maturity_rate | 0.0040 | 1.17 | 0.2460 | 0.0108 |

#### Overheating

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| unemployment | 0.0029 | 0.52 | 0.6014 | 0.0038 |
| industrial_production | -0.0027 | -0.49 | 0.6242 | 0.0033 |
| m2_real_money_supply | -0.0026 | -0.47 | 0.6403 | 0.0030 |
| retail_sales | -0.0019 | -0.33 | 0.7449 | 0.0015 |
| fed_reserve_discount_rate | 0.0015 | 0.24 | 0.8076 | 0.0010 |

#### Stagflation

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| fed_reserve_discount_rate | -0.0158*** | -3.49 | 0.0008 | 0.1216 |
| fedfunds | -0.0143*** | -3.38 | 0.0010 | 0.1017 |
| 10y_2y_spread | 0.0101** | 2.33 | 0.0217 | 0.0511 |
| nat_fin_condition_indx | -0.0078* | -1.78 | 0.0775 | 0.0305 |
| 10y_treasury_const_maturity_rate | -0.0073 | -1.66 | 0.1007 | 0.0265 |

#### Slowdown

Top 5 most significant predictors:

| Variable | Coefficient | t-stat | p-value | R² |
|----------|-------------|--------|---------|-----|
| PPI_inflation | 0.0103*** | 2.81 | 0.0059 | 0.0682 |
| PCE_price_index | 0.0093** | 2.54 | 0.0126 | 0.0563 |
| cpi | 0.0093** | 2.53 | 0.0129 | 0.0559 |
| 10y_treasury_const_maturity_rate | -0.0088** | -2.38 | 0.0189 | 0.0500 |
| m2_real_money_supply | 0.0085** | 2.32 | 0.0225 | 0.0473 |

## Coefficient Differences Across Regimes

Found 25 significant coefficient differences:

| Variable | Horizon | Regime1 | Regime2 | Difference | p-value |
|----------|---------|---------|---------|------------|----------|
| 10y_2y_spread | 1 | 0 | 2 | -0.0118 | 0.0208 |
| 10y_2y_spread | 1 | 2 | 3 | 0.0123 | 0.0426 |
| fed_reserve_discount_rate | 1 | 0 | 1 | 0.0136 | 0.0199 |
| fedfunds | 1 | 0 | 1 | 0.0119 | 0.0299 |
| industrial_production | 1 | 0 | 2 | 0.0109 | 0.0330 |
| industrial_production | 1 | 1 | 2 | 0.0135 | 0.0272 |
| retail_sales | 1 | 0 | 2 | 0.0102 | 0.0484 |
| unemployment | 1 | 0 | 2 | -0.0126 | 0.0133 |
| PCE_price_index | 3 | 2 | 3 | -0.0133 | 0.0373 |
| PPI_inflation | 3 | 2 | 3 | -0.0136 | 0.0334 |
| cpi | 3 | 2 | 3 | -0.0132 | 0.0395 |
| m2_real_money_supply | 3 | 2 | 3 | -0.0145 | 0.0229 |
| 10y_2y_spread | 6 | 0 | 1 | 0.0125 | 0.0298 |
| 10y_2y_spread | 12 | 0 | 2 | -0.0152 | 0.0061 |
| 10y_treasury_const_maturity_rate | 12 | 0 | 2 | 0.0113 | 0.0431 |
| 10y_treasury_const_maturity_rate | 12 | 0 | 3 | 0.0128 | 0.0112 |
| PPI_inflation | 12 | 0 | 3 | -0.0101 | 0.0443 |
| fed_reserve_discount_rate | 12 | 0 | 2 | 0.0209 | 0.0003 |
| fed_reserve_discount_rate | 12 | 0 | 3 | 0.0127 | 0.0157 |
| fed_reserve_discount_rate | 12 | 1 | 2 | 0.0173 | 0.0230 |
