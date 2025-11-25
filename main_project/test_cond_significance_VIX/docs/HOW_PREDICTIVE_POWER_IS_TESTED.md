# How Predictive Power is Tested in the Code

This document shows exactly how the code tests predictive power of macro variables for VIX.

## Overview: Three Methods Used

1. **Correlation Analysis** - Tests linear relationships
2. **Linear Regression** - Tests predictive power with R² and statistical significance
3. **Random Forest Feature Importance** - Tests non-linear relationships

All methods use **LAGGED variables** (t-1) to predict VIX at time t.

---

## Method 1: Correlation Analysis

**Location**: `analyze_correlations_by_regime()` method (lines 285-424)

### What It Does:
Tests if lagged macro variables are **correlated** with future VIX.

### Code Flow:

```python
# 1. Use lagged variables (t-1 predicts t)
macro_cols = [f'{var}_lag1' for var in macro_cols]  # Creates lagged versions

# 2. For each regime, calculate correlation
for regime in regimes:
    regime_data = data filtered by regime
    
    for var in macro_cols:
        # Prepare data: VIX at t, macro at t-1
        corr_data = regime_data[['vix', var_lag1]].dropna()
        
        # Calculate correlation
        if using_weights:
            # Weighted correlation (accounts for regime probabilities)
            corr = weighted_correlation(VIX, macro_lag1)
            # Calculate p-value using effective sample size
            t_stat = corr * sqrt((n_eff - 2) / (1 - corr²))
            pvalue = 2 * (1 - t.cdf(abs(t_stat), n_eff - 2))
        else:
            # Standard Pearson correlation
            corr, pvalue = stats.pearsonr(VIX, macro_lag1)
        
        # Store results
        results.append({
            'correlation': corr,
            'pvalue': pvalue,  # Tests if correlation ≠ 0
            'abs_correlation': abs(corr)
        })
```

### What It Tests:
- **Null Hypothesis**: Correlation = 0 (no relationship)
- **Alternative**: Correlation ≠ 0 (there is a relationship)
- **Significance**: p < 0.05 means relationship is statistically significant

### Output:
- Correlation coefficient (-1 to +1)
- p-value (probability correlation is due to chance)
- Absolute correlation (strength of relationship)

---

## Method 2: Linear Regression

**Location**: `analyze_regressions_by_regime()` method (lines 426-604)

### What It Does:
Tests if lagged macro variables can **predict** VIX using linear regression.

### Code Flow:

```python
# 1. Use lagged variables
macro_cols_lagged = [f'{var}_lag1' for var in macro_cols]

# 2. For each regime, run regression
for regime in regimes:
    X = regime_data[macro_cols_lagged]  # Predictors at t-1
    y = regime_data['vix']              # Target at t
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. For each variable individually
    for var in macro_cols:
        X_var = X_scaled[[var_lag1]]
        y_var = VIX
        
        # Fit regression: VIX(t) = α + β * macro(t-1) + ε
        model = LinearRegression()
        model.fit(X_var, y_var, sample_weight=weights)
        
        # Calculate predictions
        y_pred = model.predict(X_var)
        
        # Calculate R-squared (predictive power)
        r2 = model.score(X_var, y_var)  # = 1 - (SS_res / SS_tot)
        
        # Calculate t-statistic and p-value
        n = len(X_var)
        mse = mean((y_var - y_pred)²)
        var_x = variance(X_var)
        se_coef = sqrt(mse / (var_x * n))
        t_stat = coefficient / se_coef
        pvalue = 2 * (1 - t.cdf(abs(t_stat), n - 2))
        
        # Store results
        results.append({
            'coefficient': model.coef_[0],  # β: effect size
            'r_squared': r2,                # Predictive power (0-1)
            't_statistic': t_stat,          # How many std errors from 0
            'pvalue': pvalue                # Tests if coefficient ≠ 0
        })
```

### What It Tests:
- **Null Hypothesis**: β = 0 (variable has no predictive power)
- **Alternative**: β ≠ 0 (variable predicts VIX)
- **R²**: Proportion of VIX variance explained (0 = no power, 1 = perfect)
- **Significance**: p < 0.05 means coefficient is significantly different from 0

### Output:
- Coefficient (β): How much VIX changes per unit change in macro variable
- R²: Predictive power (0-1 scale)
- t-statistic: How many standard errors the coefficient is from 0
- p-value: Probability coefficient is due to chance

---

## Method 3: Random Forest Feature Importance

**Location**: `analyze_feature_importance_by_regime()` method (lines 606-742)

### What It Does:
Tests predictive power using **non-linear** relationships and interactions.

### Code Flow:

```python
# 1. Use lagged variables
macro_cols_lagged = [f'{var}_lag1' for var in macro_cols]

# 2. For each regime
for regime in regimes:
    X = regime_data[macro_cols_lagged]  # All predictors at t-1
    y = regime_data['vix']              # Target at t
    
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. Fit Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,      # 100 trees
        max_depth=10,          # Limit depth to prevent overfitting
        min_samples_split=5,  # Minimum samples to split
        random_state=42
    )
    rf.fit(X_scaled, y, sample_weight=weights)
    
    # 4. Get feature importance
    # Measures how much each variable contributes to predictions
    importance = rf.feature_importances_
    
    # 5. Calculate R² (overall model fit)
    y_pred = rf.predict(X_scaled)
    r2 = rf.score(X_scaled, y)
    
    # Store results
    results.append({
        'variable': var_name,
        'importance': importance,  # Relative importance (0-1)
        'model_r2': r2            # Overall model predictive power
    })
```

### What It Tests:
- **Feature Importance**: How much each variable contributes to predictions
  - Higher importance = more predictive power
  - Captures non-linear relationships and interactions
- **Model R²**: Overall predictive power of all variables together

### Output:
- Feature importance: Relative contribution of each variable (0-1 scale)
- Model R²: Overall predictive power when using all variables

---

## How Results Are Combined

**Location**: `create_summary_report()` method (lines 744-855)

### Composite Relevance Score:

```python
# For each variable in each regime:
# 1. Normalize each metric to 0-1 scale
norm_abs_corr = abs_correlation / max(abs_correlation)
norm_r2 = r_squared / max(r_squared)
norm_rf_imp = rf_importance / max(rf_importance)

# 2. Combine with weights
relevance_score = (
    0.4 * norm_abs_corr +  # 40% weight on correlation
    0.3 * norm_r2 +        # 30% weight on regression R²
    0.3 * norm_rf_imp      # 30% weight on RF importance
)
```

This creates a **composite score** ranking variables by overall predictive power.

---

## Statistical Significance Testing

### Correlation Significance:
```python
# Tests: H₀: ρ = 0 vs H₁: ρ ≠ 0
t_stat = corr * sqrt((n - 2) / (1 - corr²))
pvalue = 2 * (1 - t.cdf(abs(t_stat), n - 2))
```

### Regression Significance:
```python
# Tests: H₀: β = 0 vs H₁: β ≠ 0
t_stat = coefficient / standard_error
pvalue = 2 * (1 - t.cdf(abs(t_stat), n - k - 1))
```

### Significance Levels:
- **p < 0.001**: *** (highly significant)
- **p < 0.01**: ** (very significant)
- **p < 0.05**: * (significant)
- **p ≥ 0.05**: Not significant

---

## Key Points

1. **All methods use LAGGED variables** (t-1) to ensure prediction, not correlation
2. **Statistical significance** is tested with proper t-tests and p-values
3. **Multiple metrics** are used (correlation, R², feature importance)
4. **Regime-conditional** analysis tests relationships separately in each regime
5. **Weighted analysis** accounts for regime probability uncertainty

---

## Limitations

⚠️ **These are IN-SAMPLE tests**:
- Models are fit on the full dataset
- No out-of-sample validation
- Results show relationships in historical data, not future prediction

To test **real predictive power**, you would need:
- Walk-forward validation (train on past, test on future)
- Out-of-sample R²
- Comparison to baseline models

