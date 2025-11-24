# Understanding P-Values and Composite Score

## Question 1: How are P-Values Combined? What's the Main P-Value?

### Answer: **P-Values are NOT Combined - They Are Separate**

The code keeps **two separate p-values** for each variable:

1. **`correlation_pvalue`** - Tests if correlation ≠ 0
2. **`regression_pvalue`** - Tests if regression coefficient ≠ 0

### Why Two Separate P-Values?

They test **different hypotheses**:

- **Correlation p-value**: Tests if there's a **linear relationship**
  - H₀: Correlation = 0 (no linear relationship)
  - H₁: Correlation ≠ 0 (there is a linear relationship)
  
- **Regression p-value**: Tests if the variable has **predictive power**
  - H₀: Regression coefficient = 0 (no predictive power)
  - H₁: Regression coefficient ≠ 0 (has predictive power)

### Which One is the "Main" P-Value?

**Neither is "main" - they serve different purposes:**

- **Use correlation p-value** if you want to know: "Is there a relationship?"
- **Use regression p-value** if you want to know: "Does this variable predict VIX?"

**For predictive power, the regression p-value is more relevant** because:
- It tests if the variable actually predicts VIX (not just correlates)
- R² shows how much variance is explained
- It's the standard test for predictive models

### Example from Your Results:

```
Variable: bofa_highyield_spread
- correlation_pvalue: 0.00e+00 (highly significant)
- regression_pvalue: 0.00e+00 (highly significant)
- Both are significant → Strong evidence of predictive power
```

```
Variable: unemployment
- correlation_pvalue: 1.21e-02 (significant at p<0.05)
- regression_pvalue: 1.77e-04 (highly significant)
- Both significant → Evidence of predictive power
```

```
Variable: 3m_yield
- correlation_pvalue: 4.45e-02 (significant at p<0.05)
- regression_pvalue: 5.77e-01 (NOT significant)
- Correlation significant but regression NOT → Weak evidence
```

### Code Location:

```python
# Lines 790-808: P-values are stored separately
metrics['correlation_pvalue'] = regime_corr.loc[var_name, 'pvalue']  # From correlation test
metrics['regression_pvalue'] = reg_data.loc[var_name, 'pvalue']      # From regression test

# They are NOT combined - kept as separate columns
```

---

## Question 2: Why Use a Composite Score? What Does It Mean?

### Answer: **Composite Score Ranks by "Importance", Not "Significance"**

The composite score is **NOT based on p-values**. It's based on the **magnitude of effects**:

1. **Correlation strength** (how strong the relationship is)
2. **R²** (how much variance is explained)
3. **Feature importance** (how much the variable contributes in Random Forest)

### Why Not Just Use P-Values?

**P-values test significance, not importance:**
- A variable can be **highly significant** (p < 0.001) but have **weak effect** (small R²)
- A variable can have **strong effect** (high R²) but be **not significant** (if sample size is small)

**Example:**
- Variable A: R² = 0.50, p = 0.001 → Strong AND significant
- Variable B: R² = 0.01, p = 0.001 → Weak but significant (due to large sample)
- Variable C: R² = 0.30, p = 0.10 → Strong but not significant (due to small sample)

**For ranking importance, you want Variable A > Variable C > Variable B**

### How Composite Score is Calculated:

```python
# Step 1: Normalize each metric to 0-1 scale (within each regime)
norm_abs_corr = abs_correlation / max(abs_correlation_in_regime)
norm_r2 = r_squared / max(r_squared_in_regime)
norm_rf_imp = rf_importance / max(rf_importance_in_regime)

# Step 2: Weighted average
relevance_score = (
    0.4 * norm_abs_corr +  # 40% weight on correlation strength
    0.3 * norm_r2 +        # 30% weight on regression R²
    0.3 * norm_rf_imp      # 30% weight on RF importance
)
```

### What the Composite Score Means:

**Interpretation:**
- **Score = 1.0**: Best predictor in that regime (top in all three metrics)
- **Score = 0.5**: Middle-tier predictor
- **Score = 0.0**: Weakest predictor (or no data)

**It's a relative ranking within each regime:**
- Scores are normalized within each regime
- A score of 0.8 in Regime 0 ≠ 0.8 in Regime 1
- It tells you: "This variable is more/less important relative to others in this regime"

### Why These Weights? (40%, 30%, 30%)

**Arbitrary but reasonable:**
- **40% correlation**: Emphasizes linear relationships (most interpretable)
- **30% R²**: Emphasizes predictive power (what you care about)
- **30% RF importance**: Captures non-linear relationships (complementary)

**These weights are subjective** - you could use different weights:
- Equal weights (33%, 33%, 33%)
- More weight on R² (20%, 50%, 30%)
- More weight on RF (20%, 20%, 60%)

### Example Interpretation:

```
Variable: bofa_highyield_spread
- relevance_score: 1.000
- Meaning: Top predictor in this regime (best across all three metrics)

Variable: unemployment
- relevance_score: 0.143
- Meaning: Lower importance relative to other variables in this regime
- But still significant (p < 0.05) → Has predictive power, just weaker
```

---

## Key Distinctions

### P-Values vs. Composite Score:

| Aspect | P-Values | Composite Score |
|--------|----------|-----------------|
| **Purpose** | Test statistical significance | Rank relative importance |
| **Question** | "Is this relationship real?" | "How important is this variable?" |
| **Based on** | Statistical tests (t-tests) | Magnitude of effects |
| **Scale** | 0 to 1 (probability) | 0 to 1 (relative ranking) |
| **Interpretation** | p < 0.05 = significant | Higher = more important |
| **Independent** | Yes - separate tests | No - normalized within regime |

### How to Use Both:

1. **First, check p-values**: Is the relationship statistically significant?
   - If p ≥ 0.05 → Relationship may be due to chance, ignore it
   - If p < 0.05 → Relationship is real, proceed

2. **Then, check composite score**: How important is it relative to others?
   - High score (e.g., > 0.5) → Important predictor
   - Low score (e.g., < 0.2) → Less important, but still significant

3. **Best predictors**: High composite score AND significant p-values

---

## Summary

1. **P-values are NOT combined** - you have two separate p-values:
   - `correlation_pvalue`: Tests linear relationship
   - `regression_pvalue`: Tests predictive power (this is more relevant for prediction)

2. **Composite score is NOT based on p-values** - it's based on:
   - Correlation strength (normalized)
   - R² (normalized)
   - RF importance (normalized)
   - Weighted average: 40% + 30% + 30%

3. **Use both together**:
   - P-values tell you: "Is this real?" (significance)
   - Composite score tells you: "How important is this?" (relative ranking)

4. **For your question about predictive power**:
   - **Primary metric**: `regression_pvalue` (tests if variable predicts VIX)
   - **Secondary metric**: `relevance_score` (ranks importance)
   - **Best evidence**: Significant regression p-value + High relevance score

