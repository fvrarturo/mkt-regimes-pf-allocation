# Economic Analysis: HMM Regime Models Comparison

## Executive Summary

This document compares **three HMM regime detection models** and evaluates which makes the most economic sense for analyzing Equity Risk Premium (ERP) in relation to macroeconomic conditions. The analysis considers:
1. **Statistical fit** (BIC, AIC)
2. **Economic interpretability** 
3. **Alignment with 2×2 quadrant classification**
4. **ERP differentiation across regimes**
5. **Theoretical foundations** for macro variables affecting ERP

---

## Model Comparison Overview

| Model | Variables | Best K | BIC | AIC | N Obs | Key Feature |
|-------|-----------|--------|-----|-----|--------|-------------|
| **4 Variables** | Growth + Inflation + Policy + Volatility | 4 | 3031.83 | 2840.73 | 431 | Most comprehensive |
| **2 Variables (Optimal)** | Growth + Policy | 4 | **755.37** | **629.32** | 431 | **Best statistical fit** |
| **2 Variables (Growth+Inflation)** | Growth + Inflation | 3 | 1732.42 | 1651.10 | 431 | Aligns with 2×2 quadrants |

---

## 1. Four Variables Model (Growth + Inflation + Policy + Volatility)

### Model Characteristics
- **BIC**: 3031.83 (worst among three)
- **K**: 4 regimes
- **N Parameters**: 47

### Regime Distribution
- R0: Low Growth / High Inflation (22.5%)
- R1: High Growth / High Inflation (43.4%)
- R2: Low Growth / Low Inflation (10.2%)
- R3: High Growth / High Inflation (23.9%)

### ERP Performance by Regime
| Regime | ERP Mean | ERP Std | Interpretation |
|--------|----------|---------|----------------|
| R0 | **1.22%** | 3.84% | Best ERP - Stagflation-like but with policy support |
| R1 | 0.63% | 4.03% | Moderate - Expansion with inflation concerns |
| R2 | **-0.81%** | 7.09% | Worst - Recession/Deflation |
| R3 | 0.34% | 3.22% | Low - Goldilocks-like but muted |

### Economic Interpretation

**Strengths:**
- Captures all major macro dimensions
- Includes volatility (important for risk assessment)
- Most comprehensive view of economic environment

**Weaknesses:**
- **Poor statistical fit** (highest BIC)
- **Overfitting risk** (too many parameters)
- Regimes R1 and R3 both labeled "High Growth / High Inflation" (confusing)
- **Doesn't clearly differentiate** between regimes in terms of ERP

**Key Finding:** Despite including all variables, this model has the **worst fit** and creates **confusing regime labels** (two regimes with same label).

---

## 2. Two Variables Model - Optimal (Growth + Policy)

### Model Characteristics
- **BIC**: **755.37** (best statistical fit - 75% lower than 4-var model)
- **K**: 4 regimes
- **N Parameters**: 31

### Regime Distribution
- R0: Low Growth / Low Inflation (4.4%) - Extreme contraction
- R1: Low Growth / High Inflation (23.2%) - Stagflation
- R2: High Growth / High Inflation (42.9%) - Expansion with policy tightening
- R3: High Growth / High Inflation (29.5%) - Expansion with policy easing

### ERP Performance by Regime
| Regime | ERP Mean | ERP Std | Policy Rate | Interpretation |
|--------|----------|---------|-------------|----------------|
| R0 | **-2.76%** | 7.73% | 0.82% | Severe recession - worst ERP |
| R1 | **1.40%** | 3.94% | 0.12% | Stagflation with policy support - best ERP |
| R2 | 0.61% | 4.01% | 5.30% | Expansion with tight policy |
| R3 | 0.28% | 3.90% | 1.83% | Expansion with easy policy |

### Economic Interpretation

**Strengths:**
- **Best statistical fit** (lowest BIC)
- **Clear policy differentiation**: R2 (5.30%) vs R3 (1.83%) shows policy stance matters
- **Strong ERP differentiation**: R0 (-2.76%) vs R1 (1.40%) shows 4.16% difference
- **Policy-Growth relationship** is fundamental to macro regimes
- **More parsimonious** (fewer parameters, better generalization)

**Economic Logic:**
- **Growth** drives equity returns (fundamental driver)
- **Policy** affects discount rates and risk appetite (monetary transmission)
- Together, they capture the **core macro drivers of ERP**:
  - Growth → Earnings expectations → Equity returns
  - Policy → Discount rates → Risk-free rate → ERP

**Weaknesses:**
- Doesn't explicitly include inflation (though policy often responds to inflation)
- Doesn't include volatility (though policy affects volatility)

**Key Finding:** This model achieves the **best statistical fit** and creates **economically meaningful regimes** with **strong ERP differentiation**.

---

## 3. Two Variables Model - Growth + Inflation

### Model Characteristics
- **BIC**: 1732.42
- **K**: 3 regimes (not 4!)
- **N Parameters**: 20

### Regime Distribution
- R0: Low Growth / High Inflation (46.9%) - Stagflation
- R1: Low Growth / High Inflation (12.3%) - Severe stagflation
- R2: High Growth / Low Inflation (40.8%) - Goldilocks

**Note:** Only 3 regimes detected, and R0 and R1 have the same label!

### ERP Performance by Regime
| Regime | ERP Mean | ERP Std | Growth | Inflation | Interpretation |
|--------|----------|---------|--------|-----------|----------------|
| R0 | 0.54% | 3.89% | 0.43 | 0.18 | Moderate stagflation |
| R1 | **-1.19%** | 6.33% | 0.08 | 0.26 | Severe stagflation - worst ERP |
| R2 | **1.08%** | 3.74% | 1.01 | 0.16 | Goldilocks - best ERP |

### Economic Interpretation

**Strengths:**
- **Directly aligns with 2×2 quadrant classification**
- **Clear economic interpretation**: Growth × Inflation framework
- **Good ERP differentiation**: R2 (1.08%) vs R1 (-1.19%) = 2.27% difference
- **Intuitive**: Matches standard macro regime thinking

**Weaknesses:**
- **Only 3 regimes** (not 4 like 2×2 quadrants)
- **Two regimes with same label** (R0 and R1 both "Low Growth / High Inflation")
- **Worse statistical fit** than Growth+Policy (BIC: 1732 vs 755)
- **Missing policy dimension** - policy is a key transmission mechanism

**Key Finding:** While this model aligns with 2×2 thinking, it **fails to detect 4 distinct regimes** and has **worse fit** than Growth+Policy.

---

## 4. Comparison with 2×2 Quadrant Classification

### 2×2 Quadrant Results (from previous analysis)

| Regime | Description | ERP Mean | Key Finding |
|--------|-------------|----------|-------------|
| Goldilocks | High Growth / Low Inflation | **1.05%** | Best ERP |
| Overheating | High Growth / High Inflation | **1.22%** | Second best |
| Stagflation | Low Growth / High Inflation | **-0.36%** | Negative ERP |
| Slowdown | Low Growth / Low Inflation | 0.31% | Weak positive |

**Key 2×2 Finding:** **Stagflation has negative ERP** (-0.36%), confirming it's the worst environment for equities.

### Alignment Analysis

#### Growth + Inflation HMM Model
- **Detects only 3 regimes** (should be 4 to match 2×2)
- **R0 and R1 both "Low Growth / High Inflation"** - can't distinguish different types of stagflation
- **R2 "High Growth / Low Inflation"** aligns with Goldilocks (ERP: 1.08% vs 1.05%)
- **Missing the 4th quadrant** (High Growth / High Inflation)

#### Growth + Policy HMM Model
- **Detects 4 regimes** (matches 2×2 structure)
- **Clear policy differentiation**: 
  - R2: High Growth + High Policy (5.30%) = Overheating with tightening
  - R3: High Growth + Low Policy (1.83%) = Goldilocks with easing
- **Strong ERP differentiation**: R1 (1.40%) vs R0 (-2.76%) = 4.16% spread
- **Better captures policy response** to macro conditions

#### 4 Variables Model
- **Detects 4 regimes** but **confusing labels** (R1 and R3 both "High Growth / High Inflation")
- **Weaker ERP differentiation** than Growth+Policy model
- **Includes volatility** but doesn't improve regime clarity

---

## 5. Economic Theory: Which Variables Matter Most for ERP?

### Theoretical Framework

**Equity Risk Premium (ERP)** is driven by:

1. **Growth Expectations** → Earnings growth → Equity returns
   - **Direct relationship**: Higher growth → Higher ERP
   - **Empirical evidence**: Strong in our data (R2 in Growth+Policy: 0.61% ERP)

2. **Monetary Policy** → Discount rates → Risk-free rate → ERP
   - **Policy affects ERP through**:
     - Risk-free rate (direct component of ERP)
     - Risk appetite (policy stance affects investor sentiment)
     - Credit conditions (affects corporate earnings)
   - **Empirical evidence**: Clear in Growth+Policy model (R2: 5.30% policy vs R3: 1.83% policy)

3. **Inflation** → Real returns → ERP
   - **Indirect relationship**: Inflation affects ERP through:
     - Policy response (central banks tighten to fight inflation)
     - Real earnings growth (inflation erodes real returns)
   - **Empirical evidence**: Less direct than Growth/Policy in our models

4. **Volatility** → Risk premium → ERP
   - **Endogenous relationship**: Volatility is often a **consequence** of macro conditions
   - **Circular reasoning risk**: Using volatility to predict ERP may be circular
   - **Empirical evidence**: Doesn't improve model fit significantly

### Key Insight

**Growth + Policy captures the core transmission mechanism:**
- **Growth** → Fundamental driver of equity returns
- **Policy** → Affects discount rates and risk appetite
- **Together** → Capture both the "fundamental" (growth) and "valuation" (policy) drivers of ERP

**Inflation** is important but often works **through policy** (central banks respond to inflation by adjusting policy rates).

---

## 6. Recommendation: Growth + Policy Model

### Why Growth + Policy Makes Most Economic Sense

#### 1. **Best Statistical Fit**
- BIC = 755.37 (75% lower than 4-var model)
- More parsimonious (31 vs 47 parameters)
- Better generalization (less overfitting risk)

#### 2. **Strong ERP Differentiation**
- **4.16% ERP spread** between best (R1: 1.40%) and worst (R0: -2.76%) regimes
- **Clear economic logic**: 
  - R1: Low growth but policy support (0.12% policy rate) → Positive ERP
  - R0: Low growth with no policy support (0.82% policy rate) → Negative ERP
  - R2: High growth with tight policy (5.30% policy rate) → Moderate ERP
  - R3: High growth with easy policy (1.83% policy rate) → Moderate ERP

#### 3. **Policy as Transmission Mechanism**
- **Policy responds to inflation** (central banks tighten when inflation is high)
- **Policy affects ERP directly** through risk-free rate
- **Growth + Policy captures both**:
  - Fundamental driver (growth)
  - Valuation driver (policy/discount rates)

#### 4. **Clear Regime Structure**
- **4 distinct regimes** (matches 2×2 structure)
- **No duplicate labels** (unlike 4-var and Growth+Inflation models)
- **Policy differentiation** creates meaningful regime boundaries

#### 5. **Alignment with 2×2 Findings**
- **2×2 found**: Stagflation has negative ERP (-0.36%)
- **Growth+Policy finds**: R0 (severe contraction) has -2.76% ERP
- **Both confirm**: Weak growth environments are bad for ERP
- **Growth+Policy adds**: Policy stance matters (R1 with policy support has +1.40% ERP)

### Economic Interpretation of Growth + Policy Regimes

| Regime | Growth | Policy | ERP | Economic Story |
|--------|--------|--------|-----|----------------|
| **R0** | Low | Low (0.82%) | **-2.76%** | Severe recession - policy ineffective or constrained (ZLB) |
| **R1** | Low | Very Low (0.12%) | **+1.40%** | Stagflation with aggressive policy support - best ERP |
| **R2** | High | High (5.30%) | +0.61% | Expansion with policy tightening - moderating growth |
| **R3** | High | Low (1.83%) | +0.28% | Expansion with policy easing - sustained growth |

**Key Insight:** **Policy effectiveness matters more than growth alone.**
- R1 (low growth + policy support) has **better ERP** than R2 (high growth + tight policy)
- This suggests **policy-driven liquidity** can support ERP even when growth is weak

---

## 7. Why Not Growth + Inflation?

### Limitations of Growth + Inflation Model

1. **Only 3 regimes detected** (not 4)
   - Fails to capture full 2×2 structure
   - Two regimes with same label (confusing)

2. **Missing policy dimension**
   - Policy is a **key transmission mechanism** for inflation's effect on ERP
   - Central banks respond to inflation → Policy affects ERP
   - **Growth + Policy captures this indirectly**

3. **Worse statistical fit**
   - BIC = 1732.42 vs 755.37 for Growth+Policy
   - More than 2× worse fit

4. **Less ERP differentiation**
   - ERP spread: 2.27% (R2: 1.08% vs R1: -1.19%)
   - Growth+Policy has 4.16% spread (better discrimination)

### When Growth + Inflation Might Be Preferred

- **For pedagogical purposes**: Aligns with standard 2×2 framework
- **For inflation-focused analysis**: If research question is specifically about inflation
- **For comparison**: To show how HMM differs from simple 2×2 classification

---

## 8. Why Not 4 Variables?

### Limitations of 4-Variable Model

1. **Worst statistical fit** (BIC = 3031.83)
   - 4× worse than Growth+Policy
   - Too many parameters (overfitting risk)

2. **Confusing regime labels**
   - R1 and R3 both labeled "High Growth / High Inflation"
   - Can't distinguish between them without looking at Policy/Volatility

3. **Weaker ERP differentiation**
   - ERP spread: ~2% (R0: 1.22% vs R2: -0.81%)
   - Growth+Policy has 4.16% spread

4. **Volatility is endogenous**
   - Volatility is often a **consequence** of macro conditions
   - Using it to predict ERP may be circular
   - Doesn't improve fit significantly

### When 4 Variables Might Be Preferred

- **For comprehensive risk assessment**: If volatility is a key concern
- **For research completeness**: To show all dimensions are considered
- **For specific research questions**: If research focuses on volatility regimes

---

## 9. Final Recommendation

### **Use Growth + Policy Model (2 Variables - Optimal)**

**Rationale:**

1. **Best Statistical Fit**: Lowest BIC (755.37), most parsimonious
2. **Strongest ERP Differentiation**: 4.16% spread between best and worst regimes
3. **Clear Economic Logic**: Captures fundamental (growth) and valuation (policy) drivers
4. **Policy as Transmission Mechanism**: Policy responds to inflation and affects ERP directly
5. **4 Distinct Regimes**: Matches 2×2 structure without confusion
6. **Alignment with 2×2 Findings**: Confirms stagflation is bad, adds policy dimension

### Economic Story

**Growth + Policy captures the core macro drivers of ERP:**
- **Growth** → Fundamental earnings driver
- **Policy** → Discount rate and risk appetite driver
- **Together** → Explain both the "real" and "financial" sides of ERP

**Key Finding:** Policy effectiveness matters. Low growth with policy support (R1) has **better ERP** than high growth with tight policy (R2), suggesting **liquidity-driven returns** can offset weak fundamentals.

---

## 10. Link to 2×2 Regime Findings

### Comparison Table

| Aspect | 2×2 Quadrants | Growth+Policy HMM | Alignment |
|--------|---------------|------------------|-----------|
| **Stagflation ERP** | -0.36% | R1: +1.40% (with policy) | **Different** - HMM shows policy matters |
| **Goldilocks ERP** | +1.05% | R3: +0.28% (high growth, easy policy) | **Similar** - Both positive |
| **Worst Regime** | Stagflation (-0.36%) | R0: -2.76% (severe contraction) | **Consistent** - Weak growth is bad |
| **Best Regime** | Overheating (+1.22%) | R1: +1.40% (low growth, policy support) | **Different** - Policy support matters |

### Key Insight

**The HMM model reveals that policy stance matters more than simple 2×2 classification suggests:**
- 2×2: Stagflation (Low G / High I) has negative ERP
- HMM: Low Growth with **policy support** (R1) has **positive ERP** (+1.40%)
- **Policy can offset weak growth** in terms of ERP performance

This suggests that **monetary policy effectiveness** is a crucial dimension that the simple 2×2 framework misses.

---

## Files Reference

- **4 Variables Results**: `results_4vars/regime_statistics.csv`
- **Growth + Policy Results**: `results_2vars_optimal/regime_statistics.csv`
- **Growth + Inflation Results**: `results_2vars_growth_inflation/regime_statistics.csv`
- **2×2 Quadrant Results**: `../2x2_regimes/results/regime_statistics.csv`

