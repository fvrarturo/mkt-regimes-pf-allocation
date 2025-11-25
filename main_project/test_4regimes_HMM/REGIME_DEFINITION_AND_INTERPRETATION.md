# Regime Definition Process and Interpretation Guide

## Table of Contents
1. [Regime Definition Process](#regime-definition-process)
2. [Interpretation of Scatter Plot Values](#interpretation-of-scatter-plot-values)
3. [Regime Characteristics](#regime-characteristics)

---

## Regime Definition Process

### Overview

The regime detection system uses a Hidden Markov Model (HMM) to identify 4 distinct economic regimes based on growth and inflation characteristics. The process involves three main steps: feature selection, regime discovery, and regime interpretation.

### 1. Features Used in HMM (4 Total Features)

The HMM model uses **only 4 features** to discover regimes:

#### Macro Features (2):
- **`gdp`**: GDP growth proxy
  - Preferred column: `zscore_pct_change_mom` (z-scored month-over-month percentage change)
  - Fallback: `pct_change_mom`, `zscore_value`, or `value`
  
- **`PCE_price_index`**: Inflation proxy
  - Preferred column: `zscore_pct_change_mom` (z-scored month-over-month percentage change)
  - Fallback: `pct_change_mom`, `zscore_value`, or `value`

#### Sentiment Features (2):
- **`ec_growth_sentiment`**: Economic growth sentiment score
- **`inflation_sentiment`**: Inflation sentiment score

### 2. Feature Processing

1. **Standardization**: Macro and sentiment features are standardized separately using `StandardScaler`
   - Macro features: standardized together
   - Sentiment features: standardized together

2. **Weighted Combination**: Features are combined with weights:
   - `macro_weight = 0.4` (default)
   - `sentiment_weight = 0.6` (default)
   - Formula: `combined_features = macro_scaled * 0.4 + sentiment_scaled * 0.6`

3. **HMM Discovery**: The HMM discovers 4 hidden states from the 4-dimensional feature space
   - Uses Gaussian HMM with diagonal covariance
   - Multiple random initializations (default: 10) to avoid local optima
   - No explicit labels - states are discovered purely from data patterns

###3. Regime Interpretation (Post-HMM)

After the HMM discovers 4 hidden states, they are assigned to quadrants based on growth/inflation characteristics:

#### Step 1: Calculate Proxies

For each discovered regime, calculate:

- **Growth Proxy** = Average of:
  - `gdp` (z-scored month-over-month percentage change)
  - `ec_growth_sentiment` (raw sentiment score)

- **Inflation Proxy** = Average of:
  - `PCE_price_index` (z-scored month-over-month percentage change)
  - `inflation_sentiment` (raw sentiment score)

#### Step 2: Assign to Quadrants

Regimes are classified using **absolute thresholds** relative to the historical mean (zero):

- **High Growth** = `avg_growth >= 0` (above historical average)
- **Low Growth** = `avg_growth < 0` (below historical average)
- **High Inflation** = `avg_inflation >= 0` (above historical average)
- **Low Inflation** = `avg_inflation < 0` (below historical average)

**Note**: The HMM discovers regimes from data patterns. Not all theoretical quadrants may be present in the data. The classification reflects the actual regimes found.

#### Result: Actual Regimes Discovered

Based on the data, the following regimes have been identified:

1. **High Growth / High Inflation** (Expansion with Rising Prices)
   - Strong economic growth with increasing inflation
   - Typically requires monetary tightening
   - **Example**: R1

2. **High Growth / Low Inflation** (Goldilocks Economy)
   - Ideal conditions: strong growth with low inflation
   - Best environment for risk assets
   - **Example**: R2

3. **Low Growth / Low Inflation** (Recession/Deflation)
   - Weak growth with low inflation
   - Typically requires monetary easing and fiscal stimulus
   - **Examples**: R0 (moderate contraction), R3 (extreme contraction)
   - R0 and R3 represent different severities of contractionary conditions

**Note**: The "Low Growth / High Inflation" (Stagflation) quadrant is **not present** in the current data. All Low Growth regimes in the dataset have below-average inflation, indicating that stagflationary periods are not captured as distinct regimes in this analysis.

---

## Interpretation of Scatter Plot Values

### Overview

The scatter plot "Regime Characteristics: Growth vs Inflation" visualizes the average growth and inflation characteristics of each regime. Understanding the axes is crucial for proper interpretation.

### What the Values Represent

The axes show **standardized values** (z-scores) relative to the **historical mean** across all time periods:

- **0** = Historical mean (average across all time periods)
- **Positive values** = Above average conditions
- **Negative values** = Below average conditions

### How Values are Calculated

#### Average Growth (X-axis)

Calculated as the mean of two components:

1. **`gdp`**: 
   - Z-scored month-over-month percentage change
   - Preferred column: `zscore_pct_change_mom`
   - This is already standardized (mean=0, std=1 across all time)

2. **`ec_growth_sentiment`**: 
   - Raw sentiment score (not standardized in this calculation)

**Formula**: `avg_growth = mean([gdp, ec_growth_sentiment])` for all observations in the regime

#### Average Inflation (Y-axis)

Calculated as the mean of two components:

1. **`PCE_price_index`**: 
   - Z-scored month-over-month percentage change
   - Preferred column: `zscore_pct_change_mom`
   - This is already standardized (mean=0, std=1 across all time)

2. **`inflation_sentiment`**: 
   - Raw sentiment score (not standardized in this calculation)

**Formula**: `avg_inflation = mean([PCE_price_index, inflation_sentiment])` for all observations in the regime

### Interpreting the Values

#### Average Growth = 0
- GDP growth and growth sentiment are at their **historical average**
- Neither expansionary nor contractionary

#### Average Growth > 0
- **Above-average growth conditions**
- GDP growth and/or growth sentiment exceed historical norms
- Indicates expansionary economic environment

#### Average Growth < 0
- **Below-average growth conditions**
- GDP growth and/or growth sentiment below historical norms
- Indicates contractionary economic environment

#### Average Inflation = 0
- PCE inflation and inflation sentiment are at their **historical average**
- Neither inflationary nor deflationary

#### Average Inflation > 0
- **Above-average inflation conditions**
- PCE inflation and/or inflation sentiment exceed historical norms
- Indicates inflationary pressure

#### Average Inflation < 0
- **Below-average inflation conditions**
- PCE inflation and/or inflation sentiment below historical norms
- Indicates deflationary pressure

### Interpreting Your Specific Plot

Based on the scatter plot visualization:

- **R1 (Top-Right Quadrant)**: 
  - Above-average growth AND above-average inflation
  - **Regime**: High Growth / High Inflation (Expansion with Rising Prices)
  - **Interpretation**: Expansionary period with rising prices, typically requiring monetary tightening

- **R2 (Bottom-Right Quadrant)**: 
  - Above-average growth, below-average inflation
  - **Regime**: High Growth / Low Inflation (Goldilocks Economy)
  - **Interpretation**: Optimal economic conditions - strong growth without inflation pressure, best for risk assets

- **R0 (Bottom-Left Quadrant)**: 
  - Below-average growth AND below-average inflation
  - **Regime**: Low Growth / Low Inflation (Recession/Deflation - Moderate)
  - **Interpretation**: Moderate contractionary conditions, requires monetary easing and fiscal stimulus

- **R3 (Far Bottom-Left Quadrant)**: 
  - Very low growth AND very low inflation
  - **Regime**: Low Growth / Low Inflation (Recession/Deflation - Extreme)
  - **Interpretation**: Severe contractionary conditions, requires aggressive monetary and fiscal stimulus
  - **Note**: R3 represents the most extreme contractionary period in the dataset

### Important Notes

1. **Mixed Scaling**: The values are a **mixture** of:
   - Z-scored macro data (already standardized)
   - Raw sentiment scores (not standardized)
   
   This means the axes are not pure z-scores, but rather represent **relative positioning**:
   - Positive = above average
   - Negative = below average
   - Zero = average

2. **Reference Lines**: The dashed lines at 0.0 on both axes serve as thresholds:
   - **Vertical line (Growth = 0)**: Separates High Growth (right) from Low Growth (left)
   - **Horizontal line (Inflation = 0)**: Separates High Inflation (top) from Low Inflation (bottom)

3. **Quadrant Assignment**: The four quadrants created by these reference lines represent the theoretical regime space:
   - **Top-Right**: High Growth / High Inflation (R1)
   - **Top-Left**: Low Growth / High Inflation (Stagflation - **not present in current data**)
   - **Bottom-Right**: High Growth / Low Inflation (R2)
   - **Bottom-Left**: Low Growth / Low Inflation (R0, R3)
   
   **Note**: The actual regimes found in the data occupy 3 of the 4 theoretical quadrants. The Low Growth / High Inflation (Stagflation) quadrant is not represented in the current dataset.

---

## Regime Characteristics

### Summary Table

| Regime | Growth Level | Inflation Level | Economic Interpretation | Severity | Typical Policy Response |
|--------|--------------|-----------------|-------------------------|----------|------------------------|
| R1 | High | High | Expansion with Rising Prices | Moderate | Monetary tightening |
| R2 | High | Low | Goldilocks Economy | Optimal | Maintain accommodative policy |
| R0 | Low | Low | Recession/Deflation | Moderate | Monetary easing, fiscal stimulus |
| R3 | Low | Low | Recession/Deflation | Extreme | Aggressive monetary/fiscal stimulus |

**Key Observations**:
- **3 distinct regime types** are identified (not 4): High Growth/High Inflation, High Growth/Low Inflation, and Low Growth/Low Inflation
- **Two Low Growth/Low Inflation regimes** (R0 and R3) represent different severities of contraction
- **No Stagflation regime** (Low Growth/High Inflation) is present in the data - all Low Growth periods have below-average inflation

### Key Insights

1. **Regime Discovery**: The HMM discovers regimes purely from data patterns - no explicit labels are provided during training. The number and characteristics of regimes emerge from the data.

2. **Classification Method**: Regimes are classified using **absolute thresholds** (relative to historical mean = 0), not relative ranking. This ensures "High" means above average and "Low" means below average.

3. **Actual vs. Theoretical Regimes**: While the framework allows for 4 theoretical quadrants (High/Low Growth × High/Low Inflation), the data reveals only 3 distinct regime types. The Low Growth/High Inflation (Stagflation) quadrant is not present.

4. **Regime Severity**: The Low Growth/Low Inflation category contains two regimes (R0 and R3) representing different severities of contraction, suggesting the HMM can distinguish between moderate and extreme recessionary conditions.

5. **Feature Selection**: Only 4 features (GDP, PCE, growth sentiment, inflation sentiment) are used to ensure regimes align with the growth/inflation framework.

6. **Visualization**: The scatter plot provides an intuitive way to understand regime characteristics and their relative positioning in the growth-inflation space.

---

## Technical Details

### Code References

- **Feature Selection**: `regime_detection_hmm.py`, lines 313-342 (`prepare_features` method)
- **Regime Interpretation**: `regime_detection_hmm.py`, lines 692-834 (`interpret_regimes` method)
- **Data Loading**: `regime_detection_hmm.py`, lines 107-156 (`load_macro_data` method)

### Data Sources

- **Macro Data**: `data/macro_processed/selection/`
  - `gdp_processed.csv`
  - `PCE_price_index_processed.csv`
  
- **Sentiment Data**: `data/news_data/sentiment_scores.csv`
  - `ec_growth_sentiment`
  - `inflation_sentiment`

---

*Document created: 2025*
*Last updated: Based on regime_detection_hmm.py implementation*

