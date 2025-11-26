## 0. Setup & Core Variables

**Steps**

* Choose sample (e.g. 1990–2025, weekly or monthly).
* Define:

  * **ERP** = stock–bond excess return (e.g. equity index – Treasury index).
  * 4 **macro indices**:

    * Growth
    * Inflation
    * Monetary policy stance
    * Market volatility
  * 4 **sentiment indices** (from agentic AI):

    * Growth, inflation, policy, vol sentiment
* Standardize variables (z-scores) for modeling.

**Key outputs**

* Table: description of all variables (source, frequency, units).
* Time series plots:

  * ERP
  * Each macro index
  * Each sentiment index
* Correlation heatmap (macro vs sentiment vs ERP).

---

## 1. Simple Macro Regimes (2×2 Growth × Inflation)

**Goal:** Intuitive macro quadrants, used as a pedagogical regime layer.

### Models / definitions

* Define thresholds for:

  * Growth: High vs Low (median or economically meaningful threshold).
  * Inflation: High vs Low.
* 4 regimes:

  1. High G / Low I – “Goldilocks”
  2. High G / High I – “Overheating”
  3. Low G / High I – “Stagflation”
  4. Low G / Low I – “Slowdown / Disinflation”

**Steps**

* Classify each period into one of 4 regimes.
* For each regime compute:

  * Avg growth, inflation, policy, vol
  * Avg ERP, ERP volatility, tail stats.

**Key outputs**

* Scatter plot: Growth vs Inflation, colored by regime (dots = weeks/months).
* Table: regime summary stats (macro averages, ERP mean/vol/skew).
* Boxplots: ERP by regime.
* Simple tests:

  * t-tests for equality of ERP means across regimes.
  * F-test/ANOVA for ERP variation across regimes.

---

## 2. Full Regime Model: HMM with 4 Macro Variables

**Goal:** Main “quantitative regime engine” using all macro indices.

### Models

* Hidden Markov Model (Gaussian HMM) with K = 3 or 4 regimes.
* Input features: [growth, inflation, policy, volatility].

**Steps**

* Fit HMM on standardized macro indices.
* Extract for each date:

  * Regime assignment (most likely state).
  * Regime probabilities ( p_{r,t} ).
* For each regime:

  * Compute average macro values and ERP.

**Key outputs**

* Time plot: regime probabilities over time (stacked area or colored bands).
* Table: regime characterization (avg macro + ERP, vol, tail risk).
* Transition matrix heatmap.
* Comparison chart: HMM regimes vs simple 4 quadrants (cross-tab).

**Stat tests**

* Compare ERP means across HMM regimes (t-tests/ANOVA).
* Likelihood-based metrics: AIC/BIC for K = 2,3,4 to justify regime count.

## 2.b HMM with Concatenated Macro + Sentiment

**Goal:** See whether regimes become more informative when the model sees both *actual macro* and *text-based macro sentiment*.

### Models

* Gaussian HMM with K = 3 or 4 regimes (same K as macro-only version so they’re comparable).
* Input features:
  [
  X_t = [\text{growth}, \text{inflation}, \text{policy}, \text{vol}, \text{growth_sent}, \text{infl_sent}, \text{policy_sent}, \text{vol_sent}]
  ]
  (all standardized).

### Steps

* Standardize all 8 variables.
* Fit HMM on this 8D series.
* Extract for each date:

  * Most likely regime
  * Regime probabilities (p_{r,t})
* For each regime:

  * Compute average macro indices and sentiment indices
  * Compute ERP mean, volatility, tail risk

### Key outputs

* Time plot of regime probabilities (compare visually to macro-only HMM).
* Table: regime profiles including both macro and sentiment (e.g. “low growth, high inflation, negative growth sentiment”).
* Cross-tab:

  * Macro-only HMM regimes vs Macro+Sentiment HMM regimes (how often they agree / differ).
* Possibly a plot of:

  * how often big drawdowns occur in *sentiment-enriched* regimes vs macro-only regimes.

### Stat tests / comparisons

* Compare ERP means and distributional characteristics (e.g. volatility, skew, tails) across regimes (t-tests, ANOVA, tail-tests).
* Assess if sentiment-enriched regimes capture a higher concentration of extreme ERP events (tail isolation).
* Agreement rates between regime assignments in macro-only and macro+sentiment models.

_Note: Predictive/statistical fit metrics (e.g., R², log-likelihood, regime probabilities as features for ERP forecasting) appear in Section 4 below for all regime models._

---

## 2.c GMM Regime Model (Macro and Macro+Sentiment Variants)

**Goal:** Provide a **non-time-smoothed** clustering baseline – regimes come from distributional clusters, not Markov dynamics. This is a robustness / sanity check on the HMM regimes.

### Models

Two versions (you don’t have to show both in equal detail in the talk):

1. **GMM on 4 macro variables:**

   * Input: ([ \text{growth}, \text{inflation}, \text{policy}, \text{vol} ])
2. **Optional GMM on macro + sentiment (8D)**, same concatenation as above.

Each with K = 3 or 4 clusters (same K as HMM, again to keep results comparable).

### Steps

* Standardize variables (4D or 8D).
* Fit Gaussian Mixture Model with K components.
* For each date:

  * Get cluster assignment (hard) and posterior membership probabilities (soft).
* For each cluster:

  * Compute average macro values (and sentiment if included).
  * Compute ERP mean, volatility, tail risk.

### Key outputs

* Scatter plots in reduced space (e.g. first 2 PCs of the macro variables) colored by GMM cluster.
* Table: cluster characteristics (means of macro / sentiment / ERP).
* Cross-tabs:

  * GMM clusters vs macro-only HMM regimes.
  * GMM clusters vs simple 4 growth–inflation quadrants.
* Time series: cluster assignment over time (to show that GMM can still appear persistent even without explicit Markov structure).

### Stat tests / comparisons

* Compare ERP mean/volatility/tail statistics across GMM clusters (t-tests, ANOVA, tail quantiles).
* Compare clustering quality:

  * Within-cluster vs between-cluster variance (e.g., silhouette scores or similar in reduced space).
* Assess stability and overlap of regimes vs HMM and simple quadrants.
* (Optional) Test whether certain clusters coincide with more frequent or extreme ERP outcomes.

_Note: All predictability and regression-based evaluation using GMM cluster probabilities as features is addressed in Section 4._

---

## 3. Extremeness Models (Macro Anomalies)

You’ll build **two complementary extremeness measures** and compare them.

### 3.1 Isolation Forest

**Model**

* Isolation Forest on:

  * Version A: macro only (4 indices).
  * Version B: macro + sentiment (8 features).

**Steps**

* Standardize features.
* Fit model (choose contamination ~5–10%).
* Get anomaly scores per period; convert to extremeness index (e.g. scaled 0–1).
* Flag top X% as “extreme macro states”.

**Key outputs**

* Time series of extremeness index.
* Histogram / density of extremeness.
* Scatter: extremeness vs ERP.
* Table: ERP statistics in normal vs extreme states.

**Tests**

* Compare ERP distributions in normal vs extreme using:

  * t-test for means.
  * KS test for full distribution.
  * Differences in tail quantiles (e.g. 5th, 1st percentiles).

---

### 3.2 PCA Distance

**Model**

* PCA on standardized macro (and optionally macro+sentiment).
* Choose k PCs explaining 80–90% of variance.
* Define extremeness = distance from center in PC space.

**Steps**

* Run PCA; store loadings and variance explained.
* Compute PC scores for each date.
* Compute distance (Euclidean or Mahalanobis).
* Normalize to create PCA extremeness index.
* Define extreme vs normal by quantiles.

**Key outputs**

* Scree plot of eigenvalues.
* Biplot: PC1 vs PC2, colored by time or extremeness.
* Time series of PCA extremeness.
* Table: overlap between PCA extremes and Isolation Forest extremes.

**Tests**

* Correlation between the two extremeness measures.
* Same ERP comparison tests as above (means, tails).

---

## 4. Macro Impact on ERP

**Goal:** Show how macro variables affect ERP overall, by regime, and by extremeness.

### 4.1 Full-Sample Regressions

**Models**
[
ERP_{t+h} = \alpha + \beta' X_t + \varepsilon_{t+h}
]

* X_t: 4 macro indices (and optionally sentiments).
* Horizons h = 1, 3, 6 months (or weeks).

**Outputs**

* Tables of coefficients, t-stats, R² (per horizon).
* Variable importance ranking based on |t-stat|.

---

### 4.2 Regime-Dependent Regressions (Using HMM or GMM Probabilities)

**Approach: Probability-weighted regressions**
For regime r:
[
ERP_{t+h} = \alpha_r + \beta_r' X_t + \varepsilon_{r,t+h}
]
weighted by ( p_{r,t} )

**Outputs**

* Table: β estimates by regime (rows = variables, cols = regimes).
* Heatmap: strength of effect (β) by regime.
* Chart: comparison of ERP sensitivities across regimes.
* R²/log-likelihood for predictive fit in each regime specification.

**Tests**

* Wald/F-tests for equality of coefficients across regimes.
* t-tests for difference in β_r between regimes.
* Compare predictive power (R²/log-likelihood/AIC/BIC) of regime-based models (HMM, GMM) vs pooled regression.
* Use regime probabilities as features in ERP (or tail-event) forecasting—compare fit for macro, macro+sentiment, and clustering approaches.

---

### 4.3 Extremeness-Dependent Regressions

**Models**

* Add extremeness index as:

  * an interaction: ( ERP_{t+h} = \alpha + \beta'X_t + \delta Ext_t + \gamma (Ext_t \cdot X_t) + \varepsilon )
  * or run separate regressions in normal vs extreme subsamples.

**Outputs**

* Coefficient tables with interaction terms.
* Plot: marginal effect of macro variable on ERP as a function of extremeness.

**Tests**

* F-test: do interaction terms jointly matter?
* Compare R² with and without extremeness.

---

## 5. Forecasting Module

You’ll forecast macro variables (and/or ERP) using **2 econometric + 2 ML models**, with one ML model having a **with/without sentiment** comparison.

### Evaluation Setup (common to all)

* Rolling or expanding window estimation.
* Out-of-sample period (e.g. last 30–40% of sample).
* Forecast horizons: h = 1, 3, 6 (months or weeks).
* Metrics:

  * RMSE, MAE
  * Optional MAPE for macro
* Statistical tests:

  * Diebold–Mariano tests for forecast accuracy differences.

---

### 5.1 Econometric Model 1 – ARDL/OLS

**Model**
For each macro variable M:
[
M_{t+h} = \alpha + \sum_{j=0}^{p} \phi_j M_{t-j} + \sum_{k} \psi_k Z_{k,t} + \varepsilon_{t+h}
]

* Z = other macro indices (and optionally sentiment).

**Outputs**

* Coefficient tables and lag significance.
* Forecast vs realized plots for each macro variable.
* RMSE/MAE comparison table across horizons.

---

### 5.2 Econometric Model 2 – TVP-VAR

**Model**

* VAR(macro) with time-varying coefficients and stochastic volatility.

**Outputs**

* Time-varying impulse response plots (selected shocks).
* Time series of key coefficients (e.g. effect of policy on growth over time).
* RMSE/MAE table vs ARDL.

**Tests**

* Compare forecast errors vs ARDL (Diebold–Mariano).
* Check whether time-variation significantly improves fit (e.g. via marginal likelihood, if implemented).

---

### 5.3 ML Model 1 – XGBoost (with & without sentiment)

**Inputs**

* Features: lags of macro variables, possibly extremeness/regime probabilities.
* Version A: macro-only.
* Version B: macro + sentiment indices.

**Outputs**

* Feature importance plots (gain / SHAP-type if you want).
* Partial dependence plots for key macro/sentiment variables.
* RMSE/MAE table for both versions.
* Bar chart: accuracy improvement when adding sentiment.

**Tests**

* Diebold–Mariano tests XGBoost vs ARDL / TVP-VAR.
* Compare XGBoost macro-only vs macro+sentiment:

  * test if error difference is significant.

---

### 5.4 ML Model 2 – LSTM (sequence model)

**Inputs**

* Sequences of macro (and optionally sentiment) over past k periods.

**Outputs**

* Forecast vs realized plots.
* RMSE/MAE comparison vs ARDL, TVP-VAR, XGBoost.
* Possibly learning curves (to show training convergence).

**Tests**

* Diebold–Mariano tests LSTM vs others.
* Check stability across horizons.

---

## 6. Asset Allocation Strategies

Use the regime, extremeness, and forecasts as **signals** to tilt between stocks and bonds.

### 6.1 Regime-Based Strategy

**Rules**

* For each HMM regime r, define equity weight w_r (e.g. 30%, 50%, 70%).
* Portfolio weight at time t:
  [
  w_t = \sum_r p_{r,t} w_r
  ]

**Outputs**

* Time plot of equity weight vs time.
* Performance stats:

  * Annualized return, vol, Sharpe
  * Max drawdown
* Comparison vs 50/50 static benchmark.

---

### 6.2 Extremeness-Based Strategy

**Rules**

* If extremeness above threshold → reduce equity weight.
* Possibly combine with regime (only cut risk if in risky regime + high extremeness).

**Outputs**

* Same performance metrics as above.
* Histogram of ERP when the strategy is risk-on vs risk-off.

---

### 6.3 Forecast-Based Strategies

For each forecasting model (or a subset: ARDL, XGBoost, LSTM):

**Rules**

* If predicted ERP > θ → overweight equities.
* If predicted ERP < 0 → underweight equities (or overweight bonds).

**Outputs**

* Performance table: Sharpe, vol, max drawdown for each model’s strategy.
* Plot: cumulative returns vs benchmark.
* Hit-rates:

  * % of big drawdowns preceded by underweight signal.
  * % of big rallies preceded by overweight signal.

---

## 7. Synthesis & “What Works” Summary

**Final outputs**

* Summary table:

  * Rows: models/strategies
  * Columns: key metrics (RMSE, Sharpe, drawdown, tail capture).
* Qualitative takeaways:

  * When do regimes matter most?
  * Does extremeness improve tail risk management?
  * How much incremental forecasting power comes from sentiment?
  * Which model/strategy is “best” economically, not just statistically?

**Good final chart**

* Radar or heatmap summarizing:

  * Forecast accuracy
  * Tail identification
  * Sharpe improvement
  * Interpretability
    for each approach.
