## 0. Common forecasting setup (for GDP & inflation)

**Targets**

* ( y^{(gdp)}_t ) = growth index (or GDP growth proxy)
* ( y^{(inf)}_t ) = inflation index

You can forecast them **separately** (simpler) or jointly (for VAR/LSTM).

**Horizons**

* ( h \in {1, 3, 6} ) months ahead

For each horizon and each target, you want an out-of-sample forecast series.

**Evaluation design**

* Split sample:

  * Training: first ~60–70% of months
  * Test: last ~30–40%
* Rolling / expanding window:

  * At each test date (t), re-estimate (or update) the model using data up to (t), predict (y_{t+h}).
* Metrics:

  * RMSE, MAE (per horizon, per variable)
* Statistical comparisons:

  * Diebold–Mariano for each pair of models, per target & horizon.

You can later relate forecast performance to **extremeness/regimes** (e.g. models failing more in extreme states). 

---

## 5.1 TVP-VAR for GDP & inflation

### Variables

Use a **4-variable TVP-VAR**:

[
M_t =
\begin{bmatrix}
\text{growth}_t \
\text{inflation}_t \
\text{policy}_t \
\text{volatility}_t
\end{bmatrix}
]

This lets policy & vol influence the dynamics of GDP and inflation.

### Model structure (conceptual)

* VAR(p):
  [
  M_t = A_{0,t} + A_{1,t} M_{t-1} + \dots + A_{p,t}M_{t-p} + \varepsilon_t
  ]
* Coefficients (A_{i,t}) evolve over time (random walk or similar).
* Error covariance (H_t) is time-varying (stochastic volatility).

In practice: use an implementation of **Bayesian TVP-VAR** (or a good approximation like a rolling VAR if full TVP is too heavy).

### Steps to implement

1. **Choose lag order p** (1–3) using a standard VAR on full sample (BIC/AIC).
2. For each forecast origin (t) in the test period:

   * Estimate TVP-VAR using data up to (t) (or update via Kalman filter / MCMC).
   * Generate h-step-ahead forecasts (\hat{M}_{t+h | t}).
   * Store (\hat{y}^{(gdp)}*{t+h|t}) and (\hat{y}^{(inf)}*{t+h|t}).
3. Repeat for all t to build full out-of-sample forecast series.

### Outputs

* **Forecast performance table**

  * RMSE & MAE for GDP & inflation, all horizons.
* **Time-varying coefficients (selected)**

  * e.g. effect of policy on growth over time.
* **Impulse response snapshots**

  * Response of GDP & inflation to a policy or vol shock in different subperiods.
* Optional: compare to a **static VAR** to show value of time-variation.

---

## 5.2 XGBoost for GDP & inflation (macro-only)

You train **separate XGBoost models for each (variable, horizon)** pair:

* GDP 1-month ahead, GDP 3-month ahead, GDP 6-month ahead
* Inflation 1-month, 3-month, 6-month

### Feature construction (macro-only version)

At time t, features (X_t) could include:

* Lags of growth: ( \text{growth}*{t-1}, \dots, \text{growth}*{t-L} )
* Lags of inflation
* Lags of policy
* Lags of volatility
* Optionally: current **HMM regime probabilities** and/or **extremeness index** as extra features

A simple starting point: **L = 6 or 12** months of lags.

Target for horizon h:

[
y_{t+h} \in {\text{growth}*{t+h}, \text{inflation}*{t+h}}
]

So each training row is:

[
(X_t, y_{t+h})
]

### Training procedure

For each (variable, horizon) combo:

1. Construct training matrix using only in-sample period.
2. Use **time-series aware cross-validation** (e.g. expanding window folds).
3. Tune basic hyperparameters:

   * `n_estimators` (e.g. 200–500)
   * `max_depth` (e.g. 2–5)
   * `learning_rate` (e.g. 0.03–0.1)
   * `subsample`, `colsample_bytree` (0.7–1.0)
4. For out-of-sample forecast:

   * For each forecast origin t in test set:

     * Refit (or update) on data up to t (or less frequently if heavy).
     * Predict (y_{t+h}) using features at t.
   * Store forecasts, compute RMSE/MAE.

### Outputs (macro-only)

* RMSE/MAE table (per model, variable, horizon).
* Feature importance plots:

  * show which lags and variables matter most for each target.
* Partial dependence / SHAP plots:

  * effect of key features (e.g. last month’s inflation, vol) on forecast.

---

## 5.2bis XGBoost + sentiment (agentic AI version)

Same pipeline, **just add sentiment features**.

### Additional features

At each time t, add:

* Growth sentiment, inflation sentiment, policy sentiment, vol sentiment (current or short lags)

So (X^{\text{sent}}_t = [X^{\text{macro}}_t, \text{growth_sent}_t, \dots]).

Everything else (train/test split, horizons, tuning) stays identical.

### What you compare

* **Performance**

  * RMSE/MAE with vs without sentiment, for GDP & inflation forecasts
* **Economics**

  * Feature importance: do sentiment variables show up as key predictors?
  * PD plots: does negative inflation sentiment predict higher future inflation, etc.?

### Outputs specific to this comparison

* Side-by-side RMSE bar chart:

  * XGB macro vs XGB macro+sentiment for each target & horizon.
* Table of **Diebold–Mariano p-values** comparing the two (macro vs macro+sent) forecast errors.
* One or two nice SHAP/PD plots highlighting sentiment’s role.

This is where you **prove the value-add of the agentic AI sentiment**.

---

## 5.3 LSTM sequence model (multivariate)

LSTM is your “likely best” model. You can model GDP and inflation **jointly** as a 2-dimensional output.

### Inputs

At each time step t, input vector:

[
Z_t = [\text{growth}_t, \text{inflation}_t, \text{policy}_t, \text{vol}_t, \text{sent_growth}_t, \text{sent_infl}_t, \dots]
]

You build sequences of length **L** (e.g. 12 or 24 months):

[
(Z_{t-L+1}, \dots, Z_t)
]

### Targets

For horizon h, you can either:

* **Direct forecast**: predict ( [\text{growth}*{t+h}, \text{inflation}*{t+h}] ) directly from sequence up to t,
  or
* **Multi-step output**: last layer outputs multiple horizons at once (1,3,6); but start with direct to keep things simple.

### Model architecture (simple & sufficient)

* Input: (sequence_length = L, num_features = d)
* 1 LSTM layer (e.g. 32 or 64 units)
* Dropout (e.g. 0.2–0.3)
* Dense layer to 2 outputs (GDP & inflation)
* Loss = MSE, optimizer = Adam
* Standardization: scale each feature to mean 0, std 1 on training data only.

### Training & evaluation

1. Use training period only, with expanding-window time series split for validation (no shuffling).
2. Early stopping on validation loss to avoid overfitting.
3. For out-of-sample forecasts:

   * Fix trained model (or retrain periodically if you want).
   * For each test origin t:

     * Take last L observations up to t.
     * Predict (y_{t+h}) for GDP and inflation.
   * Compute RMSE/MAE and compare vs TVP-VAR and XGBoost.

### Outputs

* RMSE/MAE comparison table across TVP-VAR, XGB (macro & macro+sent) and LSTM, for:

  * GDP (h=1,3,6)
  * Inflation (h=1,3,6)
* Forecast vs realized plots:

  * overlay actual vs LSTM vs TVP-VAR vs XGBoost for a selected horizon (say h=3) and variable (GDP & inflation separately).
* Learning curve plot:

  * training vs validation loss over epochs (to show convergence / no overfit).

---

## 5.4 Cross-model comparison & narrative

Once you’ve got the three models running:

1. **Performance table** (core slide):

   * Rows: models (TVP-VAR, XGB macro, XGB + sent, LSTM)
   * Columns: RMSE/MAE for GDP & inflation at h=1,3,6

2. **Statistical comparison**:

   * Diebold–Mariano tests pairwise:

     * XGB vs TVP-VAR
     * XGB+sent vs XGB
     * LSTM vs best non-LSTM

3. **Interpretation**:

   * Does time-variation (TVP) help vs static benchmarks?
   * Does adding sentiment significantly improve forecasts (XGB vs XGB+sent)?
   * Does LSTM materially outperform others and in which horizons (short vs medium)?

4. **Connection to regimes/extremeness (optional but nice)**:

   * Check whether model errors are systematically larger in extreme macro states or specific regimes, using your earlier extremeness classification. 