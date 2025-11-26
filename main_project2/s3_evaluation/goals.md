# **STEP 0 — PREPARATION (before designing strategies)**

### 0.1 Compute the market you will trade

You need:

* **Equity index total return**
* **Bond index total return**
* **The portfolio return:**
  [
  r^{portfolio}_{t} = w_t r^{equity}_t + (1 - w_t)r^{bond}_t
  ]

### 0.2 Decide on signal frequency

* If your forecasts are monthly → monthly rebalancing
* No need to overcomplicate

### 0.3 Decide your benchmark

**Use a 50/50 stock–bond static allocation**, rebalanced monthly.

---

# **6.1 REGIME-BASED STRATEGY (Macro state signal)**

This strategy only uses your **HMM regime probabilities**.

---

## 🔹 Step 1 — Assign risk weights by regime

Example weights:

* Expansions → 70% equity
* Neutral regimes → 50%
* Risk-off regimes → 30%
* Crisis / high-vol regimes → 10–20%

Let (\mathbf{w}_r) be the equity weight for each regime (r).

---

## 🔹 Step 2 — Compute probability-weighted equity weight

[
w_t = \sum_r p_{r,t} w_r
]

This produces a smooth allocation.

---

## 🔹 Step 3 — Compute returns

[
r^{strat}_t = w_t r^{eq}_t + (1 - w_t) r^{bond}_t
]

---

## 🔹 Step 4 — Outputs & Evaluation

### Core performance metrics

* Annualized return
* Annualized volatility
* Sharpe ratio
* Max drawdown
* Calmar ratio
* Turnover (optional)

### Comparison plots

* Cumulative return vs 50/50
* Equity weight over time (key for demonstration)
* Regime transitions overlayed with weight changes

### Stability checks

* Does the strategy reduce losses in recession regimes?
* Does it avoid excessive whipsaws?

---

# **6.2 EXTREMENESS-BASED STRATEGY (Risk management overlay)**

This strategy uses your extremeness indicators to cut risk under extreme macro stress.

---

## 🔹 Step 1 — Define extremeness rule

Simplest form:

* If extremeness score > **90th percentile** → risk-off
* Else → neutral

You can either:

### Option A: Binary cut

[
w_t =
\begin{cases}
20% & \text{if } Ext_t > \tau \
60% & \text{otherwise} \
\end{cases}
]

### Option B: Combine with regimes

Risk-off only if:

* Regime is risk-off **AND** extremeness high
  or
* Regime probability > 0.5 and extremeness high

This produces cleaner behavior.

---

## 🔹 Step 2 — Combine with previous regime strategy (optional but recommended)

You can define:

[
w_t = w^{regime}_t \times (1 - ExtFlag_t) + w^{lowrisk} \times ExtFlag_t
]

Meaning:

* Use regime-based weight normally
* Override by moving to low-risk state when extremeness spikes

**This creates your “Regime + Tail risk management” strategy.**

---

## 🔹 Step 3 — Outputs & Evaluation

### Core outputs

* Performance table (Sharpe, vol, drawdown)
* ERP distribution split:

  * When risk-on vs risk-off
* Hit-rates:

  * % of worst drawdowns preceded by high extremeness
  * % of worst rallies missed (opportunity cost)

### Special charts

* Scatter of extremeness vs equity weight
* Histogram of extremeness during historical crashes
* Cumulative returns zoomed-in around crisis periods

This shows that extremeness provides **crash-avoidance capability**.

---

# **6.3 FORECAST-BASED STRATEGY (Return-seeking)**

This uses your forecasting outputs directly as signals.

---

## 🔹 Step 1 — Construct forecasted ERP

From each model (ARDIMA/ARDL baseline, TVP-VAR, XGB, LSTM):

[
\hat{ERP}_{t+h}
]

Pick the **1-month ahead** as primary signal.

---

## 🔹 Step 2 — Convert forecast to equity weight

Three versions:

### Version A: Threshold rule (simple)

* If (\hat{ERP}>0) → risk-on (70%)
* Else → risk-off (30%)

### Version B: Linear mapping (smooth)

[
w_t = w_{min} + \frac{\hat{ERP}*t - ERP*{5%}}{ERP_{95%} - ERP_{5%}}(w_{max} - w_{min})
]
Clipped between (20%–80%).

### Version C: Regime-conditioned forecasting

* Only go risk-on if:

  * forecast ERP positive
  * AND regime is expansionary
* Only go risk-off if:

  * forecast ERP negative
  * AND regime is risk-off

This **reduces false signals** and is likely the best performing.

---

## 🔹 Step 3 — Outputs & Evaluation

### Core metrics

* Sharpe, vol, drawdown
* Hit rate for sign prediction:
  [
  \mathbb{1}(\hat{ERP}_t \cdot ERP_t > 0)
  ]
* Hit rate for *big events*:

  * % of worst 20 drawdowns predicted (forecast < 0)
  * % of best 20 rallies predicted (forecast > 0)

### Plots

* Cumulative return vs benchmark
* Forecast series vs realized ERP
* Equity weight over time

### Cross-model comparison

A table comparing:

| Model    | Sharpe | Vol | Max DD | Sign Hit Rate | Crash Avoidance |
| -------- | ------ | --- | ------ | ------------- | --------------- |
| ARIMA    |        |     |        |               |                 |
| TVP-VAR  |        |     |        |               |                 |
| XGB      |        |     |        |               |                 |
| XGB+Sent |        |     |        |               |                 |
| LSTM     |        |     |        |               |                 |

This becomes one of the **core slides in the presentation**.

---

# **6.4 Optional but powerful: ENSEMBLE STRATEGIES**

After the three main families, you can optionally create a combined strategy:

* **Forecast × Regime**
* **Forecast × Extremeness**
* **Forecast × Regime × Extremeness**

Example:

[
w_t =
\begin{cases}
w^{low} & \hat{ERP} < 0 \text{ OR } Ext > 0.9 \
w^{high} & \hat{ERP} > 0 \text{ AND } Regime = \text{expansion} \
w^{neutral} & \text{otherwise}
\end{cases}
]

This typically performs best and reflects how PMs actually behave.

---

# 🎯 **Final Deliverables for Section 6**

Below is the checklist of what you should produce:

---

## ✔ 1. **Regime Strategy**

* Weight path over time
* Cumulative return
* Sharpe / Vol / DD
* Comparison vs 50/50

## ✔ 2. **Extremeness Strategy**

* Weight path
* Crash-avoidance hit rates
* ERP histogram by risk-on/risk-off
* Performance table

## ✔ 3. **Forecast-Based Strategy (per model)**

* Signal plots
* Performance table
* Hit rates for large drawdowns/rallies
* Forecast vs realized plot

## ✔ 4. **Master Comparison Table**

* Rows = strategies
* Columns = Sharpe, Vol, Max DD, Hit-rate, Crisis-avoidance

## ✔ 5. “Summary Slide”

* 3–4 bullet points explaining which strategy works, when, and why.