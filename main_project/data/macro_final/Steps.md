# ✅ **1. Inflation — use a *composite inflation factor* (best practice)**

### Why?

Inflation is multidimensional:

* CPI
* PCE
* PPI
  All capture **different channels** (consumer, producer, pipeline, traded-goods inflation).

Academics and central banks do not rely on one series. They use:

* a *common component* (inflation factor) extracted from several price indices
* or a trimmed composite

### Best practice (economics + ML):

Use **the first principal component (PC1)** from standardized inflation series.

This is the standard approach in:

* Stock & Watson (common inflation trend)
* Inflation nowcasting at Fed/ECB
* Macro-finance factors (e.g., Adrian et al.)

### Steps:

1. Standardize: z-score CPI_mom, PCE_mom, PPI_mom
2. Run PCA → keep PC1.
3. Use PC1 as your **inflation feature**.

This avoids giving CPI more weight than PCE just because it’s more volatile.

➡️ **Final choice: Inflation composite factor (PC1).**

---

# ✅ **2. Economic Growth — use Industrial Production month-over-month % change**

### Why?

Industrial Production is a key real-time indicator:

* Available monthly with minimal lag (unlike GDP which is quarterly)
* Highly correlated with overall economic growth and business cycles
* Directly interpretable: positive values = growth, negative values = contraction
* Widely used in macro-finance research as a proxy for real activity
* Simpler and more interpretable than composite factors for regime analysis

### Best practice:

Use **Industrial Production month-over-month percentage change**:

* Load Industrial Production processed data (`ind_prod_processed.csv`)
* Extract `pct_change_mom` (month-over-month % change)
* Use directly as `growth_factor`

This provides a direct, interpretable measure of real economic activity growth that is ideal for regime classification and forecasting applications.

➡️ **Final choice: Industrial Production MoM % Change**

---

# ✅ **3. Monetary Policy — use Federal Funds Rate**

Unlike inflation and growth, policy is largely *one-dimensional*:

* The Fed controls the **short-term risk-free rate** and the **policy stance**.
* Federal Funds Rate is the primary policy tool and most direct measure of policy stance

### Best practice for regime classification:

Use **Federal Funds Rate (effective rate)**

because it:

* Provides direct, interpretable measure of policy stance
* Essential for understanding policy-driven market dynamics
* Higher rates = tighter policy, lower rates = easier policy
* Most direct measure for regime classification

➡️ **Final choice: Federal Funds Rate**

---

# ✅ **4. Market Volatility / Financial Conditions — use VIX**

This category measures **market volatility and stress**:

* VIX (CBOE Volatility Index) - most widely recognized measure
* Directly measures expected volatility of S&P 500 options
* Highly correlated with market stress and risk-off periods
* More readily available and interpretable than composite indices

### Best practice:

Use **VIX (CBOE Volatility Index)**

because it:

* Provides direct measure of expected market volatility
* Essential for understanding volatility regimes and risk-off periods
* Higher values = higher expected volatility and market stress
* Lower values = calmer market conditions
* Widely used and interpretable

➡️ **Final choice: VIX**
---

# ⭐ Final Summary — Best Single Feature per Category

These align perfectly with modern empirical macro-finance.

| Category                                 | Best Feature                                                                 | Type      | Why                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------- | --------- | -------------------------------------------- |
| **Inflation**                            | Inflation Factor (PC1 of CPI, PCE, PPI)                                      | Composite | Captures common price trend; avoids bias     |
| **Growth**                               | Industrial Production MoM % Change                                           | Single    | Direct, interpretable real activity measure  |
| **Policy**                               | Federal Funds Rate                                                           | Single    | Direct measure of policy stance              |
| **Market Volatility / Financial Stress** | VIX                                                                          | Single    | Direct measure of expected market volatility |