# Regime-Conditional Regression Analysis

This folder contains analysis of which macro variables predict Equity Risk Premium (ERP) conditional on different economic regimes.

## 📁 Folder Organization

Results are organized into two main folders:

### 1. `results_full_sample/` - Full-Sample Analysis (With Look-Ahead Bias)
- **Purpose**: Exploratory/descriptive analysis
- **Regime Detection**: Uses entire dataset
- **Use for**: Understanding relationships, hypothesis generation
- **Not suitable for**: Actual predictions, trading strategies

### 2. `results_expanding_window/` - Expanding Window Analysis (No Look-Ahead Bias)
- **Purpose**: Predictive analysis, out-of-sample validation
- **Regime Detection**: Uses only past data (expanding windows)
- **Use for**: Actual predictions, trading strategies, real-world applications
- **Suitable for**: Out-of-sample forecasting

See `FOLDER_STRUCTURE.md` for detailed structure.

## 📊 Key Findings

### Variables with Significant Predictive Power

1. **Unemployment**: Significant in multiple regimes (β = 0.0044** to 0.0091**)
2. **Fed Reserve Discount Rate**: Strong effects in Overheating/Stagflation (β = -0.0116** to -0.0158***)
3. **National Financial Conditions Index**: Significant in recessionary regimes (β = -0.0043** to -0.0049**)
4. **Industrial Production**: Significant in Stagflation (β = -0.0081**)

### Model Comparison

- **2x2 Model**: Better regime differentiation (25 significant coefficient differences)
- **HMM Optimal**: More consistent patterns across regimes (0 significant differences)

## 📝 Scripts

### Full-Sample Analysis
- `regressions_full_sample/regime_conditional_regressions.py` - Main regression script
- `regressions_full_sample/regime_conditional_regressions_README.md` - Documentation

### Expanding Window Analysis
- `regressions_expanding_window/regime_conditional_regressions.py` - Main regression script
- `regressions_expanding_window/regime_detection_expanding_window.py` - Expanding window regime detection
- `regressions_expanding_window/run_analysis_no_lookahead.py` - Complete analysis pipeline
- `regressions_expanding_window/regime_conditional_regressions_README.md` - Documentation

## 📚 Documentation

- `FOLDER_STRUCTURE.md` - Detailed folder structure
- `regime_conditional_regressions_README.md` - Methodology documentation
- `results_full_sample/README.md` - Full-sample results overview
- `results_expanding_window/README.md` - Expanding window results overview

## 🔍 Quick Start

### To view full-sample results:
```bash
cd regressions_full_sample/results/{model}/
# See SUMMARY.md for key findings
```

### To view expanding window results:
```bash
cd regressions_expanding_window/results/{model}/
# See SUMMARY.md and RESULTS_SUMMARY.md for key findings
```

### To compare results:
- Full-sample: `regressions_full_sample/results/{model}/SUMMARY.md`
- Expanding window: `regressions_expanding_window/results/{model}/SUMMARY.md`

## ⚠️ Important Notes

1. **Full-sample results** have look-ahead bias - use for exploration only
2. **Expanding window results** are truly predictive - use for forecasting
3. **Robust relationships** are those significant in both analyses
4. **Unemployment** is the most robust predictor (holds in both analyses)

