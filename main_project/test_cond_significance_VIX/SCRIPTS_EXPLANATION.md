# Explanation of the 3 Python Scripts

## Overview

There are 3 Python files that serve different purposes in the analysis pipeline:

1. **`vix_macro_relevance.py`** - Core analyzer class (reusable library)
2. **`regime_conditional_predictors.py`** - Detailed lagged predictor analysis (uses #1)
3. **`summarize_predictive_significance.py`** - Post-processing summary (uses results from #1)

---

## File 1: `vix_macro_relevance.py`

**Purpose**: Core analyzer class - the main reusable component

**What it contains**:
- `VIXMacroRelevanceAnalyzer` class with all analysis methods
- Data loading functions
- Correlation analysis
- Regression analysis
- Random Forest feature importance
- Visualization functions
- Main function to run full analysis

**What it does**:
- Loads VIX data, regime assignments, and macro variables
- Creates lagged variables (t-1 predicts t)
- Runs three types of analysis (correlation, regression, RF)
- Creates composite relevance scores
- Saves results to CSV and creates plots

**Output**:
- `results/tables/vix_macro_relevance_summary.csv`
- `results/tables/vix_correlations_by_regime.csv`
- `results/tables/vix_regressions_by_regime.csv`
- `results/tables/vix_feature_importance_by_regime.csv`
- `results/plots/*.png` (3 plots)

**Can run standalone**: ✅ Yes - `python vix_macro_relevance.py`

---

## File 2: `regime_conditional_predictors.py`

**Purpose**: Detailed analysis script focusing on lagged predictors

**What it contains**:
- Uses `VIXMacroRelevanceAnalyzer` from file #1
- Additional detailed analysis and reporting
- Prints formatted tables to console
- More detailed breakdown by category

**What it does**:
- Imports and uses the analyzer class
- Runs the same analysis but with more detailed output
- Creates additional summary tables
- Prints formatted results to console (easier to read)

**Output**:
- `results/detailed_by_regime/lagged_correlations_regime_X.csv`
- `results/detailed_by_regime/regressions_regime_X.csv`
- `results/detailed_by_regime/summary_by_category.csv`
- Console output with formatted tables

**Can run standalone**: ✅ Yes - `python regime_conditional_predictors.py`

**Why separate?**
- Provides more detailed, readable output
- Focuses specifically on lagged predictors
- Creates regime-specific detailed files

---

## File 3: `summarize_predictive_significance.py`

**Purpose**: Post-processing script to summarize and interpret results

**What it contains**:
- Reads results CSV files (from file #1)
- Merges correlation and regression results
- Identifies significant predictors
- Creates summary tables

**What it does**:
- **Does NOT run analysis** - only processes existing results
- Reads CSV files created by `vix_macro_relevance.py`
- Merges correlation and regression p-values
- Identifies which variables are significant in both tests
- Creates summary tables showing:
  - Significant predictors by regime
  - Summary by category
  - Top predictors by correlation and R²

**Output**:
- `results/tables/predictive_significance_summary.csv`
- `results/tables/category_summary_by_regime.csv`
- Console output with formatted summary

**Can run standalone**: ✅ Yes - `python summarize_predictive_significance.py`

**Why separate?**
- Post-processing step (doesn't need to re-run analysis)
- Can be run multiple times to regenerate summaries
- Focuses on interpretation and significance testing

---

## Workflow

```
1. Run vix_macro_relevance.py
   ↓
   Creates: correlations, regressions, feature importance, plots
   
2. Run regime_conditional_predictors.py (optional)
   ↓
   Creates: detailed regime-specific files, formatted console output
   
3. Run summarize_predictive_significance.py
   ↓
   Creates: summary of significant predictors
```

---

## Could They Be Combined?

**Option 1: Keep Separate (Current)**
- ✅ **Pros**: 
  - Clear separation of concerns
  - Can run analysis once, summarize multiple times
  - Each script has a focused purpose
  - Easier to maintain and debug
- ❌ **Cons**: 
  - More files to manage
  - Need to run multiple scripts

**Option 2: Combine into One Script**
- ✅ **Pros**: 
  - Single script to run
  - Simpler for users
- ❌ **Cons**: 
  - Very long file (1000+ lines)
  - Mixes analysis and post-processing
  - Can't re-run summaries without re-running analysis
  - Harder to maintain

**Recommendation**: Keep separate - the current structure is cleaner and more maintainable.

---

## Quick Reference

| File | Purpose | When to Run | Output Location |
|------|---------|-------------|-----------------|
| `vix_macro_relevance.py` | Core analysis | First (or only) | `results/tables/`, `results/plots/` |
| `regime_conditional_predictors.py` | Detailed analysis | Optional (for detailed output) | `results/detailed_by_regime/` |
| `summarize_predictive_significance.py` | Summary | After #1 (can run multiple times) | `results/tables/` |

---

## Minimum Required

**To get results, you only need to run:**
```bash
python vix_macro_relevance.py
```

This gives you all the core results. The other two scripts provide additional detail and summaries.


