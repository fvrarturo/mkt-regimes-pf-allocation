# Cross-Model Comparison Scripts

This directory contains scripts for comparing forecast performance across all models:
- **TVP-VAR** (Time-Varying Parameter VAR)
- **XGBoost (Macro)** (Gradient boosting with macro features only)
- **XGBoost (Macro+Sent)** (Gradient boosting with macro and sentiment features)
- **LSTM** (Long Short-Term Memory neural network)

## Scripts

### 1. `main.py`
**Main comparison script** - Runs the full comparison pipeline:
- Loads metrics from all models
- Creates performance comparison tables
- Generates visualizations
- Computes relative improvements
- Creates summary statistics

**Usage:**
```bash
python main.py
```

**Outputs:**
- `performance_comparison_table.csv` - Full performance table
- `performance_rmse_table.csv` - Pivoted RMSE table
- `performance_mae_table.csv` - Pivoted MAE table
- `relative_improvement_table.csv` - Percentage improvements vs baseline
- `best_models_summary.csv` - Best model for each metric/horizon/variable
- `performance_comparison_rmse.png` - RMSE comparison bar charts
- `performance_comparison_mae.png` - MAE comparison bar charts
- `performance_heatmap.png` - Heatmap of all metrics

### 2. `plot_model_rankings.py`
**Model ranking visualizations** - Shows which model ranks best:
- Ranking heatmaps (1=best, 4=worst)
- Improvement bar charts showing relative performance

**Usage:**
```bash
python plot_model_rankings.py
```

**Outputs:**
- `model_rankings.png` - Heatmap showing rankings by metric and horizon
- `improvement_bars.png` - Bar charts showing % improvement vs TVP-VAR

### 3. `create_summary_report.py`
**Markdown report generator** - Creates a comprehensive summary report:
- Executive summary
- Key findings
- Best models by horizon
- Relative improvements
- Performance tables
- Conclusions

**Usage:**
```bash
python create_summary_report.py
```

**Outputs:**
- `model_comparison_report.md` - Comprehensive markdown report

### 4. `plot_forecast_comparisons.py`
**Forecast overlay plots** - Creates plots showing all model forecasts together (placeholder - requires forecast series to be saved/loaded)

## Modules

### `load_results.py`
- `load_all_metrics()` - Loads CSV metrics from all models
- `create_performance_table()` - Creates unified performance table
- `pivot_performance_table()` - Creates pivoted tables for easier analysis

### `plotting.py`
- `plot_performance_comparison()` - Bar charts comparing RMSE/MAE
- `plot_heatmap_performance()` - Heatmap visualization
- `plot_forecast_comparison_all_models()` - Overlay plots of forecasts
- `plot_dm_test_results()` - Diebold-Mariano test visualizations

### `stats.py`
- `run_dm_tests()` - Run Diebold-Mariano tests between model pairs
- `compute_relative_improvement()` - Compute percentage improvements

## Results Directory

All outputs are saved to `results/` directory:
- CSV tables with performance metrics
- PNG plots for visualizations
- Markdown report with summary

## Key Metrics Compared

For each model, variable (Growth, Inflation), and horizon (1, 3, 6 months):
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Relative improvements** vs baseline (TVP-VAR)
- **Rankings** (1=best, 4=worst)

## Usage Example

```bash
# Run full comparison
python main.py

# Generate rankings
python plot_model_rankings.py

# Create report
python create_summary_report.py
```

## Notes

- The scripts automatically detect and load metrics from:
  - `s21_macro/results/` (TVP-VAR)
  - `s22_ml_based/results/xgboost/` (XGBoost)
  - `s22_ml_based/results/lstm/` (LSTM)
- Forecast series comparisons require forecast data to be saved (currently placeholder)
- Diebold-Mariano tests can be run if forecast errors are available

