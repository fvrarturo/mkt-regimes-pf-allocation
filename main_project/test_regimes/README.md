# HMM Variable Selection for Regime Detection

This folder contains automated variable selection tools for Hidden Markov Model (HMM) regime detection.

## Overview

The `hmm_variable_selection.py` script automatically selects the most important macro and sentiment variables for regime detection using HMMs. It penalizes for model complexity (via BIC/AIC) and retains only essential variables to avoid overfitting.

## Features

1. **Automated Variable Selection**: Tests multiple variable combinations and selects the best subset based on penalized model selection criteria
2. **Model Quality Metrics**: Calculates BIC, AIC, silhouette score, regime stability, and regime separation
3. **Comprehensive Outputs**: Generates results tables, visualizations, and model summaries
4. **Economic Event Overlay**: Optionally overlays known economic event dates on regime transition plots

## Dependencies

Install the required packages:

```bash
pip install hmmlearn scikit-learn seaborn numpy pandas matplotlib
```

Or if using the project's requirements:

```bash
pip install -r ../../requirements.txt
pip install hmmlearn scikit-learn seaborn
```

## Usage

### Basic Usage

```python
from pathlib import Path
from hmm_variable_selection import HMMVariableSelector

# Set up paths
data_dir = Path('../../data/macro_processed')
sentiment_path = Path('../../initial_test/llm_text/sentiment_scores.csv')
output_dir = Path('results')

# Initialize selector
selector = HMMVariableSelector(
    data_dir=data_dir,
    sentiment_path=sentiment_path if sentiment_path.exists() else None,
    n_regimes=3,        # Number of regimes to detect
    min_vars=2,         # Minimum variables in subset
    max_vars=8,         # Maximum variables in subset
    max_combinations=100  # Limit on combinations to test
)

# Run analysis
selector.run_full_analysis(output_dir=output_dir)
```

### Command Line

```bash
python hmm_variable_selection.py
```

### Custom Economic Events

```python
economic_events = {
    '2008-09-15': 'Lehman Brothers Bankruptcy',
    '2020-03-23': 'COVID-19 Lockdown',
    # Add more events...
}

selector.run_full_analysis(
    output_dir=output_dir,
    economic_events=economic_events
)
```

## Output Files

The script generates the following outputs in the `results/` directory:

1. **`hmm_selection_results.csv`**: Table with all tested variable subsets and their metrics (BIC, AIC, silhouette, stability, etc.)
2. **`best_model_summary.json`**: Summary of the best-performing model
3. **`regime_transitions.png`**: Time series plots showing each variable with regime coloring, plus regime state over time
4. **`model_comparison.png`**: Scatter plots comparing models across different metrics

## How It Works

1. **Data Loading**: Loads all macro variables from `data/macro_processed/` organized by category (ec_growth, inflation, mkt_vol, mon_policy, other). For each category, selects the most frequent series (e.g., `pct_change_mom` for inflation).

2. **Sentiment Integration**: Optionally loads sentiment data from `sentiment_scores.csv` if available.

3. **Variable Subset Generation**: Generates combinations of variables within size constraints (min_vars to max_vars).

4. **HMM Fitting**: For each variable subset:
   - Standardizes the data
   - Fits a Gaussian HMM with specified number of regimes
   - Calculates evaluation metrics

5. **Model Selection**: Selects the variable subset with the **lowest BIC** (Bayesian Information Criterion), which automatically penalizes for unnecessary complexity.

6. **Output Generation**: Creates comprehensive results tables and visualizations.

## Model Selection Criteria

The script uses multiple metrics to evaluate models:

- **BIC (Bayesian Information Criterion)**: Primary selection criterion - penalizes complexity more than AIC
- **AIC (Akaike Information Criterion)**: Alternative information criterion
- **Silhouette Score**: Measures how well-separated regimes are
- **Regime Stability**: Average duration of regimes (higher is better)
- **Regime Separation**: Average distance between regime means (higher is better)

## Interpretation

- **Lower BIC/AIC**: Better model fit adjusted for complexity
- **Higher Silhouette**: Better regime separation
- **Higher Stability**: More persistent regimes (less switching)
- **Higher Separation**: More distinct regime characteristics

The best model balances these metrics, with BIC being the primary criterion to avoid overfitting.

## Example Results

After running, you'll see output like:

```
Best Model Selected (Lowest BIC)
================================================================================
Variables: gdp, PCE_price_index, fedfunds, vix
BIC: 1234.56
AIC: 1200.34
Silhouette Score: 0.456
Regime Stability: 12.34
Regime Separation: 2.567
```

## Notes

- The script automatically handles missing data by dropping rows with any NaN values
- All variables are standardized before HMM fitting
- The number of regimes can be adjusted based on economic theory or data characteristics
- Limiting `max_combinations` helps control runtime for large variable sets

