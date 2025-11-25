# HMM Regime Detection with Macro and Sentiment Data

This module implements Hidden Markov Model (HMM) based regime detection that combines 4 macro factors and 4 sentiment scores to identify 4 economic regimes based on growth/inflation combinations.

## Regimes

The system identifies 4 regimes:
1. **High Growth / High Inflation** - Economic expansion with rising prices
2. **High Growth / Low Inflation** - Goldilocks economy (ideal conditions)
3. **Low Growth / High Inflation** - Stagflation
4. **Low Growth / Low Inflation** - Recession/deflation

## Features

- **Weighted Combination**: Macro features (40%) + Sentiment features (60%)
- **Separate Standardization**: Macro and sentiment features are standardized independently before combination
- **HMM with Soft Probabilities**: Provides probability distributions over regimes, not just hard assignments
- **Regime Transition Analysis**: Tracks probabilities of switching between regimes
- **Safeguards Against Overfitting**:
  - Walk-forward validation to prevent look-ahead bias
  - BIC/AIC model selection criteria
  - Multiple random initializations
  - Time series cross-validation
- **Comprehensive Outputs**: JSON results, CSV files, and visualizations

## Data Requirements

### Macro Factors (from `data/macro_processed/selection/`):
1. `fedfunds_processed.csv` - Federal Funds Rate
2. `vix_processed.csv` - VIX volatility index
3. `PCE_price_index_processed.csv` - Personal Consumption Expenditures Price Index
4. `gdp_processed.csv` - Gross Domestic Product

### Sentiment Scores (from `data/news_data/`):
1. `inflation_sentiment` - Sentiment about inflation
2. `ec_growth_sentiment` - Sentiment about economic growth
3. `monetary_policy_sentiment` - Sentiment about monetary policy
4. `market_vol_sentiment` - Sentiment about market volatility

## Installation

```bash
pip install pandas numpy scikit-learn hmmlearn matplotlib seaborn
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pathlib import Path
from regime_detection_hmm import RegimeDetectionHMM

# Set up paths
project_root = Path(__file__).parent.parent
macro_dir = project_root / 'data' / 'macro_processed' / 'selection'
sentiment_path = project_root / 'data' / 'news_data' / 'sentiment_scores.csv'
output_dir = Path(__file__).parent / 'results'

# Initialize detector
detector = RegimeDetectionHMM(
    macro_dir=macro_dir,
    sentiment_path=sentiment_path,
    macro_weight=0.4,
    sentiment_weight=0.6,
    n_regimes=4
)

# Run full analysis
results = detector.run_full_analysis(output_dir=output_dir)
```

### Command Line

```bash
python regime_detection_hmm.py
```

## Output Files

The analysis generates several output files in the `results/` directory:

1. **`regime_detection_results.json`** - Complete results including:
   - Model metrics (AIC, BIC, log-likelihood)
   - Validation results
   - Regime characteristics
   - Transition matrix
   - Regime probabilities and states

2. **`regime_assignments.csv`** - Time series of regime assignments and probabilities

3. **`transition_matrix.csv`** - Regime transition probabilities

4. **`regime_analysis.png`** - Visualizations of:
   - Regime states over time
   - Regime probabilities over time
   - Transition matrix heatmap

5. **`regime_characteristics.png`** - Visualizations of:
   - Regime distribution
   - Growth vs Inflation scatter plot

## Methodology

### Data Combination

1. **Load Data**: Macro factors and sentiment scores are loaded
2. **Frequency Alignment**: 
   - Sentiment data is daily (news published every day)
   - Macro data has various frequencies (monthly, quarterly)
   - **Solution**: Aggregate sentiment to monthly (mean) and resample macro to monthly
   - Forward-fill macro data to use most recent available values
   - This reduces noise while ensuring we use the most recent news + most recent available macro data
   - Each month uses aggregated sentiment + most recent macro observation available (forward-filled)
3. **Standardization**: Each feature type is standardized separately:
   - Macro features: StandardScaler on macro data
   - Sentiment features: StandardScaler on sentiment data
4. **Weighted Combination**: 
   ```
   X_combined = (X_macro_scaled × 0.4) + (X_sentiment_scaled × 0.6)
   ```

### HMM Fitting

1. **Multiple Initializations**: Model is fitted with multiple random seeds to avoid local optima
2. **Full Covariance**: Uses full covariance matrices to capture feature relationships
3. **Soft Probabilities**: Uses posterior probabilities for regime assignments

### Validation

1. **Walk-Forward Validation**: Time series cross-validation prevents look-ahead bias
2. **BIC/AIC**: Model selection criteria prevent overfitting
3. **Out-of-Sample Testing**: Separate train/test evaluation

## Safeguards Against Common Problems

### Look-Ahead Bias
- **Walk-forward validation**: Only uses past data to predict future
- **Time series splits**: Proper temporal ordering maintained
- **Separate scalers**: Training scalers fitted only on training data

### Overfitting
- **BIC/AIC criteria**: Penalize model complexity
- **Cross-validation**: Evaluate on held-out data
- **Regularization**: HMM naturally regularizes through transition probabilities

### Data Leakage
- **Separate standardization**: Macro and sentiment standardized independently
- **Temporal ordering**: All operations respect time ordering
- **No future information**: Features only use past and current data

## Regime Interpretation

Regimes are interpreted based on:
- **Growth proxies**: GDP and economic growth sentiment
- **Inflation proxies**: PCE price index and inflation sentiment

Each regime is characterized by:
- Average growth level (high/low) - determined relative to overall median
- Average inflation level (high/low) - determined relative to overall median
- Number of observations
- Date range
- Percentage of total time period

**Note**: Growth and inflation levels are determined relative to the overall median of the dataset, not absolute zero. This ensures proper classification even when data is standardized.

## Example Output

```
HMM REGIME DETECTION ANALYSIS
================================================================================
Loading macro data...
  Loaded fedfunds: 432 observations
  Loaded vix: 8829 observations
  Loaded PCE_price_index: 802 observations
  Loaded gdp: 316 observations
Loading sentiment data...
  Loaded sentiment: 1869 observations
Combining macro and sentiment data...
  Combined dataset: 316 observations
  Date range: 1990-01-01 to 2024-12-30

Performing walk-forward validation...
  Fold 1/5
    Train score: -245.32, Test score: -48.21
    Train BIC: 5234.12, Test BIC: 1023.45
  ...

Fitting final model on all data...
  Best log-likelihood: -1234.56

ANALYSIS SUMMARY
================================================================================
Model Metrics:
  AIC: 2567.89
  BIC: 2789.12
  Log-likelihood: -1234.56

Regime Characteristics:
  Regime 0: High Growth / Low Inflation
    Observations: 89 (28.2%)
    Date range: 1995-03-01 to 2019-12-31
  Regime 1: Low Growth / High Inflation
    Observations: 45 (14.2%)
    Date range: 2008-01-01 to 2022-06-30
  ...
```

## Customization

You can customize the analysis by adjusting:

- **Weights**: Change `macro_weight` and `sentiment_weight` (must sum to 1.0)
- **Number of regimes**: Adjust `n_regimes` (default: 4)
- **Validation splits**: Change `n_splits` in `walk_forward_validation()`
- **Initializations**: Adjust `n_init` in `fit_hmm()`

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- hmmlearn >= 0.2.7
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## References

- Hidden Markov Models for regime detection
- Time series cross-validation for financial data
- BIC/AIC for model selection
- Walk-forward analysis for backtesting

