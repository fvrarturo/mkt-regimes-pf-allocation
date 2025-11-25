# Safeguards Against Look-Ahead Bias and Overfitting

This document details all safeguards implemented to prevent look-ahead bias and overfitting in the HMM regime detection system.

## Look-Ahead Bias Prevention

### 1. **Walk-Forward Validation**
- **Implementation**: Uses `TimeSeriesSplit` from scikit-learn
- **How it works**: 
  - Data is split temporally (no random shuffling)
  - Training set always comes before test set chronologically
  - Each fold uses only past data to predict future
- **Location**: `walk_forward_validation()` method

### 2. **Separate Scaler Fitting in Each Fold**
- **Problem**: Fitting scalers on full dataset leaks future information
- **Solution**: 
  - In each validation fold, scalers are fitted ONLY on training data
  - Test data is transformed using scalers fitted on training data
  - Fresh scalers created for each fold to avoid contamination
- **Code**: Lines 536-560 in `regime_detection_hmm.py`
- **Critical**: Test features use `train_macro_scaler.transform()` not `fit_transform()`

### 3. **Temporal Forward-Fill**
- **Implementation**: Forward-fill only uses past/current information
- **How it works**:
  - Each day uses the most recent macro observation available up to that date
  - No future information is used
  - Forward-fill is inherently temporal (only looks backward)
- **Location**: `combine_data()` method, line 250
- **Note**: Forward-fill is done on full dataset, but this is safe because it's temporal

### 4. **Temporal Ordering Maintained**
- All data operations respect chronological order
- No random shuffling or cross-validation that breaks temporal structure
- Date sorting ensures proper time series structure

### 5. **No Future Information in Features**
- Features only use:
  - Current day's sentiment (available in real-time)
  - Most recent available macro data (forward-filled from past)
- No future values are ever used

## Overfitting Prevention

### 1. **BIC/AIC Model Selection**
- **Implementation**: Calculates BIC and AIC for each model
- **How it works**:
  - BIC = -2*log_likelihood + log(n)*k (penalizes complexity more)
  - AIC = -2*log_likelihood + 2*k (penalizes complexity)
  - Lower is better - penalizes models with too many parameters
- **Location**: `calculate_bic_aic()` method
- **Parameter count**: Includes all HMM parameters (means, covariances, transitions)

### 2. **Cross-Validation Evaluation**
- **Implementation**: Walk-forward validation evaluates on held-out test sets
- **How it works**:
  - Model performance measured on unseen test data
  - Average train vs test scores compared
  - Large gap indicates overfitting
- **Location**: `walk_forward_validation()` method
- **Warning**: Alerts if test score much worse than train score

### 3. **Multiple Random Initializations**
- **Implementation**: HMM fitted with multiple random seeds
- **How it works**:
  - Avoids local optima
  - Selects best model based on log-likelihood
  - Prevents overfitting to specific initialization
- **Location**: `fit_hmm()` method, default `n_init=10`

### 4. **HMM Natural Regularization**
- **Implementation**: HMM transition probabilities provide natural regularization
- **How it works**:
  - Transition matrix enforces regime persistence
  - Prevents excessive regime switching
  - Smooths regime assignments over time

### 5. **Parameter Count Awareness**
- **Implementation**: Tracks number of parameters in model
- **How it works**:
  - For 4 regimes, 8 features: 71 parameters
  - With 12,761 observations: ~178 observations per parameter (good ratio)
  - BIC/AIC penalize high parameter counts
- **Location**: `calculate_bic_aic()` method

## Validation Flow

```
1. Load and combine data (forward-fill is temporal-safe)
   ↓
2. Walk-forward validation:
   For each fold:
     a. Split data temporally (train before test)
     b. Fit scalers ONLY on training data
     c. Transform test data with training scalers
     d. Fit HMM on training data
     e. Evaluate on test data
     f. Calculate BIC/AIC
   ↓
3. Final model (for production):
   a. Fit scalers on ALL available data (correct for production)
   b. Fit HMM on all data
   c. Generate regime probabilities
```

## Key Safeguards Summary

| Safeguard | Type | Implementation |
|-----------|------|----------------|
| Temporal splits | Look-ahead bias | TimeSeriesSplit |
| Separate scalers | Look-ahead bias | Fresh scalers per fold |
| Forward-fill | Look-ahead bias | Temporal-only (backward-looking) |
| BIC/AIC | Overfitting | Penalize complexity |
| Cross-validation | Overfitting | Test on held-out data |
| Multiple init | Overfitting | Avoid local optima |
| Parameter tracking | Overfitting | Monitor model complexity |

## Testing the Safeguards

To verify safeguards are working:

1. **Check validation output**: Test scores should be reasonable (not much worse than train)
2. **Check BIC/AIC**: Should prefer simpler models when appropriate
3. **Check regime stability**: Regimes should persist (not switch every day)
4. **Check temporal order**: Dates should always be in chronological order

## Known Limitations

1. **Forward-fill assumption**: Assumes macro data doesn't change until next observation
   - This is standard practice in financial modeling
   - Alternative: Use interpolation, but forward-fill is more conservative

2. **Scaler on full data for final model**: 
   - This is correct for production (use all available information)
   - Validation ensures model generalizes well

3. **Large parameter count**: 
   - 71 parameters for 4 regimes, 8 features
   - Mitigated by BIC/AIC and cross-validation
   - Consider reducing to 3 regimes if overfitting persists

## Recommendations

If overfitting is detected:

1. Reduce number of regimes (e.g., 4 → 3)
2. Use diagonal covariance instead of full (reduces parameters)
3. Increase minimum training size
4. Add more regularization to HMM
5. Reduce number of features

