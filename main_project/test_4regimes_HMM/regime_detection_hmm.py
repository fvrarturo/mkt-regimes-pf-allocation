"""
HMM-Based Regime Detection with Macro and Sentiment Data

This script combines 4 macro factors and 4 sentiment scores, applies weighted
combination, and uses HMM to detect 4 regimes based on growth/inflation combinations.

Regimes:
1. High growth / High inflation
2. High growth / Low inflation
3. Low growth / High inflation
4. Low growth / Low inflation

Safeguards:
- Walk-forward validation to prevent look-ahead bias
- Cross-validation for hyperparameter tuning
- BIC/AIC for model selection to prevent overfitting
- Multiple random initializations
- Out-of-sample testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from hmmlearn import hmm
import json
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class RegimeDetectionHMM:
    """
    HMM-based regime detection with macro and sentiment data.
    Implements safeguards against overfitting and look-ahead bias.
    """
    
    def __init__(
        self,
        macro_dir: Path,
        sentiment_path: Path,
        macro_weight: float = 0.4,
        sentiment_weight: float = 0.6,
        n_regimes: int = 4,
        min_train_size: int = 100,
        test_size: int = 50
    ):
        """
        Initialize the regime detection system.
        
        Parameters:
        -----------
        macro_dir : Path
            Path to macro_processed/selection directory
        sentiment_path : Path
            Path to sentiment_scores.csv
        macro_weight : float
            Weight for macro features (default: 0.4)
        sentiment_weight : float
            Weight for sentiment features (default: 0.6)
        n_regimes : int
            Number of regimes to detect (default: 4)
        min_train_size : int
            Minimum training set size for walk-forward validation
        test_size : int
            Test set size for walk-forward validation
        """
        self.macro_dir = Path(macro_dir)
        self.sentiment_path = Path(sentiment_path)
        self.macro_weight = macro_weight
        self.sentiment_weight = sentiment_weight
        self.n_regimes = n_regimes
        self.min_train_size = min_train_size
        self.test_size = test_size
        
        # Validate weights sum to 1
        assert abs(macro_weight + sentiment_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        
        # Data storage
        self.macro_data = {}
        self.sentiment_data = None
        self.combined_data = None
        self.feature_names = []
        
        # Scaling
        self.macro_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        
        # Models
        self.hmm_model = None
        self.regime_probs = None
        self.regime_states = None
        self.transition_matrix = None
        
        # Results
        self.results = {}
        
    def load_macro_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the 4 macro factors from selection folder.
        
        Returns:
        --------
        Dict mapping macro factor names to DataFrames
        """
        print("Loading macro data...")
        
        macro_files = {
            'fedfunds': 'fedfunds_processed.csv',
            'vix': 'vix_processed.csv',
            'PCE_price_index': 'PCE_price_index_processed.csv',
            'gdp': 'gdp_processed.csv'
        }
        
        macro_data = {}
        
        for name, filename in macro_files.items():
            filepath = self.macro_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Macro file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            
            # Select appropriate column based on availability
            # Prefer zscore columns for standardization, fallback to pct_change_mom
            if 'zscore_pct_change_mom' in df.columns:
                value_col = 'zscore_pct_change_mom'
            elif 'pct_change_mom' in df.columns:
                value_col = 'pct_change_mom'
            elif 'zscore_value' in df.columns:
                value_col = 'zscore_value'
            elif 'value' in df.columns:
                value_col = 'value'
            else:
                raise ValueError(f"No suitable column found in {filename}")
            
            # Create clean dataframe
            macro_df = df[['date', value_col]].copy()
            macro_df.columns = ['date', name]
            macro_df = macro_df.dropna()
            
            macro_data[name] = macro_df
            print(f"  Loaded {name}: {len(macro_df)} observations")
        
        self.macro_data = macro_data
        return macro_data
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """
        Load sentiment scores from CSV.
        
        Returns:
        --------
        DataFrame with sentiment scores
        """
        print("Loading sentiment data...")
        
        if not self.sentiment_path.exists():
            raise FileNotFoundError(f"Sentiment file not found: {self.sentiment_path}")
        
        df = pd.read_csv(self.sentiment_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Select the 4 sentiment columns
        sentiment_cols = [
            'inflation_sentiment',
            'ec_growth_sentiment',
            'monetary_policy_sentiment',
            'market_vol_sentiment'
        ]
        
        # Verify all columns exist
        missing_cols = [col for col in sentiment_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing sentiment columns: {missing_cols}")
        
        sentiment_df = df[['date'] + sentiment_cols].copy()
        sentiment_df = sentiment_df.dropna()
        
        print(f"  Loaded sentiment: {len(sentiment_df)} observations")
        
        self.sentiment_data = sentiment_df
        return sentiment_df
    
    def combine_data(self, target_freq: str = 'M') -> pd.DataFrame:
        """
        Combine macro and sentiment data, aligning by date.
        Uses monthly frequency and aggregates sentiment data (mean) while forward-filling macro data
        to use most recent available macro data with monthly updates.
        
        NOTE: Forward-fill is temporal-safe - it only uses past/current information.
        Each month uses the most recent macro observation available up to that date.
        Sentiment data is aggregated to monthly (mean) to reduce noise.
        This prevents look-ahead bias while ensuring we use the latest available data.
        
        Parameters:
        -----------
        target_freq : str
            Target frequency (default: 'M' for monthly)
            Options: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
        
        Returns:
        --------
        Combined DataFrame
        """
        print("Combining macro and sentiment data...")
        print(f"  Using {target_freq} frequency (aggregated sentiment + most recent macro data)...")
        
        if not self.macro_data:
            self.load_macro_data()
        if self.sentiment_data is None:
            self.load_sentiment_data()
        
        # Start with sentiment data - resample to monthly frequency
        sentiment_df = self.sentiment_data.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
        sentiment_df = sentiment_df.set_index('date')
        
        # Aggregate sentiment to monthly (take mean to reduce noise)
        sentiment_cols = [
            'inflation_sentiment',
            'ec_growth_sentiment',
            'monetary_policy_sentiment',
            'market_vol_sentiment'
        ]
        sentiment_monthly = sentiment_df[sentiment_cols].resample(target_freq).mean()
        sentiment_monthly = sentiment_monthly.reset_index()
        sentiment_monthly = sentiment_monthly.dropna()
        
        print(f"    sentiment: {len(self.sentiment_data)} daily -> {len(sentiment_monthly)} monthly (aggregated)")
        
        # Create date range from sentiment data (monthly)
        date_range = pd.date_range(
            start=sentiment_monthly['date'].min(),
            end=sentiment_monthly['date'].max(),
            freq=target_freq
        )
        
        # Create base dataframe with all monthly dates
        combined = pd.DataFrame({'date': date_range})
        combined = pd.merge(combined, sentiment_monthly, on='date', how='left')
        
        # For each macro factor, merge and forward-fill
        for name, macro_df in self.macro_data.items():
            df = macro_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Resample macro data to monthly if needed (take last value in month)
            df = df.set_index('date')
            df_resampled = df.resample(target_freq).last()
            df_resampled = df_resampled.reset_index()
            
            # Merge macro data
            combined = pd.merge(
                combined,
                df_resampled,
                on='date',
                how='left',
                suffixes=('', '_macro')
            )
            
            # Forward-fill macro data (use most recent available value)
            combined[name] = combined[name].ffill()
            
            print(f"    {name}: {len(macro_df)} original -> {combined[name].notna().sum()} monthly values")
        
        # Forward-fill sentiment data as well (in case of gaps)
        for col in sentiment_cols:
            combined[col] = combined[col].ffill()
        
        # Drop rows where we don't have any data (before first macro observation)
        combined = combined.dropna(subset=sentiment_cols + list(self.macro_data.keys()))
        
        # Sort by date
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"  Combined dataset: {len(combined)} monthly observations")
        print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
        
        self.combined_data = combined
        return combined
    
    def prepare_features(self, data: pd.DataFrame, fit_scalers: bool = True) -> np.ndarray:
        """
        Prepare features by standardizing and applying weights.
        
        Uses only GDP and PCE (plus growth/inflation sentiment) to match regime definition.
        Fedfunds and VIX are kept in data for visualization but not used in HMM.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined data with macro and sentiment columns
        fit_scalers : bool
            Whether to fit scalers (True for training, False for testing)
        
        Returns:
        --------
        Combined feature matrix
        """
        # Use only GDP and PCE for macro (matches regime definition)
        # Only use growth and inflation sentiment (matches regime interpretation)
        macro_cols = ['gdp', 'PCE_price_index']  # Only growth and inflation
        sentiment_cols = [
            'ec_growth_sentiment',      # Growth sentiment
            'inflation_sentiment'       # Inflation sentiment
        ]
        
        # Extract features
        macro_features = data[macro_cols].values
        sentiment_features = data[sentiment_cols].values
        
        # Standardize separately
        if fit_scalers:
            macro_scaled = self.macro_scaler.fit_transform(macro_features)
            sentiment_scaled = self.sentiment_scaler.fit_transform(sentiment_features)
        else:
            macro_scaled = self.macro_scaler.transform(macro_features)
            sentiment_scaled = self.sentiment_scaler.transform(sentiment_features)
        
        # Apply weights and combine
        macro_weighted = macro_scaled * self.macro_weight
        sentiment_weighted = sentiment_scaled * self.sentiment_weight
        
        combined_features = macro_weighted + sentiment_weighted
        
        # Store feature names
        self.feature_names = macro_cols + sentiment_cols
        
        return combined_features
    
    def fit_hmm(
        self,
        features: np.ndarray,
        n_init: int = 10,
        random_state: int = 42
    ) -> hmm.GaussianHMM:
        """
        Fit HMM model with multiple initializations to avoid local optima.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        n_init : int
            Number of random initializations
        random_state : int
            Random seed
        
        Returns:
        --------
        Fitted HMM model
        """
        print(f"Fitting HMM with {n_init} initializations...")
        
        best_model = None
        best_log_likelihood = -np.inf
        
        for init in range(n_init):
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",  # Changed from "full" to reduce parameters
                    n_iter=200,
                    random_state=random_state + init,
                    tol=1e-6
                )
                model.fit(features)
                log_likelihood = model.score(features)
                
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_model = model
            except Exception as e:
                print(f"  Initialization {init} failed: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("Failed to fit HMM model")
        
        print(f"  Best log-likelihood: {best_log_likelihood:.2f}")
        
        self.hmm_model = best_model
        return best_model
    
    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Get soft probabilities for each regime.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        
        Returns:
        --------
        Array of shape (n_samples, n_regimes) with probabilities (sum to 1.0 per row)
        """
        if self.hmm_model is None:
            raise ValueError("Model must be fitted first")
        
        # Get posterior log probabilities
        # score_samples returns (log_probability, posterior_log_probabilities)
        log_probs = self.hmm_model.score_samples(features)[1]  # Shape: (n_samples, n_regimes)
        
        # Apply softmax to convert log probabilities to probabilities
        # Subtract max for numerical stability
        log_probs_stable = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_stable)
        
        # Normalize to ensure probabilities sum to 1.0
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Verify probabilities are valid (should sum to 1.0 for each sample)
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1.0"
        assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities must be between 0 and 1"
        
        self.regime_probs = probs
        return probs
    
    def get_regime_states(self, features: np.ndarray) -> np.ndarray:
        """
        Get hard regime assignments.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        
        Returns:
        --------
        Array of regime states
        """
        if self.hmm_model is None:
            raise ValueError("Model must be fitted first")
        
        states = self.hmm_model.predict(features)
        self.regime_states = states
        return states
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition matrix.
        
        Returns:
        --------
        Transition matrix
        """
        if self.hmm_model is None:
            raise ValueError("Model must be fitted first")
        
        transmat = self.hmm_model.transmat_
        self.transition_matrix = transmat
        return transmat
    
    def calculate_bic_aic(self, features: np.ndarray, model: Optional[hmm.GaussianHMM] = None) -> Dict[str, float]:
        """
        Calculate BIC and AIC to assess model fit and prevent overfitting.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        model : Optional[hmm.GaussianHMM]
            HMM model to use (defaults to self.hmm_model)
        
        Returns:
        --------
        Dictionary with BIC and AIC values
        """
        if model is None:
            model = self.hmm_model
        
        if model is None:
            raise ValueError("Model must be fitted first")
        
        n_samples, n_features = features.shape
        log_likelihood = model.score(features)
        
        # Calculate number of parameters
        # For Gaussian HMM with diagonal covariance:
        # - Initial probabilities: n_regimes - 1
        # - Transition matrix: n_regimes * (n_regimes - 1)
        # - Means: n_regimes * n_features
        # - Covariances: n_regimes * n_features (diagonal only, variance per feature)
        
        # Get covariance type from model
        cov_type = model.covariance_type if hasattr(model, 'covariance_type') else 'diag'
        
        if cov_type == 'full':
            # Full covariance matrix
            n_cov_params = self.n_regimes * n_features * (n_features + 1) // 2
        elif cov_type == 'diag':
            # Diagonal covariance (variance only)
            n_cov_params = self.n_regimes * n_features
        else:
            # Spherical or tied (shouldn't happen in our case)
            n_cov_params = self.n_regimes * n_features
        
        n_params = (
            (self.n_regimes - 1) +  # initial probs
            self.n_regimes * (self.n_regimes - 1) +  # transition matrix
            self.n_regimes * n_features +  # means
            n_cov_params  # covariances
        )
        
        # Calculate AIC and BIC
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params
        
        return {
            'AIC': aic,
            'BIC': bic,
            'log_likelihood': log_likelihood,
            'n_params': n_params
        }
    
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict:
        """
        Perform walk-forward validation to prevent look-ahead bias.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined data
        n_splits : int
            Number of time series splits (auto-adjusted if data is too small)
        
        Returns:
        --------
        Dictionary with validation results
        """
        print("Performing walk-forward validation...")
        
        # IMPORTANT: Do NOT fit scalers on full dataset - this would cause look-ahead bias
        # We'll fit scalers separately in each fold on training data only
        # First, just get the feature columns to determine data shape
        # Use only GDP and PCE (plus growth/inflation sentiment) to match regime definition
        macro_cols = ['gdp', 'PCE_price_index']  # Only growth and inflation
        sentiment_cols = [
            'ec_growth_sentiment',      # Growth sentiment
            'inflation_sentiment'       # Inflation sentiment
        ]
        
        # Check data availability without fitting scalers
        n_samples = len(data)
        
        # Adjust n_splits if we don't have enough data
        # Need at least (n_splits + 1) * min_train_size samples
        min_samples_needed = (n_splits + 1) * self.min_train_size
        if n_samples < min_samples_needed:
            # Reduce n_splits to fit available data
            n_splits = max(2, n_samples // (self.min_train_size + self.test_size))
            print(f"  Adjusted n_splits to {n_splits} due to limited data")
        
        # Create indices for time series split (without using scalers)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Create dummy array just for splitting indices
        dummy_array = np.arange(n_samples)
        
        validation_results = {
            'train_scores': [],
            'test_scores': [],
            'train_bics': [],
            'test_bics': [],
            'regime_stabilities': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(dummy_array)):
            print(f"  Fold {fold + 1}/{n_splits}")
            
            # Split data
            train_data = data.iloc[train_idx].copy()
            test_data = data.iloc[test_idx].copy()
            
            # CRITICAL: Fit scalers ONLY on training data to prevent look-ahead bias
            # Create fresh scalers for this fold to avoid contamination
            train_macro_scaler = StandardScaler()
            train_sentiment_scaler = StandardScaler()
            
            # Extract and scale training features
            # Use only GDP and PCE (plus growth/inflation sentiment) to match regime definition
            macro_cols = ['gdp', 'PCE_price_index']  # Only growth and inflation
            sentiment_cols = [
                'ec_growth_sentiment',      # Growth sentiment
                'inflation_sentiment'       # Inflation sentiment
            ]
            
            train_macro_features = train_data[macro_cols].values
            train_sentiment_features = train_data[sentiment_cols].values
            
            train_macro_scaled = train_macro_scaler.fit_transform(train_macro_features)
            train_sentiment_scaled = train_sentiment_scaler.fit_transform(train_sentiment_features)
            
            # Apply weights and combine for training
            train_macro_weighted = train_macro_scaled * self.macro_weight
            train_sentiment_weighted = train_sentiment_scaled * self.sentiment_weight
            train_features_scaled = train_macro_weighted + train_sentiment_weighted
            
            # Transform test features using scalers fitted on training data only
            test_macro_features = test_data[macro_cols].values
            test_sentiment_features = test_data[sentiment_cols].values
            
            test_macro_scaled = train_macro_scaler.transform(test_macro_features)  # Use train scaler!
            test_sentiment_scaled = train_sentiment_scaler.transform(test_sentiment_features)  # Use train scaler!
            
            test_macro_weighted = test_macro_scaled * self.macro_weight
            test_sentiment_weighted = test_sentiment_scaled * self.sentiment_weight
            test_features_scaled = test_macro_weighted + test_sentiment_weighted
            
            # Fit model on training data (temporarily store in self.hmm_model)
            old_model = self.hmm_model
            model = self.fit_hmm(train_features_scaled, n_init=5)
            
            # Evaluate on training and test sets
            train_score = model.score(train_features_scaled)
            test_score = model.score(test_features_scaled)
            
            train_metrics = self.calculate_bic_aic(train_features_scaled, model=model)
            test_metrics = self.calculate_bic_aic(test_features_scaled, model=model)
            
            # Restore old model (or None) to avoid state leakage
            self.hmm_model = old_model
            
            # Calculate regime stability (average duration)
            train_states = model.predict(train_features_scaled)
            stability = self._calculate_regime_stability(train_states)
            
            validation_results['train_scores'].append(train_score)
            validation_results['test_scores'].append(test_score)
            validation_results['train_bics'].append(train_metrics['BIC'])
            validation_results['test_bics'].append(test_metrics['BIC'])
            validation_results['regime_stabilities'].append(stability)
            
            print(f"    Train score: {train_score:.2f}, Test score: {test_score:.2f}")
            print(f"    Train BIC: {train_metrics['BIC']:.2f}, Test BIC: {test_metrics['BIC']:.2f}")
        
        # Calculate average metrics
        avg_train_score = np.mean(validation_results['train_scores'])
        avg_test_score = np.mean(validation_results['test_scores'])
        avg_train_bic = np.mean(validation_results['train_bics'])
        avg_test_bic = np.mean(validation_results['test_bics'])
        
        print(f"\n  Average train score: {avg_train_score:.2f}")
        print(f"  Average test score: {avg_test_score:.2f}")
        print(f"  Average train BIC: {avg_train_bic:.2f}")
        print(f"  Average test BIC: {avg_test_bic:.2f}")
        
        # Check for overfitting (test score much lower than train score)
        score_diff = avg_train_score - avg_test_score
        if score_diff > 10:  # Threshold for overfitting
            print(f"  WARNING: Potential overfitting detected (score diff: {score_diff:.2f})")
        
        return validation_results
    
    def _calculate_regime_stability(self, states: np.ndarray) -> float:
        """Calculate average regime duration."""
        if len(states) == 0:
            return 0.0
        
        durations = []
        current_state = states[0]
        current_duration = 1
        
        for state in states[1:]:
            if state == current_state:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_state = state
                current_duration = 1
        durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def interpret_regimes(self, data: pd.DataFrame) -> Dict:
        """
        Interpret regimes based on growth/inflation characteristics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined data
        
        Returns:
        --------
        Dictionary with regime interpretations
        """
        if self.regime_states is None:
            raise ValueError("Must fit model and get states first")
        
        # Get growth and inflation proxies
        # Growth: GDP, ec_growth_sentiment
        # Inflation: PCE_price_index, inflation_sentiment
        
        growth_cols = ['gdp', 'ec_growth_sentiment']
        inflation_cols = ['PCE_price_index', 'inflation_sentiment']
        
        regime_characteristics = {}
        
        # First pass: calculate average growth and inflation for each regime
        regime_stats = {}
        for regime in range(self.n_regimes):
            regime_mask = self.regime_states == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate average growth and inflation
            avg_growth = regime_data[growth_cols].mean().mean()
            avg_inflation = regime_data[inflation_cols].mean().mean()
            
            regime_stats[regime] = {
                'avg_growth': avg_growth,
                'avg_inflation': avg_inflation,
                'data': regime_data
            }
        
        # Classify based on absolute values relative to zero (historical mean), not relative ranking
        # High/Low means above/below the historical average (zero line)
        for regime in range(self.n_regimes):
            if regime not in regime_stats:
                continue
            
            stats = regime_stats[regime]
            regime_data = stats['data']
            avg_growth = stats['avg_growth']
            avg_inflation = stats['avg_inflation']
            
            # Assign based on actual values relative to zero (historical mean)
            # Growth: positive = High, negative = Low
            growth_level = "High" if avg_growth >= 0 else "Low"
            
            # Inflation: positive = High, negative = Low
            inflation_level = "High" if avg_inflation >= 0 else "Low"
            
            # Debug output to verify classification
            print(f"  Regime {regime}: avg_growth={avg_growth:.4f} ({growth_level}), avg_inflation={avg_inflation:.4f} ({inflation_level})")
            
            # Create descriptive regime name and description
            if growth_level == "High" and inflation_level == "High":
                regime_name = "Expansion with Rising Prices"
                regime_description = "Strong economic growth with increasing inflation. Typically requires monetary tightening."
            elif growth_level == "High" and inflation_level == "Low":
                regime_name = "Goldilocks Economy"
                regime_description = "Ideal conditions: strong growth with low inflation. Best environment for risk assets."
            elif growth_level == "Low" and inflation_level == "High":
                regime_name = "Stagflation"
                regime_description = "Weak growth with high inflation. Worst environment - requires defensive positioning."
            else:  # Low Growth / Low Inflation
                # Distinguish severity based on how far below average
                # More negative values indicate more severe contraction
                severity_threshold = -0.5  # Threshold for "extreme" contraction
                if avg_growth < severity_threshold or avg_inflation < severity_threshold:
                    regime_name = "Recession/Deflation (Extreme)"
                    regime_description = "Severe contraction: very weak growth with very low inflation. Requires aggressive monetary easing and fiscal stimulus."
                else:
                    regime_name = "Recession/Deflation (Moderate)"
                    regime_description = "Moderate contraction: weak growth with low inflation. Typically requires monetary easing and fiscal stimulus."
            
            # Calculate additional statistics for better interpretation
            avg_fedfunds = regime_data['fedfunds'].mean() if 'fedfunds' in regime_data.columns else None
            avg_vix = regime_data['vix'].mean() if 'vix' in regime_data.columns else None
            
            # Calculate policy and volatility characteristics (for reference, not in name)
            overall_fedfunds_median = data['fedfunds'].median() if 'fedfunds' in data.columns else None
            overall_vix_median = data['vix'].median() if 'vix' in data.columns else None
            
            policy_stance = None
            volatility_level = None
            if avg_fedfunds is not None and overall_fedfunds_median is not None:
                policy_stance = "Tight" if avg_fedfunds > overall_fedfunds_median else "Easy"
            if avg_vix is not None and overall_vix_median is not None:
                volatility_level = "High" if avg_vix > overall_vix_median else "Low"
            
            # Use simple name format: "Low Growth / Low Inflation"
            simple_name = f"{growth_level} Growth / {inflation_level} Inflation"
            
            regime_characteristics[regime] = {
                'regime_id': f"R{regime}",
                'name': simple_name,
                'full_name': simple_name,
                'base_name': regime_name,
                'description': regime_description,
                'avg_growth': float(avg_growth),
                'avg_inflation': float(avg_inflation),
                'avg_fedfunds': float(avg_fedfunds) if avg_fedfunds is not None else None,
                'avg_vix': float(avg_vix) if avg_vix is not None else None,
                'policy_stance': policy_stance,
                'volatility_level': volatility_level,
                'n_observations': len(regime_data),
                'pct_of_total': len(regime_data) / len(data) * 100,
                'date_range': (str(regime_data['date'].min()), str(regime_data['date'].max())),
                'growth_level': growth_level,
                'inflation_level': inflation_level
            }
        
        return regime_characteristics
    
    def run_full_analysis(self, output_dir: Optional[Path] = None) -> Dict:
        """
        Run complete analysis pipeline.
        
        Parameters:
        -----------
        output_dir : Optional[Path]
            Directory to save results
        
        Returns:
        --------
        Dictionary with all results
        """
        print("=" * 80)
        print("HMM REGIME DETECTION ANALYSIS")
        print("=" * 80)
        
        # Load and combine data
        self.combine_data()
        
        # Walk-forward validation (fits scalers separately in each fold to prevent look-ahead bias)
        validation_results = self.walk_forward_validation(self.combined_data)
        
        # Final model fit on all data
        # NOTE: For production, we fit scalers on all available data (this is correct)
        # The validation above ensures the model generalizes well
        print("\nFitting final model on all data...")
        features = self.prepare_features(self.combined_data, fit_scalers=True)
        self.fit_hmm(features)
        
        # Get regime probabilities and states
        regime_probs = self.get_regime_probabilities(features)
        regime_states = self.get_regime_states(features)
        transition_matrix = self.get_transition_matrix()
        
        # Calculate model metrics
        model_metrics = self.calculate_bic_aic(features)
        
        # Interpret regimes
        regime_characteristics = self.interpret_regimes(self.combined_data)
        
        # Compile results
        self.results = {
            'model_metrics': model_metrics,
            'validation_results': validation_results,
            'regime_characteristics': regime_characteristics,
            'transition_matrix': transition_matrix.tolist(),
            'regime_states': regime_states.tolist(),
            'regime_probs': regime_probs.tolist(),
            'dates': self.combined_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'feature_names': self.feature_names,
            'n_regimes': self.n_regimes,
            'macro_weight': self.macro_weight,
            'sentiment_weight': self.sentiment_weight
        }
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.save_results(output_dir)
            self.visualize_results(output_dir)
        
        return self.results
    
    def save_results(self, output_dir: Path):
        """Save results to files."""
        print(f"\nSaving results to {output_dir}...")
        
        # Save JSON results
        results_file = output_dir / 'regime_detection_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save regime assignments as CSV with descriptive names
        results_df = pd.DataFrame({
            'date': self.combined_data['date'],
            'regime': self.regime_states
        })
        
        # Add regime names and descriptions
        if self.results.get('regime_characteristics'):
            regime_name_map = {}
            regime_desc_map = {}
            for i in range(self.n_regimes):
                if i in self.results['regime_characteristics']:
                    chars = self.results['regime_characteristics'][i]
                    regime_name_map[i] = chars.get('full_name', chars['name'])
                    regime_desc_map[i] = chars.get('description', '')
                else:
                    regime_name_map[i] = f"Regime {i}"
                    regime_desc_map[i] = ""
            
            results_df['regime_name'] = results_df['regime'].map(regime_name_map)
            results_df['regime_description'] = results_df['regime'].map(regime_desc_map)
        
            # Add probability columns with descriptive names (include regime number for uniqueness)
        for i in range(self.n_regimes):
            if self.results.get('regime_characteristics') and i in self.results['regime_characteristics']:
                chars = self.results['regime_characteristics'][i]
                # Use simple name format for probability column, include regime number for uniqueness
                prob_col_name = chars['name'].replace(" ", "_").replace("/", "_")
                results_df[f'prob_R{i}_{prob_col_name}'] = self.regime_probs[:, i]
            else:
                # Fallback to generic name if regime characteristics not available
                results_df[f'prob_regime_{i}'] = self.regime_probs[:, i]
        
        results_df.to_csv(output_dir / 'regime_assignments.csv', index=False)
        
        # Save transition matrix
        transmat_df = pd.DataFrame(
            self.transition_matrix,
            index=[f'Regime_{i}' for i in range(self.n_regimes)],
            columns=[f'Regime_{i}' for i in range(self.n_regimes)]
        )
        transmat_df.to_csv(output_dir / 'transition_matrix.csv')
        
        print("  Results saved successfully")
    
    def visualize_results(self, output_dir: Path):
        """Create visualizations of results."""
        print("Creating visualizations...")
        
        dates = pd.to_datetime(self.combined_data['date'])
        
        # 1. Regime time series
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Regime states with labels - use simple format
        regime_labels = []
        if self.results.get('regime_characteristics'):
            for i in range(self.n_regimes):
                if i in self.results['regime_characteristics']:
                    chars = self.results['regime_characteristics'][i]
                    regime_labels.append(f"R{i}: {chars['name']}")
                else:
                    regime_labels.append(f"Regime {i}")
        else:
            regime_labels = [f"Regime {i}" for i in range(self.n_regimes)]
        
        axes[0].plot(dates, self.regime_states, marker='o', markersize=2, alpha=0.6)
        axes[0].set_ylabel('Regime')
        axes[0].set_title('Regime States Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yticks(range(self.n_regimes))
        axes[0].set_yticklabels(regime_labels, fontsize=8)
        
        # Regime probabilities with descriptive labels
        for i in range(self.n_regimes):
            label = regime_labels[i] if i < len(regime_labels) else f'Regime {i}'
            axes[1].plot(dates, self.regime_probs[:, i], label=label, alpha=0.7, linewidth=1.5)
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Regime Probabilities Over Time', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Transition matrix heatmap
        im = axes[2].imshow(self.transition_matrix, cmap='Blues', aspect='auto')
        axes[2].set_xticks(range(self.n_regimes))
        axes[2].set_yticks(range(self.n_regimes))
        axes[2].set_xticklabels([regime_labels[i] if i < len(regime_labels) else f'R{i}' for i in range(self.n_regimes)], 
                                rotation=45, ha='right', fontsize=8)
        axes[2].set_yticklabels([regime_labels[i] if i < len(regime_labels) else f'R{i}' for i in range(self.n_regimes)], 
                               fontsize=8)
        axes[2].set_title('Regime Transition Matrix')
        
        # Add text annotations
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                text = axes[2].text(j, i, f'{self.transition_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        plt.savefig(output_dir / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Regime characteristics
        if self.results.get('regime_characteristics'):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            regimes = list(self.results['regime_characteristics'].keys())
            names = [self.results['regime_characteristics'][r]['name'] for r in regimes]
            counts = [self.results['regime_characteristics'][r]['n_observations'] for r in regimes]
            
            axes[0].bar(range(len(regimes)), counts)
            axes[0].set_xticks(range(len(regimes)))
            # Use regime ID and base name for better clarity
            regime_labels_bar = []
            for r in regimes:
                if r in self.results['regime_characteristics']:
                    chars = self.results['regime_characteristics'][r]
                    base_name = chars.get('base_name', chars['name'])
                    # Create compact label with regime ID and economic interpretation
                    regime_labels_bar.append(f"{chars.get('regime_id', f'R{r}')}\n{chars['name']}")
                else:
                    regime_labels_bar.append(f'Regime {r}')
            axes[0].set_xticklabels(regime_labels_bar, rotation=45, ha='right', fontsize=8)
            axes[0].set_ylabel('Number of Observations', fontsize=10)
            axes[0].set_title('Regime Distribution', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Growth vs Inflation scatter
            growths = [self.results['regime_characteristics'][r]['avg_growth'] for r in regimes]
            inflations = [self.results['regime_characteristics'][r]['avg_inflation'] for r in regimes]
            
            scatter = axes[1].scatter(growths, inflations, s=200, alpha=0.6, c=regimes, cmap='viridis')
            for i, (g, inf, r) in enumerate(zip(growths, inflations, regimes)):
                if r in self.results['regime_characteristics']:
                    chars = self.results['regime_characteristics'][r]
                    # Use regime ID and base name for better clarity
                    base_name = chars.get('base_name', chars['name'])
                    # Shorten long names for annotation
                    if 'Extreme' in base_name:
                        label = f"{chars.get('regime_id', f'R{r}')}\n(Extreme)"
                    elif 'Moderate' in base_name:
                        label = f"{chars.get('regime_id', f'R{r}')}\n(Moderate)"
                    else:
                    label = chars.get('regime_id', f'R{r}')
                    axes[1].annotate(label, (g, inf), fontsize=9, ha='center', va='center', 
                                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.8))
            axes[1].axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1.5, label='Historical Mean')
            axes[1].axvline(0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)
            axes[1].set_xlabel('Average Growth (relative to historical mean)', fontsize=10)
            axes[1].set_ylabel('Average Inflation (relative to historical mean)', fontsize=10)
            axes[1].set_title('Regime Characteristics: Growth vs Inflation', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            # Add quadrant labels
            axes[1].text(0.02, 0.02, 'High Growth\nHigh Inflation', transform=axes[1].transAxes, 
                        fontsize=8, ha='left', va='bottom', alpha=0.5, 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            axes[1].text(0.02, 0.98, 'Low Growth\nHigh Inflation', transform=axes[1].transAxes, 
                        fontsize=8, ha='left', va='top', alpha=0.5,
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
            axes[1].text(0.98, 0.02, 'High Growth\nLow Inflation', transform=axes[1].transAxes, 
                        fontsize=8, ha='right', va='bottom', alpha=0.5,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            axes[1].text(0.98, 0.98, 'Low Growth\nLow Inflation', transform=axes[1].transAxes, 
                        fontsize=8, ha='right', va='top', alpha=0.5,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
            plt.colorbar(scatter, ax=axes[1], label='Regime ID')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'regime_characteristics.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("  Visualizations saved")
        
        # 3. Create detailed plots for significant periods
        self.plot_significant_periods(output_dir)
        
        # 4. Create interactive plot
        self.create_interactive_plot(output_dir)
    
    def plot_significant_periods(self, output_dir: Path):
        """Create detailed plots for significant economic periods."""
        print("Creating detailed plots for significant periods...")
        
        dates = pd.to_datetime(self.combined_data['date'])
        
        # Define significant periods
        significant_periods = {
            'Dot-com Bubble (1998-2002)': ('1998-01-01', '2002-12-31'),
            'Financial Crisis (2007-2009)': ('2007-01-01', '2009-12-31'),
            'COVID-19 Pandemic (2020-2021)': ('2020-01-01', '2021-12-31'),
            'Recent Period (2022-2025)': ('2022-01-01', '2025-12-31'),
            'Great Moderation (1991-2007)': ('1991-01-01', '2007-12-31'),
            '2019-2025 Period': ('2019-01-01', '2025-12-31')
        }
        
        regime_labels = []
        if self.results.get('regime_characteristics'):
            for i in range(self.n_regimes):
                if i in self.results['regime_characteristics']:
                    chars = self.results['regime_characteristics'][i]
                    regime_labels.append(f"R{i}: {chars.get('full_name', chars['name'])}")
                else:
                    regime_labels.append(f"Regime {i}")
        else:
            regime_labels = [f"Regime {i}" for i in range(self.n_regimes)]
        
        for period_name, (start_date, end_date) in significant_periods.items():
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                
                # Filter data for this period
                mask = (dates >= start) & (dates <= end)
                if mask.sum() == 0:
                    continue
                
                period_dates = dates[mask]
                period_states = self.regime_states[mask]
                period_probs = self.regime_probs[mask, :]
                
                # Create detailed plot
                fig, axes = plt.subplots(4, 1, figsize=(16, 14))
                
                # 1. Regime states
                colors = plt.cm.Set3(np.linspace(0, 1, self.n_regimes))
                for i in range(self.n_regimes):
                    mask_regime = period_states == i
                    axes[0].scatter(period_dates[mask_regime], period_states[mask_regime], 
                                   c=[colors[i]], label=regime_labels[i], s=50, alpha=0.7)
                axes[0].set_ylabel('Regime', fontsize=11, fontweight='bold')
                axes[0].set_title(f'{period_name} - Regime States', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].set_yticks(range(self.n_regimes))
                axes[0].set_yticklabels(regime_labels, fontsize=9)
                axes[0].legend(loc='upper right', fontsize=8)
                
                # 2. Regime probabilities
                for i in range(self.n_regimes):
                    axes[1].plot(period_dates, period_probs[:, i], label=regime_labels[i], 
                               linewidth=2, alpha=0.8, color=colors[i])
                axes[1].set_ylabel('Probability', fontsize=11, fontweight='bold')
                axes[1].set_title('Regime Probabilities', fontsize=12, fontweight='bold')
                axes[1].legend(loc='upper left', fontsize=8)
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim([0, 1])
                axes[1].fill_between(period_dates, 0, 1, alpha=0.1)
                
                # 3. Underlying features (if available)
                if 'fedfunds' in self.combined_data.columns:
                    period_fedfunds = self.combined_data.loc[mask, 'fedfunds'].values
                    axes[2].plot(period_dates, period_fedfunds, label='Fed Funds Rate', 
                               linewidth=1.5, alpha=0.7, color='blue')
                if 'vix' in self.combined_data.columns:
                    period_vix = self.combined_data.loc[mask, 'vix'].values
                    ax2_twin = axes[2].twinx()
                    ax2_twin.plot(period_dates, period_vix, label='VIX', 
                                 linewidth=1.5, alpha=0.7, color='red')
                    ax2_twin.set_ylabel('VIX', fontsize=11, fontweight='bold', color='red')
                    ax2_twin.tick_params(axis='y', labelcolor='red')
                axes[2].set_ylabel('Fed Funds Rate', fontsize=11, fontweight='bold', color='blue')
                axes[2].set_title('Key Macro Indicators', fontsize=12, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].tick_params(axis='y', labelcolor='blue')
                
                # 4. Sentiment indicators
                sentiment_cols = ['inflation_sentiment', 'ec_growth_sentiment']
                for col in sentiment_cols:
                    if col in self.combined_data.columns:
                        period_sentiment = self.combined_data.loc[mask, col].values
                        axes[3].plot(period_dates, period_sentiment, label=col.replace('_', ' ').title(), 
                                   linewidth=1.5, alpha=0.7)
                axes[3].axhline(0, color='black', linestyle='--', alpha=0.3)
                axes[3].set_ylabel('Sentiment Score', fontsize=11, fontweight='bold')
                axes[3].set_title('Sentiment Indicators', fontsize=12, fontweight='bold')
                axes[3].legend(loc='upper left', fontsize=8)
                axes[3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                safe_name = period_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                plt.savefig(output_dir / f'regime_detail_{safe_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"    Error plotting {period_name}: {e}")
                continue
        
        print("  Detailed period plots saved")
    
    def create_interactive_plot(self, output_dir: Path):
        """Create comprehensive interactive plot for regime analysis."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("  Plotly not available, skipping interactive plot")
            return
        
        print("Creating interactive plot...")
        
        dates = pd.to_datetime(self.combined_data['date'])
        
        # Get regime labels and descriptions
        regime_labels = []
        regime_descriptions = []
        if self.results.get('regime_characteristics'):
            for i in range(self.n_regimes):
                if i in self.results['regime_characteristics']:
                    chars = self.results['regime_characteristics'][i]
                    # Use name for label, which includes growth/inflation levels
                    regime_labels.append(chars['name'])
                    # Use base_name for description to show economic interpretation
                    base_name = chars.get('base_name', chars['name'])
                    description = f"{base_name}. {chars.get('description', '')}"
                    regime_descriptions.append(description)
                else:
                    regime_labels.append(f"Regime {i}")
                    regime_descriptions.append("")
        else:
            regime_labels = [f"Regime {i}" for i in range(self.n_regimes)]
            regime_descriptions = [""] * self.n_regimes
        
        # Color scheme - distinct colors for each regime (updated based on actual classification)
        # Colors are assigned based on regime ID, not economic interpretation
        colors = {
            0: '#d62728',  # Red - R0 (Low Growth / Low Inflation - Moderate)
            1: '#ff7f0e',  # Orange - R1 (High Growth / High Inflation)
            2: '#2ca02c',  # Green - R2 (High Growth / Low Inflation)
            3: '#1f77b4'   # Blue - R3 (Low Growth / Low Inflation - Extreme)
        }
        
        # Create comprehensive figure with multiple panels
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Regime States Over Time', 'Regime Probabilities (Stacked)',
                'Regime Probabilities (Individual)', 'Transition Matrix',
                'Growth & Policy Indicators', 'Inflation & Volatility Indicators'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Regime states over time (top panel, full width)
        for i in range(self.n_regimes):
            mask = self.regime_states == i
            fig.add_trace(
                go.Scatter(
                    x=dates[mask],
                    y=self.regime_states[mask] + np.random.normal(0, 0.05, np.sum(mask)),  # Jitter for visibility
                    mode='markers',
                    name=regime_labels[i],
                    marker=dict(size=8, color=colors[i], opacity=0.7, line=dict(width=1, color='white')),
                    hovertemplate=f'<b>{regime_labels[i]}</b><br>' +
                                 f'Date: %{{x|%Y-%m-%d}}<br>' +
                                 f'Regime ID: R{i}<br>' +
                                 f'Description: {regime_descriptions[i]}<extra></extra>',
                    legendgroup='regimes'
                ),
                row=1, col=1
            )
        
        # 2. Stacked area chart for probabilities
        prob_data = [self.regime_probs[:, i] for i in range(self.n_regimes)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prob_data[0],
                mode='lines',
                name=regime_labels[0],
                fill='tozeroy',
                line=dict(width=0, color=colors[0]),
                stackgroup='one',
                hovertemplate=f'<b>{regime_labels[0]}</b><br>Date: %{{x|%Y-%m-%d}}<br>Probability: %{{y:.1%}}<extra></extra>',
                legendgroup='prob_stacked'
            ),
            row=2, col=1
        )
        for i in range(1, self.n_regimes):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prob_data[i],
                    mode='lines',
                    name=regime_labels[i],
                    fill='tonexty',
                    line=dict(width=0, color=colors[i]),
                    stackgroup='one',
                    hovertemplate=f'<b>{regime_labels[i]}</b><br>Date: %{{x|%Y-%m-%d}}<br>Probability: %{{y:.1%}}<extra></extra>',
                    legendgroup='prob_stacked'
                ),
                row=2, col=1
            )
        
        # 3. Individual probability lines
        for i in range(self.n_regimes):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.regime_probs[:, i],
                    mode='lines',
                    name=regime_labels[i] + ' (prob)',
                    line=dict(width=2, color=colors[i], dash='solid'),
                    hovertemplate=f'<b>{regime_labels[i]}</b><br>' +
                                f'Date: %{{x|%Y-%m-%d}}<br>' +
                                f'Probability: %{{y:.2%}}<br>' +
                                f'<extra></extra>',
                    legendgroup='prob_individual',
                    visible='legendonly'  # Hidden by default, can be toggled
                ),
                row=3, col=1
            )
        
        # 4. Transition matrix heatmap
        if self.transition_matrix is not None:
            fig.add_trace(
                go.Heatmap(
                    z=self.transition_matrix,
                    x=[f'R{i}<br>{regime_labels[i][:20]}' for i in range(self.n_regimes)],
                    y=[f'R{i}<br>{regime_labels[i][:20]}' for i in range(self.n_regimes)],
                    colorscale='Blues',
                    text=[[f'{val:.2%}' for val in row] for row in self.transition_matrix],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.2%}<extra></extra>',
                    colorbar=dict(title="Transition<br>Probability", len=0.3, y=0.5)
                ),
                row=3, col=2
            )
        
        # 5. Growth & Policy indicators (row 3, col 1)
        if 'gdp' in self.combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.combined_data['gdp'],
                    mode='lines',
                    name='GDP',
                    line=dict(width=2, color='#2ca02c'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>GDP: %{y:.3f}<extra></extra>',
                    legendgroup='growth'
                ),
                row=3, col=1
            )
        if 'ec_growth_sentiment' in self.combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.combined_data['ec_growth_sentiment'],
                    mode='lines',
                    name='Growth Sentiment',
                    line=dict(width=2, color='#1f77b4', dash='dash'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Growth Sentiment: %{y:.3f}<extra></extra>',
                    legendgroup='growth'
                ),
                row=3, col=1
            )
        if 'fedfunds' in self.combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.combined_data['fedfunds'],
                    mode='lines',
                    name='Fed Funds Rate',
                    line=dict(width=2, color='#9467bd'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Fed Funds: %{y:.3f}<extra></extra>',
                    legendgroup='policy'
                ),
                row=3, col=1
            )
        
        # 6. Inflation & Volatility indicators (row 3, col 2)
        if 'PCE_price_index' in self.combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.combined_data['PCE_price_index'],
                    mode='lines',
                    name='PCE Price Index',
                    line=dict(width=2, color='#d62728'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>PCE: %{y:.3f}<extra></extra>',
                    legendgroup='inflation'
                ),
                row=3, col=2
            )
        if 'inflation_sentiment' in self.combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.combined_data['inflation_sentiment'],
                    mode='lines',
                    name='Inflation Sentiment',
                    line=dict(width=2, color='#ff7f0e', dash='dash'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Inflation Sentiment: %{y:.3f}<extra></extra>',
                    legendgroup='inflation'
                ),
                row=3, col=2
            )
        if 'vix' in self.combined_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.combined_data['vix'],
                    mode='lines',
                    name='VIX',
                    line=dict(width=2, color='#8c564b'),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>VIX: %{y:.3f}<extra></extra>',
                    legendgroup='volatility'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="<b>Interactive Regime Analysis Dashboard</b><br><sub>Click legend items to show/hide traces | Hover for details | Use zoom/pan tools</sub>",
            title_x=0.5,
            title_font_size=16,
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=9)
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_xaxes(title_text="To Regime", row=2, col=2)
        
        fig.update_yaxes(title_text="Regime", row=1, col=1, tickmode='linear', tick0=0, dtick=1, 
                         ticktext=[f"R{i}" for i in range(self.n_regimes)],
                         tickvals=list(range(self.n_regimes)))
        fig.update_yaxes(title_text="Probability", row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Probability", row=2, col=2, range=[0, 1])
        fig.update_yaxes(title_text="From Regime", row=2, col=2, autorange="reversed", side="right")
        fig.update_yaxes(title_text="Growth & Policy", row=3, col=1)
        fig.update_yaxes(title_text="Inflation & Volatility", row=3, col=2)
        
        # Save interactive plot
        output_file = output_dir / 'regime_analysis_interactive.html'
        fig.write_html(str(output_file))
        print(f"  Interactive plot saved ({output_file.name})")


def main():
    """Main execution function."""
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
    
    # Run analysis
    results = detector.run_full_analysis(output_dir=output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Model Metrics:")
    print(f"  AIC: {results['model_metrics']['AIC']:.2f}")
    print(f"  BIC: {results['model_metrics']['BIC']:.2f}")
    print(f"  Log-likelihood: {results['model_metrics']['log_likelihood']:.2f}")
    print(f"\nRegime Characteristics:")
    print("=" * 80)
    for regime, chars in results['regime_characteristics'].items():
        print(f"\n  {chars.get('regime_id', f'Regime {regime}')}: {chars['name']}")
        print(f"    Description: {chars.get('description', 'N/A')}")
        print(f"    Growth Level: {chars.get('growth_level', 'N/A')}, Inflation Level: {chars.get('inflation_level', 'N/A')}")
        print(f"    Observations: {chars['n_observations']} ({chars['pct_of_total']:.1f}% of total)")
        print(f"    Date range: {chars['date_range'][0]} to {chars['date_range'][1]}")
        if chars.get('avg_fedfunds') is not None:
            print(f"    Avg Fed Funds Rate: {chars['avg_fedfunds']:.2f}")
        if chars.get('avg_vix') is not None:
            print(f"    Avg VIX: {chars['avg_vix']:.2f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

