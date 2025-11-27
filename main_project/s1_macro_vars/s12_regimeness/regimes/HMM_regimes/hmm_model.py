"""
HMM Model for Regime Detection using 4 Macro Variables

This module implements a Gaussian Hidden Markov Model (HMM) for detecting
macroeconomic regimes using 4 standardized macro indices:
- Growth
- Inflation
- Monetary Policy
- Market Volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


class HMMRegimeModel:
    """
    Gaussian HMM for regime detection using 4 macro variables.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'diag',
        n_iter: int = 200,
        random_state: int = 42
    ):
        """
        Initialize HMM model.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes (K = 2, 3, or 4)
        covariance_type : str
            Type of covariance matrix ('diag', 'full', 'spherical', 'tied')
        n_iter : int
            Maximum number of iterations for EM algorithm
        random_state : int
            Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Model components
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['growth', 'inflation', 'policy', 'volatility']
        
        # Results storage
        self.regime_states = None
        self.regime_probs = None
        self.transition_matrix = None
        self.model_metrics = {}
        
    def prepare_features(
        self,
        data: pd.DataFrame,
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Prepare and standardize macro features.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with columns: growth_factor, inflation_factor,
            monetary_policy_factor, market_volatility_factor
        fit_scaler : bool
            Whether to fit the scaler (True for training, False for testing)
        
        Returns:
        --------
        np.ndarray: Standardized feature matrix (n_samples, 4)
        """
        # Extract the 4 macro variables
        feature_cols = [
            'growth_factor',
            'inflation_factor',
            'monetary_policy_factor',
            'market_volatility_factor'
        ]
        
        # Check all columns exist
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        features = data[feature_cols].values
        
        # Standardize features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def fit(
        self,
        features: np.ndarray,
        n_init: int = 10
    ) -> 'HMMRegimeModel':
        """
        Fit HMM model with multiple initializations.
        
        Parameters:
        -----------
        features : np.ndarray
            Standardized feature matrix (n_samples, 4)
        n_init : int
            Number of random initializations to avoid local optima
        
        Returns:
        --------
        self: Returns self for method chaining
        """
        print(f"Fitting HMM with {self.n_regimes} regimes...")
        print(f"  Using {n_init} random initializations...")
        
        best_model = None
        best_log_likelihood = -np.inf
        
        for init in range(n_init):
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state + init,
                    tol=1e-6
                )
                model.fit(features)
                log_likelihood = model.score(features)
                
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_model = model
                    
            except Exception as e:
                print(f"    Initialization {init + 1} failed: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("Failed to fit HMM model after all initializations")
        
        self.model = best_model
        print(f"  Best log-likelihood: {best_log_likelihood:.2f}")
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Get most likely regime assignments.
        
        Parameters:
        -----------
        features : np.ndarray
            Standardized feature matrix
        
        Returns:
        --------
        np.ndarray: Regime assignments (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        states = self.model.predict(features)
        self.regime_states = states
        return states
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get regime probabilities for each observation.
        
        Parameters:
        -----------
        features : np.ndarray
            Standardized feature matrix
        
        Returns:
        --------
        np.ndarray: Regime probabilities (n_samples, n_regimes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        # Get posterior log probabilities
        log_probs = self.model.score_samples(features)[1]  # Shape: (n_samples, n_regimes)
        
        # Convert log probabilities to probabilities (softmax)
        log_probs_stable = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_stable)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Verify probabilities sum to 1.0
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1.0"
        
        self.regime_probs = probs
        return probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition matrix.
        
        Returns:
        --------
        np.ndarray: Transition matrix (n_regimes, n_regimes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        transmat = self.model.transmat_
        self.transition_matrix = transmat
        return transmat
    
    def calculate_model_metrics(self, features: np.ndarray) -> Dict[str, float]:
        """
        Calculate AIC, BIC, and log-likelihood for model selection.
        
        Parameters:
        -----------
        features : np.ndarray
            Standardized feature matrix
        
        Returns:
        --------
        Dict with AIC, BIC, log_likelihood, and n_params
        """
        if self.model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        n_samples, n_features = features.shape
        log_likelihood = self.model.score(features)
        
        # Calculate number of parameters
        # For Gaussian HMM with diagonal covariance:
        # - Initial probabilities: n_regimes - 1
        # - Transition matrix: n_regimes * (n_regimes - 1)
        # - Means: n_regimes * n_features
        # - Covariances: n_regimes * n_features (diagonal only)
        
        if self.covariance_type == 'full':
            n_cov_params = self.n_regimes * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diag':
            n_cov_params = self.n_regimes * n_features
        elif self.covariance_type == 'spherical':
            n_cov_params = self.n_regimes
        else:  # tied
            n_cov_params = n_features * (n_features + 1) // 2
        
        n_params = (
            (self.n_regimes - 1) +  # initial probs
            self.n_regimes * (self.n_regimes - 1) +  # transition matrix
            self.n_regimes * n_features +  # means
            n_cov_params  # covariances
        )
        
        # Calculate AIC and BIC
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params
        
        metrics = {
            'AIC': aic,
            'BIC': bic,
            'log_likelihood': log_likelihood,
            'n_params': n_params,
            'n_samples': n_samples
        }
        
        self.model_metrics = metrics
        return metrics
    
    def interpret_regimes(
        self,
        data: pd.DataFrame,
        regime_states: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Interpret regimes based on average macro values.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Original data with macro factors
        regime_states : np.ndarray
            Regime assignments
        
        Returns:
        --------
        Dict mapping regime_id to characteristics
        """
        # Get available macro columns (only those present in data)
        all_macro_cols = [
            'growth_factor',
            'inflation_factor',
            'monetary_policy_factor',
            'market_volatility_factor'
        ]
        macro_cols = [col for col in all_macro_cols if col in data.columns]
        
        regime_characteristics = {}
        
        for regime_id in range(self.n_regimes):
            regime_mask = regime_states == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate average macro values (only for available columns)
            avg_macro = {}
            for col in macro_cols:
                avg_macro[col.replace('_factor', '')] = regime_data[col].mean()
            
            # Determine levels relative to overall median (only if column exists)
            growth_level = None
            inflation_level = None
            policy_level = None
            vol_level = None
            
            if 'growth_factor' in data.columns:
                growth_level = "High" if avg_macro.get('growth', 0) >= data['growth_factor'].median() else "Low"
            if 'inflation_factor' in data.columns:
                inflation_level = "High" if avg_macro.get('inflation', 0) >= data['inflation_factor'].median() else "Low"
            if 'monetary_policy_factor' in data.columns:
                policy_level = "High" if avg_macro.get('monetary_policy', 0) >= data['monetary_policy_factor'].median() else "Low"
            if 'market_volatility_factor' in data.columns:
                vol_level = "High" if avg_macro.get('market_volatility', 0) >= data['market_volatility_factor'].median() else "Low"
            
            # Create descriptive name based on available variables
            name_parts = []
            if growth_level is not None and inflation_level is not None:
                regime_name = f"{growth_level} Growth / {inflation_level} Inflation"
            elif growth_level is not None:
                regime_name = f"{growth_level} Growth"
            elif inflation_level is not None:
                regime_name = f"{inflation_level} Inflation"
            else:
                regime_name = f"Regime {regime_id}"
            
            regime_characteristics[regime_id] = {
                'regime_id': regime_id,
                'name': regime_name,
                'avg_growth': float(avg_macro.get('growth', np.nan)),
                'avg_inflation': float(avg_macro.get('inflation', np.nan)),
                'avg_policy': float(avg_macro.get('monetary_policy', np.nan)),
                'avg_volatility': float(avg_macro.get('market_volatility', np.nan)),
                'growth_level': growth_level,
                'inflation_level': inflation_level,
                'policy_level': policy_level,
                'volatility_level': vol_level,
                'n_observations': len(regime_data),
                'pct_of_total': len(regime_data) / len(data) * 100,
                'date_range': (
                    str(regime_data['date'].min()),
                    str(regime_data['date'].max())
                )
            }
        
        return regime_characteristics

