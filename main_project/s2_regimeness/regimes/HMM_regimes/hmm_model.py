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
import sys
import io
from contextlib import contextmanager

# Suppress convergence warnings from hmmlearn (these are common and not critical)
warnings.filterwarnings('ignore', category=UserWarning, module='hmmlearn')
warnings.filterwarnings('ignore', message='.*convergence.*')
warnings.filterwarnings('ignore', message='.*not converging.*')


@contextmanager
def suppress_convergence_warnings():
    """Context manager to suppress hmmlearn convergence warnings."""
    # Capture stderr and filter out convergence messages
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        stderr_output = sys.stderr.getvalue()
        sys.stderr = old_stderr
        # Only print if there are non-convergence errors
        if stderr_output and 'convergence' not in stderr_output.lower() and 'not converging' not in stderr_output.lower():
            print(stderr_output, file=old_stderr, end='')


class HMMRegimeModel:
    """
    Gaussian HMM for regime detection using 4 macro variables.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'diag',
        n_iter: int = 200,
        random_state: int = 42,
        variables: Optional[List[str]] = None,
        covar_reg: float = 0.1,
        min_covar: float = 0.01
    ):
        """
        Initialize HMM model.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes (K = 2, 3, 4, 5, or 6)
        covariance_type : str
            Type of covariance matrix ('diag', 'full', 'spherical', 'tied')
        n_iter : int
            Maximum number of iterations for EM algorithm
        random_state : int
            Random seed for reproducibility
        variables : List[str], optional
            List of variable names to use (e.g., ['growth_factor', 'inflation_factor']).
            If None, uses all 4 variables.
        covar_reg : float
            Covariance regularization factor (default 0.1). Higher values shrink covariances more.
        min_covar : float
            Minimum covariance value to prevent regimes from becoming too narrow (default 0.01).
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.covar_reg = covar_reg
        self.min_covar = min_covar
        
        # Set variables to use
        if variables is None:
            self.variables = [
                'growth_factor',
                'inflation_factor',
                'monetary_policy_factor',
                'market_volatility_factor'
            ]
        else:
            self.variables = variables
        
        # Model components
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [v.replace('_factor', '') for v in self.variables]
        
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
            DataFrame with macro factor columns
        fit_scaler : bool
            Whether to fit the scaler (True for training, False for testing)
        
        Returns:
        --------
        np.ndarray: Standardized feature matrix (n_samples, n_features)
        """
        # Check all required columns exist
        missing_cols = [col for col in self.variables if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        features = data[self.variables].values
        
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
                # Suppress convergence warnings for this specific fit
                with warnings.catch_warnings(), suppress_convergence_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='hmmlearn')
                    warnings.filterwarnings('ignore', message='.*convergence.*')
                    warnings.filterwarnings('ignore', message='.*not converging.*')
                    
                    model = hmm.GaussianHMM(
                        n_components=self.n_regimes,
                        covariance_type=self.covariance_type,
                        n_iter=self.n_iter,
                        random_state=self.random_state + init,
                        tol=1e-5  # Slightly relaxed tolerance to reduce warnings
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
        
        # Apply regularization to encourage regime separation
        print(f"  Covariances before regularization:")
        if best_model.covars_.ndim == 3:
            for i in range(self.n_regimes):
                diag_before = np.diag(best_model.covars_[i])
                print(f"    R{i}: {diag_before}")
        
        self._regularize_covariances(best_model, features)
        
        print(f"  Covariances after regularization:")
        if best_model.covars_.ndim == 3:
            for i in range(self.n_regimes):
                diag_after = np.diag(best_model.covars_[i])
                print(f"    R{i}: {diag_after}")
        
        self.model = best_model
        print(f"  Best log-likelihood: {best_log_likelihood:.2f}")
        
        return self
    
    def _regularize_covariances(self, model: hmm.GaussianHMM, features: np.ndarray) -> None:
        """
        Apply regularization to covariance matrices to encourage regime separation.
        
        This shrinks large covariances and ensures minimum covariance values,
        which helps prevent regimes from overlapping too much.
        
        Parameters:
        -----------
        model : hmm.GaussianHMM
            Fitted HMM model to regularize
        features : np.ndarray
            Feature matrix used for fitting
        """
        # Get current covariances
        covars = model.covars_.copy()
        n_features = features.shape[1]
        
        if self.covariance_type == 'diag':
            # For diagonal covariance, hmmlearn stores as (n_regimes, n_features, n_features)
            # but only diagonal elements are used. We need to extract and modify diagonals.
            if covars.ndim == 3:
                # Extract diagonal elements: shape (n_regimes, n_features)
                diag_covars = np.array([np.diag(covars[i]) for i in range(self.n_regimes)])
            elif covars.ndim == 2:
                # Already in (n_regimes, n_features) format
                diag_covars = covars
            else:
                print(f"Warning: Unexpected covars shape {covars.shape}, skipping regularization")
                return
            
            # Calculate average covariance across all regimes
            avg_covar = np.mean(diag_covars, axis=0)  # Average across regimes for each feature
            
            # Regularize: shrink large covariances towards average, but keep minimum
            for i in range(self.n_regimes):
                for j in range(n_features):
                    # Shrink towards average, but ensure minimum
                    diag_covars[i, j] = (
                        (1 - self.covar_reg) * diag_covars[i, j] + 
                        self.covar_reg * avg_covar[j]
                    )
                    diag_covars[i, j] = np.maximum(diag_covars[i, j], self.min_covar)
            
            # For diagonal covariance, hmmlearn stores as (n_regimes, n_features, n_features)
            # but the setter accepts (n_regimes, n_features) and converts it internally
            # However, we need to ensure the model is in a valid state
            # Try setting directly - hmmlearn should handle the conversion
            try:
                model.covars_ = diag_covars
            except ValueError:
                # If that fails, reconstruct the 3D array
                new_covars_3d = np.zeros((self.n_regimes, n_features, n_features))
                for i in range(self.n_regimes):
                    np.fill_diagonal(new_covars_3d[i], diag_covars[i])
                model.covars_ = new_covars_3d
            
        elif self.covariance_type == 'full':
            # For full covariance, regularize diagonal elements
            for i in range(self.n_regimes):
                diag = np.diag(covars[i])
                avg_diag = np.mean(diag)
                diag = (1 - self.covar_reg) * diag + self.covar_reg * avg_diag
                diag = np.maximum(diag, self.min_covar)
                np.fill_diagonal(covars[i], diag)
            model.covars_ = covars
            
        elif self.covariance_type == 'spherical':
            # Single value per regime
            avg_covar = np.mean(covars)
            for i in range(self.n_regimes):
                covars[i] = (
                    (1 - self.covar_reg) * covars[i] + 
                    self.covar_reg * avg_covar
                )
                covars[i] = np.maximum(covars[i], self.min_covar)
            model.covars_ = covars
    
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

