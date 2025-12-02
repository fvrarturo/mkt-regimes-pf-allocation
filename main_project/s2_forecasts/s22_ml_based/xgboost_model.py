"""
XGBoost model implementation for forecasting GDP and inflation.

Classes:
- XGBoostForecaster: XGBoost model for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class XGBoostForecaster:
    """
    XGBoost forecaster for time series data with time-series aware cross-validation.
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        random_state: int = 42
    ):
        """
        Initialize XGBoost forecaster.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.random_state = random_state
        
        self.models = {}  # Store models for each (variable, horizon) pair
        self.feature_names = None
    
    def _train_val_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_fraction: float = 0.15,
        min_val: int = 24
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.Series, Optional[pd.Series]]:
        """
        Split training data into (train, validation) preserving order.
        """
        n_obs = len(X)
        val_size = max(int(n_obs * val_fraction), min_val)
        if n_obs <= val_size + 10:
            return X, None, y, None
        
        X_fit = X.iloc[:-val_size]
        X_val = X.iloc[-val_size:]
        y_fit = y.iloc[:-val_size]
        y_val = y.iloc[-val_size:]
        return X_fit, X_val, y_fit, y_val
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5,
        param_grid: Optional[Dict] = None
    ) -> Dict:
        """
        Tune hyperparameters using time-series cross-validation.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        n_splits : int
            Number of CV splits
        param_grid : dict, optional
            Parameter grid to search. If None, uses default grid.
        
        Returns:
        --------
        dict
            Best hyperparameters
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [2, 3, 5],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0.0, 0.1, 0.3]
            }
        
        # Use TimeSeriesSplit for time-series aware CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        best_score = np.inf
        best_params = None
        
        # Simple random grid search for efficiency
        print("  Tuning hyperparameters...")
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"  Testing {total_combinations} combinations (this may take a while)...")
        
        from itertools import product
        rng = np.random.default_rng(self.random_state)
        all_combinations = list(product(*param_grid.values()))
        sample_size = min(30, len(all_combinations))
        if len(all_combinations) > sample_size:
            indices = rng.choice(len(all_combinations), size=sample_size, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations
        
        for idx, params_tuple in enumerate(combinations):
            params = dict(zip(param_grid.keys(), params_tuple))
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                mse = np.mean((y_val - y_pred) ** 2)
                cv_scores.append(mse)
            
            avg_score = np.mean(cv_scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
            
            if (idx + 1) % 5 == 0:
                print(f"    Tested {idx + 1}/{len(combinations)} combinations, best MSE: {best_score:.4f}")
        
        print(f"  Best parameters: {best_params}")
        print(f"  Best CV MSE: {best_score:.4f}")
        
        return best_params
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        variable: str,
        horizon: int,
        tune: bool = True
    ) -> xgb.XGBRegressor:
        """
        Fit XGBoost model for a specific variable and horizon.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        variable : str
            Variable name (e.g., 'growth', 'inflation')
        horizon : int
            Forecast horizon in months
        tune : bool
            Whether to tune hyperparameters
        
        Returns:
        --------
        xgb.XGBRegressor
            Fitted model
        """
        # Store feature names
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        
        # Tune hyperparameters if requested
        if tune:
            best_params = self.tune_hyperparameters(X_train, y_train)
            self.n_estimators = best_params['n_estimators']
            self.max_depth = best_params['max_depth']
            self.learning_rate = best_params['learning_rate']
            self.subsample = best_params['subsample']
            self.colsample_bytree = best_params['colsample_bytree']
            self.min_child_weight = best_params.get('min_child_weight', self.min_child_weight)
            self.gamma = best_params.get('gamma', self.gamma)
        
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        X_fit, X_val, y_fit, y_val = self._train_val_split(X_train, y_train)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        model.fit(
            X_fit,
            y_fit,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=25 if eval_set else None
        )
        
        # Store model
        key = (variable, horizon)
        self.models[key] = model
        
        return model
    
    def predict(
        self,
        X: pd.DataFrame,
        variable: str,
        horizon: int
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        variable : str
            Variable name
        horizon : int
            Forecast horizon
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        key = (variable, horizon)
        if key not in self.models:
            raise ValueError(f"No model found for {variable}, horizon {horizon}")
        
        model = self.models[key]
        return model.predict(X)
    
    def get_feature_importance(
        self,
        variable: str,
        horizon: int,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Parameters:
        -----------
        variable : str
            Variable name
        horizon : int
            Forecast horizon
        importance_type : str
            Type of importance ('gain', 'weight', 'cover')
        
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        key = (variable, horizon)
        if key not in self.models:
            raise ValueError(f"No model found for {variable}, horizon {horizon}")
        
        model = self.models[key]
        
        # Get feature names from the model (XGBoost stores them)
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            feature_names = list(model.feature_names_in_)
        elif self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'f{i}' for i in range(len(model.feature_importances_))]
        
        # Get importance scores from booster
        importance_dict = model.get_booster().get_score(importance_type=importance_type)
        
        # Map feature names to importance scores
        # XGBoost uses 'f0', 'f1', etc. as keys in get_score()
        importance_values = []
        for i, feat_name in enumerate(feature_names):
            # Try both the feature name and the f{i} format
            score = importance_dict.get(feat_name, 0)
            if score == 0:
                score = importance_dict.get(f'f{i}', 0)
            importance_values.append(score)
        
        # Alternative: use feature_importances_ directly if get_score doesn't work
        if all(v == 0 for v in importance_values):
            importance_values = model.feature_importances_.tolist()
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def forecast_rolling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        variable: str,
        horizon: int,
        refit_frequency: int = 12
    ) -> pd.Series:
        """
        Generate rolling window forecasts with periodic refitting.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Initial training features
        y_train : pd.Series
            Initial training targets
        X_test : pd.DataFrame
            Test features
        variable : str
            Variable name
        horizon : int
            Forecast horizon
        refit_frequency : int
            Number of periods between refits (default: 12 months)
        
        Returns:
        --------
        pd.Series
            Forecasts indexed by test dates
        """
        forecasts = pd.Series(index=X_test.index, dtype=float)
        
        # Initial fit
        self.fit(X_train, y_train, variable, horizon, tune=False)
        
        # Rolling forecast
        current_X_train = X_train.copy()
        current_y_train = y_train.copy()
        
        for idx, test_date in enumerate(X_test.index):
            if idx % refit_frequency == 0 and idx > 0:
                # Refit model
                print(f"    Refitting model at {test_date.date()} (iteration {idx})")
                self.fit(current_X_train, current_y_train, variable, horizon, tune=False)
            
            # Predict
            X_test_row = X_test.loc[[test_date]]
            forecast = self.predict(X_test_row, variable, horizon)[0]
            forecasts.loc[test_date] = forecast
            
            # Update training data (for next iteration)
            # Note: In practice, we'd need actual values, but for forecasting we use predictions
            # This is a simplified version - in real scenario, you'd update with actuals as they arrive
        
        return forecasts
