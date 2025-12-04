"""
XGBoost model for ERP forecasting.
Supports macro-only and macro+sentiment variants.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class XGBoostERPForecaster:
    """
    XGBoost model for forecasting ERP.
    
    Features:
    - Creates lagged features from macro variables
    - Supports optional sentiment features
    - Annual retraining starting from 2002-03
    """
    
    def __init__(
        self,
        n_lags: int = 12,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 20,
        min_validation_samples: int = 24,
        validation_fraction: float = 0.2
    ):
        """
        Initialize XGBoost forecaster.
        
        Parameters:
        -----------
        n_lags : int
            Number of lags to include for each feature
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        subsample : float
            Subsample ratio
        colsample_bytree : float
            Column subsample ratio
        early_stopping_rounds : int
            Early stopping rounds for validation
        min_validation_samples : int
            Minimum samples needed for validation split
        validation_fraction : float
            Fraction of training data to use for validation
        """
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.min_validation_samples = min_validation_samples
        self.validation_fraction = validation_fraction
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.use_sentiment = False
        
    def create_features(
        self,
        macro_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        target_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Create feature matrix with lagged variables.
        
        Parameters:
        -----------
        macro_df : pd.DataFrame
            Macro variables indexed by date
        sentiment_df : pd.DataFrame, optional
            Sentiment data indexed by date
        target_date : pd.Timestamp, optional
            Only use data up to this date (for out-of-sample)
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix with lagged variables
        """
        # Filter data up to target_date if provided
        if target_date is not None:
            macro_df = macro_df[macro_df.index <= target_date]
            if sentiment_df is not None:
                sentiment_df = sentiment_df[sentiment_df.index <= target_date]
        
        # Select numeric columns only
        macro_features = macro_df.select_dtypes(include=[np.number]).copy()
        
        # Forward fill missing values to handle gaps in data
        macro_features = macro_features.ffill()
        
        # Create lagged features
        feature_list = []
        feature_names = []
        
        for col in macro_features.columns:
            series = macro_features[col]
            for lag in range(1, self.n_lags + 1):
                lagged = series.shift(lag)
                feature_list.append(lagged)
                feature_names.append(f"{col}_lag{lag}")
        
        # Add sentiment features if provided
        if sentiment_df is not None:
            self.use_sentiment = True
            sentiment_numeric = sentiment_df.select_dtypes(include=[np.number]).copy()
            sentiment_numeric = sentiment_numeric.ffill()
            for col in sentiment_numeric.columns:
                # Current and a few lags
                for lag in range(0, min(3, self.n_lags)):
                    if lag == 0:
                        feature_list.append(sentiment_numeric[col])
                        feature_names.append(f"{col}_current")
                    else:
                        lagged = sentiment_numeric[col].shift(lag)
                        feature_list.append(lagged)
                        feature_names.append(f"{col}_lag{lag}")
        
        # Combine all features
        features_df = pd.concat(feature_list, axis=1)
        features_df.columns = feature_names
        features_df = features_df.sort_index()
        
        self.feature_names = feature_names
        return features_df
    
    def fit(
        self,
        erp: pd.Series,
        macro_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        train_end_date: pd.Timestamp = pd.Timestamp("2002-03-31")
    ):
        """
        Train the model on data up to train_end_date.
        
        Parameters:
        -----------
        erp : pd.Series
            ERP values indexed by date
        macro_df : pd.DataFrame
            Macro variables indexed by date
        sentiment_df : pd.DataFrame, optional
            Sentiment data indexed by date
        train_end_date : pd.Timestamp
            Last date to include in training
        """
        # Align data
        common_dates = erp.index.intersection(macro_df.index)
        if sentiment_df is not None:
            common_dates = common_dates.intersection(sentiment_df.index)
        
        # Filter to training period
        train_dates = common_dates[common_dates <= train_end_date]
        
        if len(train_dates) < self.n_lags + 1:
            raise ValueError(f"Not enough training data. Need at least {self.n_lags + 1} observations.")
        
        # Create features
        features_df = self.create_features(
            macro_df.reindex(train_dates),
            sentiment_df.reindex(train_dates) if sentiment_df is not None else None,
            target_date=train_end_date
        )
        
        # Align ERP with features
        erp_aligned = erp.reindex(features_df.index)
        
        # Remove rows with NaN
        valid_mask = ~(features_df.isna().any(axis=1) | erp_aligned.isna())
        X_train = features_df[valid_mask]
        y_train = erp_aligned[valid_mask]
        
        if len(X_train) == 0:
            raise ValueError("No valid training data after removing NaN values")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_array = y_train.values
        
        # Determine validation split
        val_size = 0
        if len(X_train_scaled) >= 2 * self.min_validation_samples:
            proposed_val = int(len(X_train_scaled) * self.validation_fraction)
            val_size = max(self.min_validation_samples, proposed_val)
            val_size = min(val_size, len(X_train_scaled) - self.min_validation_samples)
        
        # Build model parameters
        model_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )
        
        # In XGBoost 3.x, early_stopping_rounds goes in constructor if using eval_set
        if val_size > 0:
            model_params["early_stopping_rounds"] = self.early_stopping_rounds
        
        self.model = xgb.XGBRegressor(**model_params)
        
        # Train with validation split if possible
        if val_size > 0:
            X_tr, X_val = X_train_scaled[:-val_size], X_train_scaled[-val_size:]
            y_tr, y_val = y_train_array[:-val_size], y_train_array[-val_size:]
            self.model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(
                X_train_scaled,
                y_train_array
            )
        
    def predict(
        self,
        macro_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        prediction_date: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Make a single prediction for a given date.
        
        Parameters:
        -----------
        macro_df : pd.DataFrame
            Macro variables indexed by date
        sentiment_df : pd.DataFrame, optional
            Sentiment data indexed by date
        prediction_date : pd.Timestamp, optional
            Date to make prediction for (uses most recent data if None)
        
        Returns:
        --------
        float
            ERP forecast
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Create features up to prediction_date
        features_df = self.create_features(
            macro_df,
            sentiment_df,
            target_date=prediction_date
        )
        
        if len(features_df) == 0:
            return np.nan
        
        # Get most recent row
        if prediction_date is not None:
            available_dates = features_df.index[features_df.index <= prediction_date]
            if len(available_dates) == 0:
                return np.nan
            X = features_df.loc[available_dates[-1:]]
        else:
            X = features_df.iloc[[-1]]
        
        # Check for NaN
        if X.isna().any().any():
            return np.nan
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        return prediction
    
    def forecast_rolling(
        self,
        erp: pd.Series,
        macro_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        start_date: pd.Timestamp = pd.Timestamp("2002-03-31"),
        retrain_frequency: str = "MS"  # Monthly - retrain every month
    ) -> pd.Series:
        """
        Generate rolling forecasts with monthly retraining.
        
        Parameters:
        -----------
        erp : pd.Series
            ERP values indexed by date
        macro_df : pd.DataFrame
            Macro variables indexed by date
        sentiment_df : pd.DataFrame, optional
            Sentiment data indexed by date
        start_date : pd.Timestamp
            First date to forecast
        retrain_frequency : str
            Frequency for retraining (default: "3MS" = 3 months)
        
        Returns:
        --------
        pd.Series
            ERP forecasts indexed by date
        """
        # Get all forecast dates
        forecast_dates = erp.index[erp.index >= start_date].sort_values()
        
        if len(forecast_dates) == 0:
            return pd.Series(dtype=float)
        
        forecasts = []
        last_retrain_date = None
        
        for forecast_date in forecast_dates:
            # Check if we need to retrain (every month)
            if last_retrain_date is None:
                should_retrain = True
            else:
                # Calculate months difference
                months_diff = (forecast_date.year - last_retrain_date.year) * 12 + (forecast_date.month - last_retrain_date.month)
                should_retrain = months_diff >= 1
            
            if should_retrain:
                # Retrain model using data up to forecast_date
                try:
                    self.fit(erp, macro_df, sentiment_df, train_end_date=forecast_date)
                    last_retrain_date = forecast_date
                except Exception as e:
                    print(f"Warning: Failed to retrain at {forecast_date}: {e}")
                    if self.model is None:
                        forecasts.append(np.nan)
                        continue
            
            # Make prediction
            try:
                pred = self.predict(macro_df, sentiment_df, prediction_date=forecast_date)
                forecasts.append(pred)
            except Exception as e:
                print(f"Warning: Failed to predict at {forecast_date}: {e}")
                forecasts.append(np.nan)
        
        return pd.Series(forecasts, index=forecast_dates, name="erp_forecast")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if self.feature_names is None:
            return pd.Series()
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

