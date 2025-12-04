"""
LSTM model for ERP forecasting.
Supports macro-only and macro+sentiment variants.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will not work.")

from sklearn.preprocessing import StandardScaler


class LSTMerpForecaster:
    """
    LSTM model for forecasting ERP.
    
    Features:
    - Uses sequence of macro variables
    - Supports optional sentiment features
    - Annual retraining starting from 2002-03
    """
    
    def __init__(
        self,
        sequence_length: int = 12,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        """
        Initialize LSTM forecaster.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        lstm_units : int
            Number of LSTM units
        dropout_rate : float
            Dropout rate
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs
        early_stopping_patience : int
            Early stopping patience
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.use_sentiment = False
        
    def prepare_sequences(
        self,
        macro_df: pd.DataFrame,
        erp: pd.Series,
        sentiment_df: Optional[pd.DataFrame] = None,
        target_date: Optional[pd.Timestamp] = None,
        for_prediction: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training or prediction.
        
        Parameters:
        -----------
        macro_df : pd.DataFrame
            Macro variables indexed by date
        erp : pd.Series
            ERP values indexed by date (empty for prediction)
        sentiment_df : pd.DataFrame, optional
            Sentiment data indexed by date
        target_date : pd.Timestamp, optional
            Only use data up to this date
        for_prediction : bool
            If True, prepare for prediction (no ERP needed)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (X sequences, y targets) - y is empty array for prediction
        """
        # Filter data up to target_date if provided
        if target_date is not None:
            macro_df = macro_df[macro_df.index <= target_date]
            if sentiment_df is not None:
                sentiment_df = sentiment_df[sentiment_df.index <= target_date]
            if not erp.empty and not for_prediction:
                erp = erp[erp.index <= target_date]
        
        # Select numeric columns
        macro_features = macro_df.select_dtypes(include=[np.number]).copy()
        macro_features = macro_features.ffill()
        
        # Combine with sentiment if available
        if sentiment_df is not None:
            self.use_sentiment = True
            sentiment_numeric = sentiment_df.select_dtypes(include=[np.number]).copy()
            sentiment_numeric = sentiment_numeric.ffill()
            # Align sentiment with macro
            combined = pd.concat([macro_features, sentiment_numeric], axis=1).sort_index()
            combined = combined.ffill()
        else:
            combined = macro_features
        
        # Align ERP with features
        if not erp.empty and not for_prediction:
            common_dates = combined.index.intersection(erp.index)
        else:
            common_dates = combined.index
        
        if sentiment_df is not None:
            common_dates = common_dates.intersection(sentiment_df.index)
        
        if len(common_dates) < self.sequence_length + 1:
            return np.array([]), np.array([])
        
        # Get aligned data
        features = combined.reindex(common_dates)
        
        if not erp.empty and not for_prediction:
            erp_aligned = erp.reindex(common_dates)
            # Remove rows with NaN
            valid_mask = ~(features.isna().any(axis=1) | erp_aligned.isna())
            features = features[valid_mask]
            erp_aligned = erp_aligned[valid_mask]
        else:
            # For prediction, just remove NaN in features
            valid_mask = ~features.isna().any(axis=1)
            features = features[valid_mask]
            erp_aligned = pd.Series(dtype=float)
        
        if len(features) < self.sequence_length + 1:
            return np.array([]), np.array([])
        
        # Create sequences
        X = []
        y = []
        
        if not erp_aligned.empty:
            # Training: need both X and y
            for i in range(self.sequence_length, len(features)):
                X.append(features.iloc[i-self.sequence_length:i].values)
                y.append(erp_aligned.iloc[i])
        else:
            # Prediction: only X, y is empty
            for i in range(self.sequence_length, len(features)):
                X.append(features.iloc[i-self.sequence_length:i].values)
            y = []
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            (sequence_length, n_features)
        
        Returns:
        --------
        keras.Model
            Compiled LSTM model
        """
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, return_sequences=False, input_shape=input_shape),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
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
        # Prepare sequences
        X, y = self.prepare_sequences(
            macro_df, erp, sentiment_df, target_date=train_end_date, for_prediction=False
        )
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough training data to create sequences")
        
        # Standardize features
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_scaled = X_scaled.astype(np.float32)
        
        # Standardize targets
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        
        # Split into train/validation (80/20)
        split_idx = max(1, int(len(X_scaled) * 0.8))
        split_idx = min(split_idx, len(X_scaled) - 1)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Build model
        input_shape = (self.sequence_length, X_scaled.shape[2])
        self.model = self.build_model(input_shape)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=0
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
        
        # Prepare sequences for prediction
        X, _ = self.prepare_sequences(
            macro_df, pd.Series(dtype=float), sentiment_df, 
            target_date=prediction_date, for_prediction=True
        )
        
        if len(X) == 0:
            return np.nan
        
        # Get most recent sequence
        X_seq = X[-1:].astype(np.float32)
        
        # Standardize
        X_seq_scaled = self.scaler_X.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        
        # Predict
        y_pred_scaled = self.model.predict(X_seq_scaled, verbose=0)[0, 0]
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        return float(y_pred)
    
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

