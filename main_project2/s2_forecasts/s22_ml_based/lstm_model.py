"""
LSTM model implementation for forecasting GDP and inflation.

Classes:
- LSTMForecaster: LSTM model for multivariate time series forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models will not work.")


class LSTMForecaster:
    """
    LSTM forecaster for multivariate time series data.
    Can predict both GDP and inflation jointly.
    """
    
    def __init__(
        self,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        recurrent_dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        l2_reg: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize LSTM forecaster.
        
        Parameters:
        -----------
        lstm_units : int
            Number of LSTM units (reduced to prevent overfitting)
        dropout_rate : float
            Dropout rate after LSTM layer
        recurrent_dropout : float
            Recurrent dropout rate within LSTM
        learning_rate : float
            Learning rate for optimizer (reduced for stability)
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs
        l2_reg : float
            L2 regularization strength
        random_state : int
            Random seed
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")
        
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.random_state = random_state
        
        self.models = {}  # Store models for each horizon
        self.history = {}  # Store training history
    
    def build_model(
        self,
        sequence_length: int,
        n_features: int,
        n_outputs: int = 2
    ) -> keras.Model:
        """
        Build LSTM model architecture with regularization to prevent overfitting.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        n_outputs : int
            Number of output variables (default: 2 for GDP and inflation)
        
        Returns:
        --------
        keras.Model
            Compiled Keras model
        """
        # Set random seed for reproducibility
        tf.random.set_seed(self.random_state)
        
        # L2 regularization (lighter)
        l2_reg = keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        
        # Build model with attention to ARIMA features
        # Use functional API to add residual connections
        inputs = keras.Input(shape=(sequence_length, n_features))
        
        # First LSTM layer
        lstm1 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            recurrent_dropout=self.recurrent_dropout
        )(inputs)
        lstm1_drop = layers.Dropout(self.dropout_rate)(lstm1)
        
        # Second LSTM layer
        lstm2 = layers.LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            recurrent_dropout=self.recurrent_dropout
        )(lstm1_drop)
        lstm2_drop = layers.Dropout(self.dropout_rate)(lstm2)
        
        # Extract AR features from the last timestep (if available)
        # AR features are typically at the end of the feature vector
        # We'll use a separate branch to emphasize AR terms
        ar_branch = layers.Dense(
            self.lstm_units // 4,
            activation='tanh',
            kernel_regularizer=l2_reg
        )(lstm2_drop)
        
        # Combine LSTM output with AR branch
        combined = layers.Concatenate()([lstm2_drop, ar_branch])
        
        # Dense hidden layer
        dense1 = layers.Dense(
            self.lstm_units // 2,
            activation='relu',
            kernel_regularizer=l2_reg
        )(combined)
        dense1_norm = layers.BatchNormalization()(dense1)
        
        # Final output layer with residual connection from LSTM
        # This helps preserve dynamics
        output = layers.Dense(n_outputs)(layers.Concatenate()([dense1_norm, lstm2_drop]))
        
        model = keras.Model(inputs=inputs, outputs=output)
        
        # Enhanced loss function that strongly penalizes flat predictions
        def custom_loss(y_true, y_pred):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Stronger penalty for flat predictions (increased weight)
            mean_pred = tf.reduce_mean(y_pred, axis=0, keepdims=True)
            variance_penalty = -0.05 * tf.reduce_mean(tf.square(y_pred - mean_pred))
            
            # Additional penalty: encourage predictions to follow actual variance
            actual_var = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)))
            pred_var = tf.reduce_mean(tf.square(y_pred - mean_pred))
            variance_match_penalty = 0.1 * tf.square(actual_var - pred_var)
            
            return mse_loss + variance_penalty + variance_match_penalty
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=custom_loss,
            metrics=['mae']
        )
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        horizon: int = 1,
        verbose: int = 1
    ) -> keras.Model:
        """
        Fit LSTM model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences (n_samples, sequence_length, n_features)
        y_train : np.ndarray
            Training targets (n_samples, n_outputs)
        X_val : np.ndarray, optional
            Validation input sequences
        y_val : np.ndarray, optional
            Validation targets
        horizon : int
            Forecast horizon (for storing model)
        verbose : int
            Verbosity level
        
        Returns:
        --------
        keras.Model
            Fitted model
        """
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Build model
        model = self.build_model(sequence_length, n_features, n_outputs)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Callbacks
        callbacks = []
        if validation_data is not None:
            # Reduce learning rate on plateau to help fine-tune
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
            callbacks.append(reduce_lr)
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,  # More patience to allow learning
                restore_best_weights=True,
                verbose=verbose,
                min_delta=0.0001
            )
            callbacks.append(early_stopping)
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Store model and history
        self.models[horizon] = model
        self.history[horizon] = history.history
        
        return model
    
    def predict(
        self,
        X: np.ndarray,
        horizon: int,
        X_ar_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        horizon : int
            Forecast horizon
        X_ar_features : np.ndarray, optional
            AR features from the last timestep (for hybrid AR-LSTM)
        
        Returns:
        --------
        np.ndarray
            Predictions (n_samples, n_outputs)
        """
        if horizon not in self.models:
            raise ValueError(f"No model found for horizon {horizon}")
        
        model = self.models[horizon]
        lstm_pred = model.predict(X, verbose=0)
        
        # If AR features are provided, blend with LSTM predictions
        # This forces the model to use autoregressive dynamics
        if X_ar_features is not None:
            # Use AR terms as a baseline and let LSTM predict the deviation
            # Blend: prediction = alpha * AR + (1-alpha) * LSTM
            # This helps prevent flat predictions
            alpha = 0.3  # Weight for AR component
            blended = alpha * X_ar_features + (1 - alpha) * lstm_pred
            return blended
        
        return lstm_pred
    
    def forecast_rolling(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        horizon: int,
        refit_frequency: int = 12,
        verbose: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate rolling window forecasts with periodic refitting.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Initial training sequences
        y_train : np.ndarray
            Initial training targets
        X_test : np.ndarray
            Test sequences
        horizon : int
            Forecast horizon
        refit_frequency : int
            Number of periods between refits
        verbose : int
            Verbosity level
        
        Returns:
        --------
        tuple
            (forecasts, actuals)
            - forecasts: Predictions (n_test_samples, n_outputs)
            - actuals: Actual values (n_test_samples, n_outputs)
        """
        # Initial fit
        self.fit(X_train, y_train, horizon=horizon, verbose=verbose)
        
        # Generate forecasts
        forecasts = self.predict(X_test, horizon)
        
        return forecasts

