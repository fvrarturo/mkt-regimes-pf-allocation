"""
Forecasting Models Module
Trains Random Forest and XGBoost models for macro factor forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from pathlib import Path


def prepare_train_test_split(
    combined_df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    test_size: float = 0.2,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train-test split for forecasting.
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined DataFrame with features and targets
    feature_cols : List[str]
        List of feature column names
    target_cols : List[str]
        List of target column names
    test_size : float
        Proportion of data for test set
    shuffle : bool
        Whether to shuffle data (False for time series)
    
    Returns:
    --------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    # Prepare X (features) and y (targets)
    X = combined_df[feature_cols].fillna(combined_df[feature_cols].mean())
    y = combined_df[target_cols].fillna(combined_df[target_cols].mean())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    max_features: str = 'sqrt',
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Train Random Forest model for multi-target regression.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame
        Training targets
    X_test : pd.DataFrame
        Test features
    y_test : pd.DataFrame
        Test targets
    feature_cols : List[str]
        Feature column names
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    min_samples_split : int
        Minimum samples to split
    min_samples_leaf : int
        Minimum samples per leaf
    max_features : str
        Number of features to consider
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs
    
    Returns:
    --------
    Tuple of (model, results_dict)
    """
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        oob_score=True
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'oob_score': rf_model.oob_score_,
        'overfitting_gap': train_r2 - test_r2
    }
    
    return rf_model, results


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 1.0,
    reg_lambda: float = 1.0,
    min_child_weight: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train XGBoost model for multi-target regression.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame
        Training targets
    X_test : pd.DataFrame
        Test features
    y_test : pd.DataFrame
        Test targets
    feature_cols : List[str]
        Feature column names
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
    reg_alpha : float
        L1 regularization
    reg_lambda : float
        L2 regularization
    min_child_weight : int
        Minimum child weight
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs
    
    Returns:
    --------
    Tuple of (model, results_dict)
    """
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        n_jobs=n_jobs,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfitting_gap': train_r2 - test_r2
    }
    
    return xgb_model, results


def train_individual_models(
    combined_df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    test_size: float = 0.2
) -> Dict[str, Tuple[RandomForestRegressor, Dict]]:
    """
    Train separate Random Forest models for each target variable.
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined DataFrame
    feature_cols : List[str]
        Feature column names
    target_cols : List[str]
        Target column names
    test_size : float
        Test set proportion
    
    Returns:
    --------
    Dict mapping target variable names to (model, results) tuples
    """
    models = {}
    
    X = combined_df[feature_cols].fillna(combined_df[feature_cols].mean())
    
    for target_var in target_cols:
        y_single = combined_df[[target_var]].fillna(combined_df[target_var].mean())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_single, test_size=test_size, shuffle=False
        )
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True
        )
        rf_model.fit(X_train, y_train.values.ravel())
        
        # Evaluate
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'overfitting_gap': train_r2 - test_r2
        }
        
        models[target_var] = (rf_model, results)
    
    return models


def plot_feature_importance(
    model,
    feature_cols: List[str],
    title: str = "Feature Importance",
    top_n: int = 15,
    output_path: Optional[Path] = None
):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    -----------
    model
        Trained model with feature_importances_ attribute
    feature_cols : List[str]
        Feature column names
    title : str
        Plot title
    top_n : int
        Number of top features to show
    output_path : Path, optional
        Path to save the plot
    """
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:top_n], feature_importance['importance'][:top_n])
    plt.xlabel('Importance')
    plt.title(f'{title} - Top {top_n} Features')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

