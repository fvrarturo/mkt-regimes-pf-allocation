"""
Isolation Forest model for extremeness detection.

Two versions:
- Version A: macro only (4 indices)
- Version B: macro + sentiment (8 features)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fit_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Fit Isolation Forest model.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix (already standardized)
    contamination : float
        Expected proportion of outliers (default 0.1 = 10%)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    IsolationForest
        Fitted model
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    model.fit(X)
    return model


def compute_extremeness_scores(model, X):
    """
    Compute extremeness scores from Isolation Forest.
    
    Parameters:
    -----------
    model : IsolationForest
        Fitted Isolation Forest model
    X : np.array
        Feature matrix
    
    Returns:
    --------
    np.array
        Extremeness scores (anomaly scores, negative = more extreme)
    """
    # Get anomaly scores (negative scores = anomalies)
    scores = model.score_samples(X)
    
    # Convert to extremeness index (0-1 scale, higher = more extreme)
    # Isolation Forest returns negative scores for anomalies
    # We'll flip and normalize to 0-1
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
    
    # Flip so higher values = more extreme
    extremeness = 1 - scores_normalized
    
    return extremeness


def flag_extreme_states(extremeness, threshold_percentile=90):
    """
    Flag extreme states based on extremeness threshold.
    
    Parameters:
    -----------
    extremeness : np.array
        Extremeness scores
    threshold_percentile : float
        Percentile threshold (default 90 = top 10% are extreme)
    
    Returns:
    --------
    np.array
        Boolean array (True = extreme)
    """
    threshold = np.percentile(extremeness, threshold_percentile)
    return extremeness >= threshold


def flag_extreme_states_multiple(extremeness, percentiles=[99, 95, 90, 80]):
    """
    Flag extreme states for multiple percentile thresholds.
    
    Parameters:
    -----------
    extremeness : np.array
        Extremeness scores
    percentiles : list
        List of percentile thresholds
    
    Returns:
    --------
    dict
        Dictionary with percentile thresholds and boolean arrays
    """
    results = {}
    for p in percentiles:
        threshold = np.percentile(extremeness, p)
        results[p] = {
            'threshold': threshold,
            'is_extreme': extremeness >= threshold
        }
    return results


def run_isolation_forest_analysis(df, feature_cols, contamination=0.1, threshold_percentile=90):
    """
    Run complete Isolation Forest analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with features and ERP
    feature_cols : list
        List of feature column names
    contamination : float
        Expected proportion of outliers
    threshold_percentile : float
        Percentile threshold for extreme states
    
    Returns:
    --------
    dict
        Dictionary with model results and extremeness metrics
    """
    # Extract features
    X = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = fit_isolation_forest(X_scaled, contamination=contamination)
    
    # Compute extremeness scores
    extremeness = compute_extremeness_scores(model, X_scaled)
    
    # Flag extreme states for primary threshold
    is_extreme = flag_extreme_states(extremeness, threshold_percentile)
    
    # Flag extreme states for multiple percentiles
    percentile_flags = flag_extreme_states_multiple(extremeness, percentiles=[99, 95, 90, 80])
    
    # Create results dataframe
    results_df = df.copy()
    results_df['extremeness'] = extremeness
    results_df['is_extreme'] = is_extreme
    results_df['anomaly_label'] = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    
    # Add percentile flags to dataframe
    for p, flags in percentile_flags.items():
        results_df[f'is_extreme_p{p}'] = flags['is_extreme']
    
    return {
        'model': model,
        'scaler': scaler,
        'results_df': results_df,
        'extremeness': extremeness,
        'is_extreme': is_extreme,
        'percentile_flags': percentile_flags,
        'feature_cols': feature_cols,
        'contamination': contamination,
        'threshold_percentile': threshold_percentile
    }

