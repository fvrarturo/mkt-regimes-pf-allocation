"""
PCA Distance model for extremeness detection.

Uses PCA to reduce dimensionality and computes distance from center in PC space.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis


def fit_pca(X, variance_threshold=0.85):
    """
    Fit PCA model, selecting number of components to explain variance_threshold.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix (already standardized)
    variance_threshold : float
        Minimum variance to explain (default 0.85 = 85%)
    
    Returns:
    --------
    tuple
        (pca_model, n_components) where n_components is the selected number
    """
    # Start with all components
    pca = PCA()
    pca.fit(X)
    
    # Find number of components needed to explain variance_threshold
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Refit with selected number of components
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    return pca, n_components


def compute_pc_scores(pca_model, X):
    """
    Compute principal component scores.
    
    Parameters:
    -----------
    pca_model : PCA
        Fitted PCA model
    X : np.array
        Feature matrix
    
    Returns:
    --------
    np.array
        PC scores (n_samples, n_components)
    """
    return pca_model.transform(X)


def compute_distance_from_center(pc_scores, method='euclidean'):
    """
    Compute distance from center in PC space.
    
    Parameters:
    -----------
    pc_scores : np.array
        Principal component scores
    method : str
        Distance method: 'euclidean' or 'mahalanobis'
    
    Returns:
    --------
    np.array
        Distances from center
    """
    # Center is at origin (since we standardized)
    center = np.zeros(pc_scores.shape[1])
    
    if method == 'euclidean':
        # Simple Euclidean distance
        distances = np.sqrt(np.sum((pc_scores - center) ** 2, axis=1))
    
    elif method == 'mahalanobis':
        # Mahalanobis distance (accounts for covariance)
        cov_matrix = np.cov(pc_scores.T)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            distances = np.array([
                mahalanobis(pc_scores[i], center, cov_inv)
                for i in range(len(pc_scores))
            ])
        except np.linalg.LinAlgError:
            # Fallback to Euclidean if covariance matrix is singular
            distances = np.sqrt(np.sum((pc_scores - center) ** 2, axis=1))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return distances


def normalize_extremeness(distances):
    """
    Normalize distances to create extremeness index (0-1 scale).
    
    Parameters:
    -----------
    distances : np.array
        Raw distances from center
    
    Returns:
    --------
    np.array
        Normalized extremeness scores (0-1, higher = more extreme)
    """
    # Min-max normalization
    extremeness = (distances - distances.min()) / (distances.max() - distances.min())
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


def run_pca_distance_analysis(df, feature_cols, variance_threshold=0.85, 
                              distance_method='euclidean', threshold_percentile=90):
    """
    Run complete PCA distance analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with features and ERP
    feature_cols : list
        List of feature column names
    variance_threshold : float
        Minimum variance to explain with PCA
    distance_method : str
        Distance method: 'euclidean' or 'mahalanobis'
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
    
    # Fit PCA
    pca_model, n_components = fit_pca(X_scaled, variance_threshold)
    
    # Compute PC scores
    pc_scores = compute_pc_scores(pca_model, X_scaled)
    
    # Compute distances from center
    distances = compute_distance_from_center(pc_scores, method=distance_method)
    
    # Normalize to extremeness index
    extremeness = normalize_extremeness(distances)
    
    # Flag extreme states for primary threshold
    is_extreme = flag_extreme_states(extremeness, threshold_percentile)
    
    # Flag extreme states for multiple percentiles
    percentile_flags = flag_extreme_states_multiple(extremeness, percentiles=[99, 95, 90, 80])
    
    # Create results dataframe
    results_df = df.copy()
    results_df['extremeness'] = extremeness
    results_df['is_extreme'] = is_extreme
    results_df['distance'] = distances
    
    # Add PC scores as columns
    for i in range(n_components):
        results_df[f'PC{i+1}'] = pc_scores[:, i]
    
    # Add percentile flags to dataframe
    for p, flags in percentile_flags.items():
        results_df[f'is_extreme_p{p}'] = flags['is_extreme']
    
    return {
        'pca_model': pca_model,
        'scaler': scaler,
        'results_df': results_df,
        'extremeness': extremeness,
        'is_extreme': is_extreme,
        'percentile_flags': percentile_flags,
        'distances': distances,
        'pc_scores': pc_scores,
        'n_components': n_components,
        'explained_variance_ratio': pca_model.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca_model.explained_variance_ratio_),
        'feature_cols': feature_cols,
        'variance_threshold': variance_threshold,
        'distance_method': distance_method,
        'threshold_percentile': threshold_percentile
    }

