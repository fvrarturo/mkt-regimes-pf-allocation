"""
Regime Definitions for 2x2 Growth × Inflation Classification

Defines the four macro regimes based on Growth and Inflation thresholds.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class RegimeDefinitions:
    """
    Defines 2x2 regime classification based on Growth × Inflation.
    
    Regimes:
    1. High Growth / Low Inflation - "Goldilocks"
    2. High Growth / High Inflation - "Overheating"
    3. Low Growth / High Inflation - "Stagflation"
    4. Low Growth / Low Inflation - "Slowdown / Disinflation"
    """
    
    REGIME_NAMES = {
        0: "Goldilocks (High G / Low I)",
        1: "Overheating (High G / High I)",
        2: "Stagflation (Low G / High I)",
        3: "Slowdown / Disinflation (Low G / Low I)"
    }
    
    REGIME_SHORT_NAMES = {
        0: "Goldilocks",
        1: "Overheating",
        2: "Stagflation",
        3: "Slowdown"
    }
    
    REGIME_COLORS = {
        0: '#2ca02c',  # Green - Goldilocks
        1: '#d62728',  # Red - Overheating
        2: '#ff7f0e',  # Orange - Stagflation
        3: '#1f77b4'   # Blue - Slowdown
    }
    
    def __init__(
        self,
        growth_threshold: float = None,
        inflation_threshold: float = None,
        threshold_method: str = 'median'
    ):
        """
        Initialize regime definitions.
        
        Parameters:
        -----------
        growth_threshold : float, optional
            Threshold for high/low growth. If None, will use threshold_method.
        inflation_threshold : float, optional
            Threshold for high/low inflation. If None, will use threshold_method.
        threshold_method : str
            Method to determine thresholds if not provided:
            - 'median': Use median value
            - 'zero': Use zero (for standardized data)
            - 'mean': Use mean value
        """
        self.growth_threshold = growth_threshold
        self.inflation_threshold = inflation_threshold
        self.threshold_method = threshold_method
    
    def determine_thresholds(
        self,
        growth_data: pd.Series,
        inflation_data: pd.Series
    ) -> Tuple[float, float]:
        """
        Determine thresholds for growth and inflation.
        
        Parameters:
        -----------
        growth_data : pd.Series
            Growth factor values
        inflation_data : pd.Series
            Inflation factor values
        
        Returns:
        --------
        Tuple of (growth_threshold, inflation_threshold)
        """
        if self.growth_threshold is None:
            if self.threshold_method == 'median':
                self.growth_threshold = growth_data.median()
            elif self.threshold_method == 'zero':
                self.growth_threshold = 0.0
            elif self.threshold_method == 'mean':
                self.growth_threshold = growth_data.mean()
            else:
                raise ValueError(f"Unknown threshold_method: {self.threshold_method}")
        
        if self.inflation_threshold is None:
            if self.threshold_method == 'median':
                self.inflation_threshold = inflation_data.median()
            elif self.threshold_method == 'zero':
                self.inflation_threshold = 0.0
            elif self.threshold_method == 'mean':
                self.inflation_threshold = inflation_data.mean()
            else:
                raise ValueError(f"Unknown threshold_method: {self.threshold_method}")
        
        return self.growth_threshold, self.inflation_threshold
    
    def classify_regime(
        self,
        growth: float,
        inflation: float
    ) -> int:
        """
        Classify a single observation into one of 4 regimes.
        
        Parameters:
        -----------
        growth : float
            Growth factor value
        inflation : float
            Inflation factor value
        
        Returns:
        --------
        int: Regime ID (0-3)
        """
        if growth >= self.growth_threshold:
            if inflation >= self.inflation_threshold:
                return 1  # High G / High I - Overheating
            else:
                return 0  # High G / Low I - Goldilocks
        else:
            if inflation >= self.inflation_threshold:
                return 2  # Low G / High I - Stagflation
            else:
                return 3  # Low G / Low I - Slowdown
    
    def classify_dataframe(
        self,
        data: pd.DataFrame,
        growth_col: str = 'growth_factor',
        inflation_col: str = 'inflation_factor'
    ) -> pd.Series:
        """
        Classify entire dataframe into regimes.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with growth and inflation columns
        growth_col : str
            Name of growth column
        inflation_col : str
            Name of inflation column
        
        Returns:
        --------
        pd.Series: Regime assignments (0-3)
        """
        # Determine thresholds if needed
        self.determine_thresholds(
            data[growth_col],
            data[inflation_col]
        )
        
        # Classify each row
        regimes = data.apply(
            lambda row: self.classify_regime(
                row[growth_col],
                row[inflation_col]
            ),
            axis=1
        )
        
        return regimes
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get full regime name."""
        return self.REGIME_NAMES.get(regime_id, f"Unknown Regime {regime_id}")
    
    def get_regime_short_name(self, regime_id: int) -> str:
        """Get short regime name."""
        return self.REGIME_SHORT_NAMES.get(regime_id, f"R{regime_id}")
    
    def get_regime_color(self, regime_id: int) -> str:
        """Get color for regime."""
        return self.REGIME_COLORS.get(regime_id, '#808080')

