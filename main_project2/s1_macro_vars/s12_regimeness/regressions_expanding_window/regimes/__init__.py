"""
Regime Detection Module

This module contains code for detecting economic regimes using expanding windows
to avoid look-ahead bias.
"""

from .regime_detection_expanding_window import ExpandingWindowRegimeDetector

__all__ = ['ExpandingWindowRegimeDetector']


