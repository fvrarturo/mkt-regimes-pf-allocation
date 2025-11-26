"""
Regression Analysis Module

This module contains code for running regime-conditional regressions,
creating visualizations, and generating summaries.
"""

from .conditional_regression import RegimeConditionalRegressor
from .plotting import create_all_plots
from .summary import create_summary, create_statistics_summary

__all__ = [
    'RegimeConditionalRegressor',
    'create_all_plots',
    'create_summary',
    'create_statistics_summary'
]


