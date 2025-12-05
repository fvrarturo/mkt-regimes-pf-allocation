"""
Analysis functions for trading strategy.

This module handles ERP statistics and z-score calculations.
"""

import pandas as pd
import numpy as np


def geometric_return(r):
    """
    Calculate geometric average return.
    
    Parameters
    ----------
    r : pd.Series
        Return series
    
    Returns
    -------
    float
        Geometric mean return
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    gross = (1 + r).prod()
    return gross ** (1 / len(r)) - 1


def calculate_erp_statistics(erp):
    """
    Calculate summary statistics for ERP.
    
    Parameters
    ----------
    erp : pd.Series
        Equity risk premium series
    
    Returns
    -------
    dict
        Dictionary with statistics
    """
    erp = erp.dropna()
    
    g_erp_m = geometric_return(erp)
    a_erp_m = erp.mean()
    std_erp = erp.std()
    
    return {
        "geometric_mean": g_erp_m,
        "arithmetic_mean": a_erp_m,
        "std": std_erp,
        "p5": erp.quantile(0.05),
        "p50": erp.quantile(0.50),
        "p95": erp.quantile(0.95),
    }


def print_erp_statistics(erp):
    """Print ERP statistics."""
    stats = calculate_erp_statistics(erp)
    
    print("\nERP geometric monthly mean   : {:.4%}".format(stats["geometric_mean"]))
    print("ERP arithmetic monthly mean  : {:.4%}".format(stats["arithmetic_mean"]))
    print("ERP monthly std deviation    : {:.4%}".format(stats["std"]))
    print("ERP 5th / 95th percentiles   : {:.4%} / {:.4%}".format(
        stats["p5"], stats["p95"]
    ))


def calculate_z_scores(erp):
    """
    Calculate z-scores for ERP.
    
    Parameters
    ----------
    erp : pd.Series
        Equity risk premium series
    
    Returns
    -------
    pd.Series
        Z-scores
    """
    erp = erp.dropna()
    erp_mean = erp.mean()
    erp_std = erp.std()
    z = (erp - erp_mean) / erp_std
    return z

