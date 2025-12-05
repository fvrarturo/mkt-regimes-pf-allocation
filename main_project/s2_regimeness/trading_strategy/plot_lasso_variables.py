"""
Plotting functions for LASSO variable inclusion over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.colors as mcolors


def plot_variable_inclusion_over_time(
    variable_inclusion_history: Dict[pd.Timestamp, Dict[int, List[str]]],
    macro_variables: List[str],
    output_path: Optional[Path] = None,
    title: str = "Variable Inclusion Over Time",
    is_hmm: bool = False
) -> None:
    """
    Plot which variables are included in LASSO regressions over time.
    
    Parameters:
    -----------
    variable_inclusion_history : Dict[date] -> Dict[regime] -> List[var]
        History of variable inclusion for each regime
    macro_variables : List[str]
        List of all macro variables
    output_path : Path, optional
        Path to save the plot
    title : str
        Plot title
    is_hmm : bool
        If True, use weighted inclusion (percentage) for HMM
        If False, use binary inclusion for 2x2
    """
    if not variable_inclusion_history:
        print("Warning: No variable inclusion history to plot")
        return
    
    # Get all dates and sort
    dates = sorted(variable_inclusion_history.keys())
    
    # Create inclusion matrix: dates x variables
    inclusion_matrix = np.zeros((len(dates), len(macro_variables)))
    
    for i, date in enumerate(dates):
        regime_inclusions = variable_inclusion_history[date]
        
        if is_hmm:
            # For HMM: compute weighted inclusion percentage
            # Sum probabilities of regimes where variable is included
            # We need to get regime probabilities for this date
            # For now, assume uniform weighting across regimes
            # In practice, this should be computed from actual regime probabilities
            n_regimes = len(regime_inclusions)
            for j, var in enumerate(macro_variables):
                # Count how many regimes include this variable
                inclusion_count = sum(
                    1 for regime_id in range(n_regimes)
                    if var in regime_inclusions.get(regime_id, [])
                )
                # Percentage inclusion (assuming uniform regime weights)
                inclusion_matrix[i, j] = inclusion_count / n_regimes if n_regimes > 0 else 0.0
        else:
            # For 2x2: binary inclusion (variable included in current regime)
            # Get the current regime (most common or use regime 0 as default)
            # Actually, for 2x2 we need to know which regime is active at each date
            # For now, mark as included if it's in ANY regime at this date
            for j, var in enumerate(macro_variables):
                included = any(
                    var in regime_inclusions.get(regime_id, [])
                    for regime_id in range(4)
                )
                inclusion_matrix[i, j] = 1.0 if included else 0.0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    if is_hmm:
        # Use colormap for percentage inclusion (shades of blue)
        cmap = plt.cm.Blues
        im = ax.imshow(inclusion_matrix.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Inclusion Percentage')
    else:
        # Use binary colormap (light blue = 0, dark blue = 1)
        colors = ['#E3F2FD', '#1976D2']  # Light blue, dark blue
        cmap = mcolors.ListedColormap(colors)
        im = ax.imshow(inclusion_matrix.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75], label='Included')
        cbar.ax.set_yticklabels(['Not Included', 'Included'])
    
    # Set labels
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro Variable', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set y-axis labels to variable names
    ax.set_yticks(range(len(macro_variables)))
    ax.set_yticklabels(macro_variables, fontsize=9)
    
    # Set x-axis labels to years
    year_positions = []
    year_labels = []
    for i, date in enumerate(dates):
        year = date.year
        if year not in year_labels:
            year_positions.append(i)
            year_labels.append(year)
    
    # Show every 2-3 years to avoid crowding
    if len(year_positions) > 10:
        step = max(1, len(year_positions) // 10)
        year_positions = year_positions[::step]
        year_labels = year_labels[::step]
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved variable inclusion plot to {output_path}")
    
    plt.close()


def plot_hmm_variable_inclusion_weighted(
    variable_inclusion_history: Dict[pd.Timestamp, Dict[int, List[str]]],
    regime_probabilities_history: Dict[pd.Timestamp, np.ndarray],
    macro_variables: List[str],
    output_path: Optional[Path] = None
) -> None:
    """
    Plot HMM variable inclusion weighted by regime probabilities.
    
    Parameters:
    -----------
    variable_inclusion_history : Dict[date] -> Dict[regime] -> List[var]
        History of variable inclusion for each regime
    regime_probabilities_history : Dict[date] -> np.ndarray
        History of regime probabilities (shape: [n_regimes])
    macro_variables : List[str]
        List of all macro variables
    output_path : Path, optional
        Path to save the plot
    """
    if not variable_inclusion_history:
        print("Warning: No variable inclusion history to plot")
        return
    
    # Get all dates and sort
    dates = sorted(variable_inclusion_history.keys())
    
    # Create weighted inclusion matrix: dates x variables
    inclusion_matrix = np.zeros((len(dates), len(macro_variables)))
    
    for i, date in enumerate(dates):
        regime_inclusions = variable_inclusion_history[date]
        regime_probs = regime_probabilities_history.get(date, np.ones(len(regime_inclusions)) / len(regime_inclusions))
        
        # Normalize probabilities
        if len(regime_probs) != len(regime_inclusions):
            regime_probs = np.ones(len(regime_inclusions)) / len(regime_inclusions)
        regime_probs = regime_probs / regime_probs.sum()  # Ensure normalization
        
        for j, var in enumerate(macro_variables):
            # Weighted sum: sum of probabilities for regimes where variable is included
            weighted_inclusion = sum(
                regime_probs[regime_id]
                for regime_id in range(len(regime_inclusions))
                if var in regime_inclusions.get(regime_id, [])
            )
            inclusion_matrix[i, j] = weighted_inclusion
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use colormap for percentage inclusion (shades of blue)
    cmap = plt.cm.Blues
    im = ax.imshow(inclusion_matrix.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, label='Weighted Inclusion Percentage')
    
    # Set labels
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro Variable', fontsize=12, fontweight='bold')
    ax.set_title('HMM LASSO Variable Inclusion (Weighted by Regime Probabilities)', fontsize=14, fontweight='bold')
    
    # Set y-axis labels to variable names
    ax.set_yticks(range(len(macro_variables)))
    ax.set_yticklabels(macro_variables, fontsize=9)
    
    # Set x-axis labels to years
    year_positions = []
    year_labels = []
    for i, date in enumerate(dates):
        year = date.year
        if year not in year_labels:
            year_positions.append(i)
            year_labels.append(year)
    
    # Show every 2-3 years to avoid crowding
    if len(year_positions) > 10:
        step = max(1, len(year_positions) // 10)
        year_positions = year_positions[::step]
        year_labels = year_labels[::step]
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved HMM variable inclusion plot to {output_path}")
    
    plt.close()

