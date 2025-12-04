"""
Script to create forecast comparison plots showing all models together.

This script creates side-by-side comparisons of forecasts from all models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir / "s21_macro"))
sys.path.append(str(base_dir / "s22_ml_based"))

from plotting import plot_forecast_comparison_all_models
from preprocessing import load_data as load_macro_data


def load_forecasts_from_models(base_dir: Path, horizons: list = [1, 3, 6]) -> dict:
    """
    Load forecast series from all models.
    
    Note: This requires that forecasts are saved or can be regenerated.
    For now, this is a placeholder structure.
    
    Parameters:
    -----------
    base_dir : Path
        Base directory
    horizons : list
        Forecast horizons
    
    Returns:
    --------
    dict
        Nested dictionary: forecasts[variable][horizon][model_name] = Series
    """
    forecasts = {
        'growth_factor': {},
        'inflation_factor': {}
    }
    
    # TODO: Implement actual loading of forecast series
    # This would require:
    # 1. Loading TVP-VAR forecasts from s21_macro/results
    # 2. Loading XGBoost forecasts (would need to be saved)
    # 3. Loading LSTM forecasts (would need to be saved)
    
    # For now, return empty structure
    # In practice, you'd load saved forecast CSV files or regenerate them
    
    return forecasts


def create_forecast_comparison_plots(
    base_dir: Path,
    horizons: list = [1, 3, 6],
    start_date: str = "2008-01-01",
    output_dir: Optional[Path] = None
) -> None:
    """
    Create forecast comparison plots for all models.
    
    Parameters:
    -----------
    base_dir : Path
        Base directory
    horizons : list
        Forecast horizons to plot
    start_date : str
        Start date for plotting
    output_dir : Path, optional
        Output directory
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load macro data for actuals
    macro_df, _ = load_macro_data(include_sentiment=False)
    
    # Load forecasts (placeholder - would need actual implementation)
    forecasts = load_forecasts_from_models(base_dir, horizons)
    
    # For each variable and horizon, create comparison plot
    for variable in ['growth_factor', 'inflation_factor']:
        actuals = macro_df[variable]
        
        for horizon in horizons:
            # Collect forecasts from all models for this variable/horizon
            forecasts_dict = {}
            
            # TVP-VAR forecasts (would need to load from saved files)
            # XGBoost forecasts (would need to load from saved files)
            # LSTM forecasts (would need to load from saved files)
            
            # For now, skip if no forecasts available
            if len(forecasts_dict) == 0:
                print(f"Skipping {variable} h={horizon}: No forecast data available")
                continue
            
            # Create plot
            plot_forecast_comparison_all_models(
                forecasts_dict=forecasts_dict,
                actuals=actuals,
                variable_name=variable,
                horizon=horizon,
                start_date=start_date,
                output_dir=output_dir
            )


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    create_forecast_comparison_plots(base_dir, horizons=[3])  # Focus on h=3 as specified in goals
    print("Forecast comparison plots created!")

