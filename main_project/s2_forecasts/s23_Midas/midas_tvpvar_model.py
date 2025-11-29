"""
MIDAS TVP-VAR model implementation.

This module implements a MIDAS (Mixed Data Sampling) augmented TVP-VAR model
that combines monthly macro factors (growth, inflation, policy, volatility)
with daily oil price data through a MIDAS framework.

Key innovation: Oil prices (daily) are aggregated via exponential weighting
into monthly MIDAS factors, then included in the TVP-VAR alongside the 4 macro factors.

Classes:
- MidasTVPVAR: Time-varying parameter VAR with MIDAS oil factor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings('ignore')


class MidasTVPVAR:
    """
    MIDAS TVP-VAR model: 5-variable VAR with daily oil aggregation.
    
    Variables:
    1. growth_factor (monthly)
    2. inflation_factor (monthly)
    3. monetary_policy_factor (monthly)
    4. market_volatility_factor (monthly)
    5. oil_midas (monthly, aggregated from daily prices)
    
    The model fits a rolling/expanding window VAR and generates forecasts
    for growth and inflation at each test date.
    """
    
    def __init__(
        self,
        lag_order: int,
        window_size: Optional[int] = None,
        min_window: int = 60
    ):
        """
        Initialize MidasTVPVAR.
        
        Parameters:
        -----------
        lag_order : int
            Number of lags in VAR model
        window_size : int, optional
            Size of rolling window. If None, uses expanding window.
        min_window : int
            Minimum number of observations required for estimation
        """
        self.lag_order = lag_order
        self.window_size = window_size
        self.min_window = min_window
        self.models = {}  # Store fitted models for each forecast origin
        self.coefficients = {}  # Store time-varying coefficients
        
    def fit_forecast(
        self,
        train_data: pd.DataFrame,
        test_dates: pd.DatetimeIndex,
        horizons: List[int] = [1, 3, 6]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fit model and generate forecasts for each test date.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data (full historical data up to test period)
            Columns: growth_factor, inflation_factor, monetary_policy_factor,
                    market_volatility_factor, oil_midas
        test_dates : pd.DatetimeIndex
            Dates at which to generate forecasts
        horizons : list
            Forecast horizons in months
        
        Returns:
        --------
        dict
            Dictionary with keys 'growth' and 'inflation', each containing
            a DataFrame with forecasts for each horizon
        """
        # Prepare output dictionaries
        forecasts = {
            'growth': pd.DataFrame(index=test_dates, columns=[f'h_{h}' for h in horizons]),
            'inflation': pd.DataFrame(index=test_dates, columns=[f'h_{h}' for h in horizons])
        }
        
        # Store coefficients over time
        var_names = list(train_data.columns)
        n_vars = len(var_names)
        n_coefs_per_eq = n_vars * self.lag_order + 1  # +1 for intercept
        
        # Initialize coefficient storage
        self.coefficients = {
            var: pd.DataFrame(
                index=test_dates,
                columns=[f'coef_{i}' for i in range(n_coefs_per_eq)]
            )
            for var in var_names
        }
        
        print(f"\nGenerating MIDAS TVP-VAR forecasts for {len(test_dates)} test dates...")
        print(f"Variables: {var_names}")
        print(f"Lag order: {self.lag_order}")
        print(f"Window mode: {'Expanding' if self.window_size is None else f'Rolling ({self.window_size})'}")
        
        successful_forecasts = 0
        
        for idx, forecast_date in enumerate(test_dates):
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  Processing forecast origin {idx + 1}/{len(test_dates)}: {forecast_date.date()}")
            
            # Get data up to forecast_date
            available_data = train_data.loc[:forecast_date].copy()
            
            # Determine window
            if self.window_size is None:
                # Expanding window: use all available data
                window_data = available_data
            else:
                # Rolling window: use last window_size observations
                if len(available_data) > self.window_size:
                    window_data = available_data.iloc[-self.window_size:].copy()
                else:
                    window_data = available_data
            
            # Check minimum window requirement
            if len(window_data) < self.min_window:
                continue
            
            try:
                # Fit VAR model on the window
                var_model = VAR(window_data)
                fitted_model = var_model.fit(maxlags=self.lag_order, ic=None)
                
                # Store model for this date
                self.models[forecast_date] = fitted_model
                
                # Store coefficients
                for var in var_names:
                    if var in fitted_model.params.columns:
                        coefs = fitted_model.params[var].values
                        if len(coefs) <= n_coefs_per_eq:
                            coefs_padded = np.pad(coefs, (0, max(0, n_coefs_per_eq - len(coefs))), 
                                                constant_values=np.nan)
                            self.coefficients[var].loc[forecast_date] = coefs_padded[:n_coefs_per_eq]
                        else:
                            self.coefficients[var].loc[forecast_date] = coefs[:n_coefs_per_eq]
                
                # Generate forecasts for each horizon
                for h in horizons:
                    try:
                        # Generate h-step-ahead forecast
                        forecast = fitted_model.forecast(
                            y=window_data.values[-self.lag_order:],
                            steps=h
                        )
                        
                        # Extract forecasts for growth and inflation
                        # Assuming order: growth_factor, inflation_factor, policy_factor, vol_factor, oil_midas
                        growth_idx = var_names.index('growth_factor')
                        inflation_idx = var_names.index('inflation_factor')
                        
                        forecasts['growth'].loc[forecast_date, f'h_{h}'] = forecast[-1, growth_idx]
                        forecasts['inflation'].loc[forecast_date, f'h_{h}'] = forecast[-1, inflation_idx]
                        
                    except Exception as e:
                        pass  # Continue if forecast fails for this horizon
                
                successful_forecasts += 1
                        
            except Exception as e:
                continue
        
        print(f"\nCompleted MIDAS TVP-VAR forecasting.")
        print(f"  Successfully forecasted: {successful_forecasts}/{len(test_dates)} dates")
        
        return forecasts
    
    def get_time_varying_coefficients(
        self,
        variable: str,
        coefficient_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract time-varying coefficients for a specific variable.
        
        Parameters:
        -----------
        variable : str
            Variable name (e.g., 'growth_factor', 'inflation_factor')
        coefficient_idx : int, optional
            If None, returns all coefficients. Otherwise returns specific coefficient index.
        
        Returns:
        --------
        pd.DataFrame
            Time-varying coefficients
        """
        if variable not in self.coefficients:
            raise ValueError(f"Variable {variable} not found in coefficients")
        
        coefs = self.coefficients[variable].copy()
        
        if coefficient_idx is not None:
            return coefs[f'coef_{coefficient_idx}']
        
        return coefs


def create_midas_oil_factor(
    daily_oil: pd.Series,
    theta: float = 0.03,
    K: int = 60
) -> pd.Series:
    """
    Convert daily oil prices to monthly MIDAS factor using exponential weights.
    
    Parameters:
    -----------
    daily_oil : pd.Series
        Daily oil prices with date index
    theta : float
        Decay parameter for exponential weights (higher = faster decay)
    K : int
        Number of daily lags to use in aggregation
    
    Returns:
    --------
    pd.Series
        Monthly MIDAS oil factor
    """
    # Create exponential weights: more recent days get more weight
    weights = np.exp(-theta * np.arange(K))
    weights = weights / weights.sum()  # Normalize
    
    # Rolling MIDAS convolution
    daily_oil_series = daily_oil.copy()
    daily_oil_series['oil_midas'] = (
        daily_oil_series
        .rolling(K)
        .apply(lambda x: np.sum(x.values * weights), raw=False)
    )
    
    # Resample to monthly, taking last trading day of each month
    monthly = daily_oil_series['oil_midas'].resample('ME').last()
    
    # Ensure proper monthly index
    monthly.index = monthly.index.to_period("M").to_timestamp()
    
    return monthly


def prepare_midas_data(
    macro_df: pd.DataFrame,
    oil_midas_series: pd.Series
) -> pd.DataFrame:
    """
    Prepare merged data: 4 macro factors + oil MIDAS factor.
    
    Parameters:
    -----------
    macro_df : pd.DataFrame
        DataFrame with 4 macro factors (monthly)
    oil_midas_series : pd.Series
        Monthly oil MIDAS factor
    
    Returns:
    --------
    pd.DataFrame
        Merged 5-variable dataset (growth, inflation, policy, vol, oil_midas)
    """
    # Ensure date indices
    macro_df = macro_df.copy()
    macro_df.index = pd.to_datetime(macro_df.index)
    oil_midas_series = oil_midas_series.copy()
    oil_midas_series.index = pd.to_datetime(oil_midas_series.index)
    
    # Merge
    merged = pd.DataFrame({
        'growth_factor': macro_df['growth_factor'],
        'inflation_factor': macro_df['inflation_factor'],
        'monetary_policy_factor': macro_df['monetary_policy_factor'],
        'market_volatility_factor': macro_df['market_volatility_factor'],
        'oil_midas': oil_midas_series
    })
    
    # Forward fill oil_midas if there are gaps
    merged['oil_midas'] = merged['oil_midas'].ffill()
    
    # Drop any remaining NaN
    merged = merged.dropna()
    
    print(f"Prepared MIDAS data: {len(merged)} observations")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")
    
    return merged
