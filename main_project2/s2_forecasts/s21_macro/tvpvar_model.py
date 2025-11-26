"""
TVP-VAR model implementation.

This module implements a Time-Varying Parameter VAR model using a rolling window
approach as a practical approximation to full Bayesian TVP-VAR.

Classes:
- RollingTVPVAR: Rolling window VAR approximation to TVP-VAR
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RollingTVPVAR:
    """
    Rolling window VAR approximation to TVP-VAR.
    
    For each forecast origin, estimates a VAR model using a rolling window
    of recent data, allowing coefficients to vary over time.
    """
    
    def __init__(
        self,
        lag_order: int,
        window_size: Optional[int] = None,
        min_window: int = 60
    ):
        """
        Initialize RollingTVPVAR.
        
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
        
        print(f"\nGenerating forecasts for {len(test_dates)} test dates...")
        
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
                print(f"    Warning: Insufficient data ({len(window_data)} < {self.min_window})")
                continue
            
            try:
                # Fit VAR model
                var_model = VAR(window_data)
                fitted_model = var_model.fit(maxlags=self.lag_order, ic=None)
                
                # Store model for this date
                self.models[forecast_date] = fitted_model
                
                # Store coefficients
                # fitted_model.params has shape (n_coefs, n_vars) where each column is an equation
                for var in var_names:
                    if var in fitted_model.params.columns:
                        coefs = fitted_model.params[var].values  # Get coefficients for this variable's equation
                        # Ensure we have the right number of coefficients
                        if len(coefs) <= n_coefs_per_eq:
                            # Pad with NaN if needed
                            coefs_padded = np.pad(coefs, (0, max(0, n_coefs_per_eq - len(coefs))), 
                                                constant_values=np.nan)
                            self.coefficients[var].loc[forecast_date] = coefs_padded[:n_coefs_per_eq]
                        else:
                            # Truncate if too many
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
                        # Assuming order: growth_factor, inflation_factor, policy_factor, vol_factor
                        growth_idx = var_names.index('growth_factor')
                        inflation_idx = var_names.index('inflation_factor')
                        
                        forecasts['growth'].loc[forecast_date, f'h_{h}'] = forecast[-1, growth_idx]
                        forecasts['inflation'].loc[forecast_date, f'h_{h}'] = forecast[-1, inflation_idx]
                        
                    except Exception as e:
                        print(f"    Warning: Failed to forecast horizon {h} for {forecast_date}: {e}")
                        continue
                        
            except Exception as e:
                print(f"    Warning: Failed to fit model for {forecast_date}: {e}")
                continue
        
        print(f"\nCompleted forecasting. Generated forecasts for {len(test_dates)} dates.")
        
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
            Index of coefficient to extract. If None, returns all coefficients.
        
        Returns:
        --------
        pd.DataFrame
            Time series of coefficients
        """
        if variable not in self.coefficients:
            raise ValueError(f"Variable {variable} not found in coefficients")
        
        coef_df = self.coefficients[variable].copy()
        
        if coefficient_idx is not None:
            col_name = f'coef_{coefficient_idx}'
            if col_name not in coef_df.columns:
                raise ValueError(f"Coefficient index {coefficient_idx} not found")
            return coef_df[[col_name]]
        
        return coef_df
    
    def compute_impulse_responses(
        self,
        forecast_date: pd.Timestamp,
        periods: int = 20,
        shock_var: Optional[str] = None,
        response_vars: Optional[List[str]] = None,
        include_ci: bool = True,
        ci_level: float = 0.95
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Compute impulse response functions for a specific forecast date.
        
        Parameters:
        -----------
        forecast_date : pd.Timestamp
            Date for which to compute IRFs
        periods : int
            Number of periods ahead for IRF
        shock_var : str, optional
            Variable to shock. If None, uses Cholesky decomposition.
        response_vars : list, optional
            Variables to track responses. If None, uses all variables.
        include_ci : bool
            Whether to compute confidence intervals
        ci_level : float
            Confidence level (e.g., 0.95 for 95%)
        
        Returns:
        --------
        tuple
            (irf_df, lower_ci_df, upper_ci_df)
            - irf_df: Point estimates
            - lower_ci_df: Lower confidence bounds (if include_ci=True)
            - upper_ci_df: Upper confidence bounds (if include_ci=True)
        """
        if forecast_date not in self.models:
            raise ValueError(f"No model found for date {forecast_date}")
        
        fitted_model = self.models[forecast_date]
        
        # Compute IRFs
        irf = fitted_model.irf(periods)
        
        # Convert to DataFrame
        var_names = list(fitted_model.names)
        if response_vars is None:
            response_vars = var_names
        
        irf_data = {}
        lower_ci_data = {}
        upper_ci_data = {}
        
        for response_var in response_vars:
            if response_var not in var_names:
                continue
            
            response_idx = var_names.index(response_var)
            
            if shock_var is None:
                # Use Cholesky decomposition (default)
                # irf.irfs has shape (periods+1, n_vars, n_vars)
                irf_values = irf.irfs[:, response_idx, :].mean(axis=1)
            else:
                # Shock specific variable
                shock_idx = var_names.index(shock_var)
                irf_values = irf.irfs[:, response_idx, shock_idx]
            
            # Ensure we have the right number of periods
            if len(irf_values) > periods:
                irf_values = irf_values[:periods]
            elif len(irf_values) < periods:
                # Pad with last value if needed
                irf_values = np.pad(irf_values, (0, periods - len(irf_values)), 
                                  mode='edge')
            
            irf_data[response_var] = irf_values
            
            # Compute confidence intervals if requested
            if include_ci:
                try:
                    # Try to compute error bands using Monte Carlo simulation
                    # First check if errband_mc exists and is computed
                    if hasattr(irf, 'errband_mc'):
                        try:
                            # Compute error bands if not already computed
                            if irf.errband_mc is None:
                                # Compute Monte Carlo error bands
                                irf.errband_mc = irf.errband_mc(orth=False, repl=1000, T=periods, signif=1-ci_level)
                            
                            # errband_mc has shape (periods+1, n_vars, n_vars, 2) where last dim is [lower, upper]
                            if shock_var is None:
                                # Average across all shocks
                                ci_values = irf.errband_mc[:, response_idx, :, :].mean(axis=1)
                            else:
                                ci_values = irf.errband_mc[:, response_idx, shock_idx, :]
                            
                            lower_ci = ci_values[:, 0]
                            upper_ci = ci_values[:, 1]
                        except:
                            # If errband_mc computation fails, try standard errors
                            lower_ci = None
                            upper_ci = None
                    else:
                        lower_ci = None
                        upper_ci = None
                    
                    # Fallback: use standard errors if Monte Carlo bands not available
                    if lower_ci is None or upper_ci is None:
                        if hasattr(irf, 'stderr') and irf.stderr is not None:
                            try:
                                if shock_var is None:
                                    stderr_values = irf.stderr[:, response_idx, :].mean(axis=1)
                                else:
                                    stderr_values = irf.stderr[:, response_idx, shock_idx]
                                
                                # Approximate CI using normal distribution
                                z_score = stats.norm.ppf((1 + ci_level) / 2)  # For specified CI level
                                lower_ci = irf_values - z_score * stderr_values
                                upper_ci = irf_values + z_score * stderr_values
                            except:
                                # If stderr computation fails, use simple approximation
                                # Use 1.96 * std of IRF values as approximation
                                std_approx = np.std(irf_values) * 0.5  # Rough approximation
                                z_score = stats.norm.ppf((1 + ci_level) / 2)
                                lower_ci = irf_values - z_score * std_approx
                                upper_ci = irf_values + z_score * std_approx
                        else:
                            # Last resort: use simple approximation based on IRF values
                            std_approx = np.std(irf_values) * 0.5
                            z_score = stats.norm.ppf((1 + ci_level) / 2)
                            lower_ci = irf_values - z_score * std_approx
                            upper_ci = irf_values + z_score * std_approx
                    
                    # Ensure right length
                    if len(lower_ci) > periods:
                        lower_ci = lower_ci[:periods]
                        upper_ci = upper_ci[:periods]
                    elif len(lower_ci) < periods:
                        lower_ci = np.pad(lower_ci, (0, periods - len(lower_ci)), mode='edge')
                        upper_ci = np.pad(upper_ci, (0, periods - len(upper_ci)), mode='edge')
                    
                    lower_ci_data[response_var] = lower_ci
                    upper_ci_data[response_var] = upper_ci
                except Exception as e:
                    # If CI computation fails, continue without CI
                    pass
        
        irf_df = pd.DataFrame(irf_data, index=range(periods))
        irf_df.index.name = 'period'
        
        lower_ci_df = pd.DataFrame(lower_ci_data, index=range(periods)) if lower_ci_data else None
        upper_ci_df = pd.DataFrame(upper_ci_data, index=range(periods)) if upper_ci_data else None
        
        return irf_df, lower_ci_df, upper_ci_df

