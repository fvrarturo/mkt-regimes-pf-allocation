"""
LASSO-based conditional regression forecasts with monthly retraining.
Tracks variable inclusion over time for both HMM and 2x2 regimes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
HMM_REGIMES_DIR = SCRIPT_DIR.parent / 'regimes' / 'HMM_regimes'
if str(HMM_REGIMES_DIR) not in sys.path:
    sys.path.insert(0, str(HMM_REGIMES_DIR))

TWO_BY_TWO_DIR = SCRIPT_DIR.parent / 'regimes' / '2x2_regimes'
if str(TWO_BY_TWO_DIR) not in sys.path:
    sys.path.insert(0, str(TWO_BY_TWO_DIR))

from hmm_model import HMMRegimeModel
from regime_definitions import RegimeDefinitions


class LassoConditionalForecaster:
    """
    LASSO-based conditional regression forecaster with monthly retraining.
    Tracks variable inclusion over time.
    """
    
    def __init__(
        self,
        hmm_combination: str = "2vars_inflation_market_volatility",
        hmm_k: int = 4,
        alpha_range: Tuple[float, float] = (0.001, 1.0),
        n_alphas: int = 50,
        cv_folds: int = 5
    ):
        """
        Initialize LASSO conditional forecaster.
        
        Parameters:
        -----------
        hmm_combination : str
            HMM variable combination to use
        hmm_k : int
            Number of HMM regimes
        alpha_range : tuple
            Range of alpha (regularization) values for LASSO
        n_alphas : int
            Number of alpha values to test
        cv_folds : int
            Number of CV folds for alpha selection
        """
        self.hmm_combination = hmm_combination
        self.hmm_k = hmm_k
        self.alpha_range = alpha_range
        self.n_alphas = n_alphas
        self.cv_folds = cv_folds
        
        # Storage for models and variable inclusion tracking
        self.hmm_model = None
        self.regime_def = None
        self.hmm_coefficients = {}  # Dict: date -> Dict: regime -> Dict: var -> coef
        self.two_by_two_coefficients = {}  # Dict: date -> Dict: regime -> Dict: var -> coef
        self.hmm_variable_inclusion = {}  # Dict: date -> Dict: regime -> List[var]
        self.two_by_two_variable_inclusion = {}  # Dict: date -> Dict: regime -> List[var]
        self.hmm_regime_probabilities = {}  # Dict: date -> np.ndarray (regime probabilities)
        self.macro_variables = None
        self.scaler = StandardScaler()
        
        # Macro variable directories (same as conditional regressions)
        self.macro_dirs = {
            'ec_growth': [
                'industrial_production_processed.csv',
                'retail_sales_processed.csv',
                'tot_business_inventories_processed.csv',
                'export_price_index_processed.csv',
                'import_price_index_processed.csv',
                'unemployment_processed.csv'
            ],
            'inflation': [
                'cpi_processed.csv',
                'PCE_price_index_processed.csv',
                'PPI_inflation_processed.csv'
            ],
            'mkt_vol': [
                'nat_fin_condition_indx_processed_monthly.csv',
                '10y_2y_spread_processed_monthly.csv'
            ],
            'mon_policy': [
                '10y_treasury_const_maturity_rate_processed.csv',
                'fed_reserve_discount_rate_processed.csv',
                'fedfunds_processed.csv',
                'm2_real_money_supply_processed.csv'
            ]
        }
    
    def load_macro_variables(self, base_dir: Path) -> pd.DataFrame:
        """Load all macro variables."""
        macro_data_dir = base_dir / 'data' / 'macro_processed_full'
        all_data = []
        
        for category, files in self.macro_dirs.items():
            category_dir = macro_data_dir / category
            for filename in files:
                file_path = category_dir / filename
                if not file_path.exists():
                    continue
                
                try:
                    df = pd.read_csv(file_path, parse_dates=['date'])
                    value_col = 'value' if 'value' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
                    var_name = filename.replace('_processed 2.csv', '').replace('_processed_monthly.csv', '').replace('_processed.csv', '').replace(' ', '_')
                    
                    df_subset = df[['date', value_col]].copy()
                    df_subset.columns = ['date', var_name]
                    df_subset['date'] = pd.to_datetime(df_subset['date'])
                    df_subset = df_subset.set_index('date').sort_index()
                    df_subset = df_subset.resample('ME').last()
                    all_data.append(df_subset)
                except Exception as e:
                    continue
        
        if not all_data:
            raise ValueError("No macro variables loaded")
        
        combined = pd.concat(all_data, axis=1)
        combined = combined.sort_index()
        self.macro_variables = combined
        return combined
    
    def load_hmm_model(self, base_dir: Path) -> HMMRegimeModel:
        """Load and fit HMM model."""
        main_project_dir = base_dir
        
        # Get variables for this combination
        variables = self._get_variables_from_combination(self.hmm_combination)
        
        # Load macro data for HMM
        macro_final_path = main_project_dir / 'data' / 'macro_final' / 'final_macro.csv'
        macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
        macro_final = macro_final.set_index('date').sort_index()
        
        # Extract HMM variables
        hmm_data = macro_final[variables].dropna()
        
        # Fit HMM model
        hmm_model = HMMRegimeModel(
            n_regimes=self.hmm_k,
            variables=variables,
            covar_reg=0.1,
            min_covar=0.01
        )
        hmm_model.fit(hmm_data)
        
        self.hmm_model = hmm_model
        return hmm_model
    
    def load_2x2_regime_definitions(self, base_dir: Path) -> RegimeDefinitions:
        """Load 2x2 regime definitions."""
        macro_final_path = base_dir / 'data' / 'macro_final' / 'final_macro.csv'
        macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
        
        regime_def = RegimeDefinitions(threshold_method='median')
        growth_data = macro_final["growth_factor"].dropna()
        inflation_data = macro_final["inflation_factor"].dropna()
        regime_def.determine_thresholds(growth_data, inflation_data)
        
        self.regime_def = regime_def
        return regime_def
    
    def _get_variables_from_combination(self, combination: str) -> List[str]:
        """Extract variables from combination name."""
        ALL_VARIABLES = [
            'growth_factor', 'inflation_factor', 'monetary_policy_factor', 'market_volatility_factor'
        ]
        if combination == 'all_4vars':
            return ALL_VARIABLES
        if combination.startswith('2vars_'):
            var_names_str = combination.replace('2vars_', '')
            if 'monetary_policy' in var_names_str:
                var_names_str = var_names_str.replace('monetary_policy', 'monetary_policy_')
            if 'market_volatility' in var_names_str:
                var_names_str = var_names_str.replace('market_volatility', 'market_volatility_')
            
            split_names = var_names_str.split('_')
            var_map = {
                'growth': 'growth_factor', 'inflation': 'inflation_factor',
                'monetary_policy': 'monetary_policy_factor', 'market_volatility': 'market_volatility_factor'
            }
            
            variables = []
            current_var_parts = []
            for part in split_names:
                current_var_parts.append(part)
                full_name_candidate = '_'.join(current_var_parts)
                if full_name_candidate in var_map:
                    variables.append(var_map[full_name_candidate])
                    current_var_parts = []
                elif f"{full_name_candidate}_factor" in ALL_VARIABLES:
                    variables.append(f"{full_name_candidate}_factor")
                    current_var_parts = []
            
            if current_var_parts:
                for part in current_var_parts:
                    if part in var_map:
                        variables.append(var_map[part])
                    elif f"{part}_factor" in ALL_VARIABLES:
                        variables.append(f"{part}_factor")
            
            seen = set()
            result = []
            for v in variables:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
            return result
        raise ValueError(f"Unknown combination format: {combination}")
    
    def train_lasso_regressions(
        self,
        erp: pd.Series,
        macro_df: pd.DataFrame,
        train_end_date: pd.Timestamp,
        base_dir: Path
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], Dict[int, List[str]], Dict[int, List[str]]]:
        """
        Train LASSO regressions for both HMM and 2x2 regimes.
        
        Returns:
        --------
        Tuple of:
        - hmm_coefficients: Dict[regime] -> Dict[var] -> coef
        - two_by_two_coefficients: Dict[regime] -> Dict[var] -> coef
        - hmm_included_vars: Dict[regime] -> List[var]
        - two_by_two_included_vars: Dict[regime] -> List[var]
        """
        # Filter data up to train_end_date for training
        # For initial training (first forecast), use data up to and including train_end_date
        # For subsequent forecasts, use data strictly before train_end_date to avoid lookahead bias
        train_dates = erp.index[erp.index <= train_end_date]
        
        # Check if we have enough data (need at least 24 months for meaningful training)
        if len(train_dates) < 24:
            raise ValueError(f"Insufficient training data: only {len(train_dates)} periods available up to {train_end_date}")
        
        erp_train = erp.reindex(train_dates)
        macro_train = macro_df.reindex(train_dates)
        
        # Align to common dates
        common_dates = erp_train.index.intersection(macro_train.index)
        if len(common_dates) == 0:
            raise ValueError(f"No common dates between ERP and macro data")
        
        erp_train = erp_train.reindex(common_dates)
        macro_train = macro_train.reindex(common_dates)
        
        # Remove rows where ERP is NaN first
        erp_valid_mask = ~erp_train.isna()
        erp_train = erp_train[erp_valid_mask]
        macro_train = macro_train[erp_valid_mask]
        
        if len(erp_train) == 0:
            raise ValueError(f"No valid ERP data: {len(train_dates)} dates, {len(common_dates)} common, {erp_valid_mask.sum()} ERP valid")
        
        # Fill macro variables: backward fill first (to fill early NaNs), then forward fill
        macro_train = macro_train.fillna(method='bfill').fillna(method='ffill')
        
        # Drop columns that are still all NaN (can't be filled)
        macro_train = macro_train.dropna(axis=1, how='all')
        
        if len(macro_train.columns) == 0:
            raise ValueError(f"No valid macro variables after filling")
        
        # Remove rows where macro still has NaN after filling
        # But allow some NaN values - require at least 50% of columns to be valid
        macro_valid_mask = (macro_train.notna().sum(axis=1) / len(macro_train.columns)) >= 0.5
        erp_train = erp_train[macro_valid_mask]
        macro_train = macro_train[macro_valid_mask]
        
        # Final fill: replace remaining NaN with 0 (or column mean)
        macro_train = macro_train.fillna(0.0)
        
        if len(erp_train) == 0:
            raise ValueError(f"No valid training data after macro filling: {len(train_dates)} dates, {len(common_dates)} common, {macro_valid_mask.sum()} macro valid")
        
        if len(erp_train) < 24:
            raise ValueError(f"Insufficient training data: only {len(erp_train)} valid periods (need at least 24)")
        
        # Standardize macro variables
        macro_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(macro_train),
            index=macro_train.index,
            columns=macro_train.columns
        )
        
        # Get HMM regime probabilities
        if self.hmm_model is None:
            self.load_hmm_model(base_dir)
        
        # Get HMM variables for regime detection
        hmm_vars = self._get_variables_from_combination(self.hmm_combination)
        macro_final_path = base_dir / 'data' / 'macro_final' / 'final_macro.csv'
        macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
        macro_final = macro_final.set_index('date').sort_index()
        
        # Align hmm_data with training dates (use same dates as erp_train after filtering)
        hmm_data = macro_final[hmm_vars].reindex(erp_train.index)
        
        # Fill missing values: backward fill first, then forward fill, then fill with 0
        hmm_data = hmm_data.fillna(method='bfill').fillna(method='ffill').fillna(0.0)
        
        if len(hmm_data) == 0:
            raise ValueError(f"No valid HMM data: {len(hmm_data)} rows")
        
        hmm_probs = self.hmm_model.predict_proba(hmm_data.values)
        hmm_probs_df = pd.DataFrame(
            hmm_probs,
            index=hmm_data.index,
            columns=[f'prob_R{i}' for i in range(self.hmm_k)]
        )
        
        # Get 2x2 regime assignments
        if self.regime_def is None:
            self.load_2x2_regime_definitions(base_dir)
        
        # Use the same dates as erp_train/macro_train
        macro_final_train = macro_final.reindex(erp_train.index)
        
        # Forward fill missing values for 2x2 classification
        macro_final_train = macro_final_train.fillna(method='bfill').fillna(method='ffill').fillna(0.0)
        
        if macro_final_train.isna().any().any():
            raise ValueError(f"2x2 macro data still has NaN after filling")
        
        two_by_two_regimes = self.regime_def.classify_dataframe(
            macro_final_train.reset_index(),
            growth_col='growth_factor',
            inflation_col='inflation_factor'
        )
        two_by_two_regimes = pd.Series(two_by_two_regimes.values, index=macro_final_train.index)
        
        # Train LASSO for each HMM regime (weighted by probabilities)
        hmm_coefficients = {}
        hmm_included_vars = {}
        
        for regime_id in range(self.hmm_k):
            weights = hmm_probs_df[f'prob_R{regime_id}'].values
            
            # Only use observations with non-zero weight
            valid_idx = weights > 0.01
            if valid_idx.sum() < 10:  # Need at least 10 observations
                hmm_coefficients[regime_id] = {}
                hmm_included_vars[regime_id] = []
                continue
            
            X_regime = macro_train_scaled.values[valid_idx]
            y_regime = erp_train.values[valid_idx]
            w_regime = weights[valid_idx]
            
            # Use LassoCV to select alpha
            alphas = np.logspace(np.log10(self.alpha_range[0]), np.log10(self.alpha_range[1]), self.n_alphas)
            lasso = LassoCV(alphas=alphas, cv=self.cv_folds, fit_intercept=True, random_state=42)
            lasso.fit(X_regime, y_regime, sample_weight=w_regime)
            
            # Store coefficients and included variables
            coef_dict = {}
            included = []
            for i, var in enumerate(macro_train.columns):
                coef = lasso.coef_[i]
                coef_dict[var] = coef
                if abs(coef) > 1e-6:  # Non-zero coefficient
                    included.append(var)
            
            hmm_coefficients[regime_id] = coef_dict
            hmm_included_vars[regime_id] = included
        
        # Train LASSO for each 2x2 regime (hard assignment)
        two_by_two_coefficients = {}
        two_by_two_included_vars = {}
        
        for regime_id in range(4):
            regime_mask = (two_by_two_regimes == regime_id).values
            if regime_mask.sum() < 10:
                two_by_two_coefficients[regime_id] = {}
                two_by_two_included_vars[regime_id] = []
                continue
            
            X_regime = macro_train_scaled.values[regime_mask]
            y_regime = erp_train.values[regime_mask]
            
            # Use LassoCV to select alpha
            alphas = np.logspace(np.log10(self.alpha_range[0]), np.log10(self.alpha_range[1]), self.n_alphas)
            lasso = LassoCV(alphas=alphas, cv=self.cv_folds, fit_intercept=True, random_state=42)
            lasso.fit(X_regime, y_regime)
            
            # Store coefficients and included variables
            coef_dict = {}
            included = []
            for i, var in enumerate(macro_train.columns):
                coef = lasso.coef_[i]
                coef_dict[var] = coef
                if abs(coef) > 1e-6:
                    included.append(var)
            
            two_by_two_coefficients[regime_id] = coef_dict
            two_by_two_included_vars[regime_id] = included
        
        return hmm_coefficients, two_by_two_coefficients, hmm_included_vars, two_by_two_included_vars
    
    def forecast_rolling(
        self,
        erp: pd.Series,
        macro_df: pd.DataFrame,
        start_date: pd.Timestamp = pd.Timestamp("2002-03-31"),
        base_dir: Optional[Path] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate rolling forecasts with monthly retraining.
        
        Strategy:
        1. Initial training: Use 10 years of data before start_date
        2. Monthly retraining: Rerun both HMM weighted and 2x2 pure regressions each month
        3. Forecasting: Use regime probabilities/thresholds to determine which coefficients to use
        4. If regime not trained yet, use constant forecast (last known forecast or mean)
        
        Returns:
        --------
        Tuple of (hmm_forecasts, two_by_two_forecasts)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent
        
        # Use provided macro_df if available, otherwise load
        if macro_df is not None:
            self.macro_variables = macro_df
        elif self.macro_variables is None:
            self.load_macro_variables(base_dir)
        
        # Load HMM model and 2x2 regime definitions
        if self.hmm_model is None:
            self.load_hmm_model(base_dir)
        if self.regime_def is None:
            self.load_2x2_regime_definitions(base_dir)
        
        # Get macro_final for regime detection
        macro_final_path = base_dir / 'data' / 'macro_final' / 'final_macro.csv'
        macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
        macro_final = macro_final.set_index('date').sort_index()
        
        # Initial training: Use 10 years before start_date
        initial_train_end = start_date
        initial_train_start = initial_train_end - pd.DateOffset(years=10)
        
        print(f"\nInitial training: {initial_train_start.date()} to {initial_train_end.date()}")
        try:
            hmm_coefs, two_by_two_coefs, hmm_included, two_by_two_included = self.train_lasso_regressions(
                erp, self.macro_variables, initial_train_end, base_dir
            )
            
            # Store initial coefficients
            self.hmm_coefficients[start_date] = hmm_coefs
            self.two_by_two_coefficients[start_date] = two_by_two_coefs
            self.hmm_variable_inclusion[start_date] = hmm_included
            self.two_by_two_variable_inclusion[start_date] = two_by_two_included
            
            print(f"✓ Initial training complete: {len([r for r in hmm_coefs.values() if r])} HMM regimes, {len([r for r in two_by_two_coefs.values() if r])} 2x2 regimes trained")
        except Exception as e:
            print(f"Error in initial training: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        # Get forecast dates
        forecast_dates = erp.index[erp.index >= start_date].sort_values()
        
        if len(forecast_dates) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        hmm_forecasts = []
        two_by_two_forecasts = []
        last_retrain_date = start_date
        last_hmm_forecast = 0.0
        last_two_by_two_forecast = 0.0
        
        for forecast_date in forecast_dates:
            # Check if we need to retrain (every month)
            months_diff = (forecast_date.year - last_retrain_date.year) * 12 + (forecast_date.month - last_retrain_date.month)
            should_retrain = months_diff >= 1
            
            if should_retrain:
                try:
                    # Use data up to (but not including) forecast_date
                    train_end = forecast_date - pd.DateOffset(days=1)
                    hmm_coefs, two_by_two_coefs, hmm_included, two_by_two_included = self.train_lasso_regressions(
                        erp, self.macro_variables, train_end, base_dir
                    )
                    
                    # Store coefficients and inclusion for this date
                    self.hmm_coefficients[forecast_date] = hmm_coefs
                    self.two_by_two_coefficients[forecast_date] = two_by_two_coefs
                    self.hmm_variable_inclusion[forecast_date] = hmm_included
                    self.two_by_two_variable_inclusion[forecast_date] = two_by_two_included
                    
                    last_retrain_date = forecast_date
                    print(f"✓ Retrained at {forecast_date.date()}")
                except Exception as e:
                    print(f"Warning: Failed to retrain at {forecast_date}: {e}")
                    # Continue with previous coefficients
            
            # Get most recent coefficients (use last retrain date or forecast_date)
            coeff_date = last_retrain_date if last_retrain_date <= forecast_date else forecast_date
            hmm_coefs_current = self.hmm_coefficients.get(coeff_date, {})
            two_by_two_coefs_current = self.two_by_two_coefficients.get(coeff_date, {})
            
            # Make predictions
            try:
                # Get macro variables for this date
                macro_row = self.macro_variables.reindex([forecast_date])
                if macro_row.isna().any().any():
                    # Use constant forecast
                    hmm_forecasts.append(last_hmm_forecast)
                    two_by_two_forecasts.append(last_two_by_two_forecast)
                    continue
                
                # Standardize using the scaler (fit on training data)
                macro_row_scaled = pd.DataFrame(
                    self.scaler.transform(macro_row),
                    index=macro_row.index,
                    columns=macro_row.columns
                )
                
                # HMM forecast: weighted by regime probabilities
                hmm_vars = self._get_variables_from_combination(self.hmm_combination)
                hmm_data = macro_final[hmm_vars].reindex([forecast_date])
                
                if hmm_data.isna().any().any():
                    hmm_forecast = last_hmm_forecast
                else:
                    hmm_probs = self.hmm_model.predict_proba(hmm_data.values)[0]
                    # Store regime probabilities for plotting
                    self.hmm_regime_probabilities[forecast_date] = hmm_probs
                    
                    hmm_forecast = 0.0
                    regimes_with_coefs = 0
                    for regime_id in range(self.hmm_k):
                        regime_weight = hmm_probs[regime_id]
                        regime_coefs = hmm_coefs_current.get(regime_id, {})
                        
                        # Check if this regime has been trained
                        if not regime_coefs:
                            # Regime not trained yet, skip (or use constant)
                            continue
                        
                        regimes_with_coefs += 1
                        regime_forecast = sum(
                            regime_coefs.get(var, 0.0) * macro_row_scaled[var].iloc[0]
                            for var in macro_row_scaled.columns
                        )
                        hmm_forecast += regime_weight * regime_forecast
                    
                    # If no regimes were trained, use constant forecast
                    if regimes_with_coefs == 0:
                        hmm_forecast = last_hmm_forecast
                
                hmm_forecasts.append(hmm_forecast)
                last_hmm_forecast = hmm_forecast
                
                # 2x2 forecast: use assigned regime
                macro_final_row = macro_final.reindex([forecast_date])
                if macro_final_row.isna().any().any():
                    two_by_two_forecast = last_two_by_two_forecast
                else:
                    regime_assignment = self.regime_def.classify_dataframe(
                        macro_final_row.reset_index(),
                        growth_col='growth_factor',
                        inflation_col='inflation_factor'
                    )[0]
                    
                    regime_coefs = two_by_two_coefs_current.get(regime_assignment, {})
                    
                    # If regime not trained yet, use constant forecast
                    if not regime_coefs:
                        two_by_two_forecast = last_two_by_two_forecast
                    else:
                        two_by_two_forecast = sum(
                            regime_coefs.get(var, 0.0) * macro_row_scaled[var].iloc[0]
                            for var in macro_row_scaled.columns
                        )
                
                two_by_two_forecasts.append(two_by_two_forecast)
                last_two_by_two_forecast = two_by_two_forecast
                    
            except Exception as e:
                print(f"Warning: Failed to predict at {forecast_date}: {e}")
                # Use constant forecast
                hmm_forecasts.append(last_hmm_forecast)
                two_by_two_forecasts.append(last_two_by_two_forecast)
        
        return (
            pd.Series(hmm_forecasts, index=forecast_dates, name="hmm_lasso_forecast"),
            pd.Series(two_by_two_forecasts, index=forecast_dates, name="2x2_lasso_forecast")
        )

