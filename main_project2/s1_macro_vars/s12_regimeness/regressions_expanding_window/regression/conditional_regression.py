"""
Regime-Conditional Regression Analysis: Core Regression Logic

This module contains the RegimeConditionalRegressor class that performs
regime-conditional regression analysis to identify which macro variables
predict ERP in different economic regimes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class RegimeConditionalRegressor:
    """
    Performs regime-conditional regression analysis to identify which macro variables
    predict ERP in different economic regimes.
    """
    
    def __init__(
        self,
        data_dir: Path,
        regime_model: str = 'hmm_optimal',  # 'hmm_optimal' or '2x2'
        output_dir: Optional[Path] = None,
        use_expanding_window: bool = False
    ):
        """
        Initialize the regressor.
        
        Parameters:
        -----------
        data_dir : Path
            Path to main_project2 directory
        regime_model : str
            Which regime model to use: 'hmm_optimal' or '2x2'
        output_dir : Path, optional
            Output directory for results
        use_expanding_window : bool
            Whether to use expanding window regime assignments (no look-ahead bias)
        """
        self.data_dir = Path(data_dir)
        self.regime_model = regime_model
        self.use_expanding_window = use_expanding_window
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            if use_expanding_window:
                self.output_dir = self.data_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_expanding_window' / 'results' / regime_model
            else:
                self.output_dir = self.data_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_full_sample' / 'results' / regime_model
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.erp_data = None
        self.regime_data = None
        self.macro_data = {}
        self.combined_data = None
        
        # Results storage
        self.regression_results = None
        self.coefficient_tables = {}
        self.statistical_tests = None
        
    def load_erp(self) -> pd.DataFrame:
        """Load ERP data."""
        print("Loading ERP data...")
        erp_path = self.data_dir / 'data' / 'macro_processed' / 'equity_risk_pr.csv'
        
        if not erp_path.exists():
            raise FileNotFoundError(f"ERP file not found: {erp_path}")
        
        erp_df = pd.read_csv(erp_path, parse_dates=['date'])
        erp_df = erp_df.set_index('date').sort_index()
        
        # Use ERP column (capitalized)
        if 'ERP' in erp_df.columns:
            erp_df = erp_df[['ERP']].copy()
            erp_df.columns = ['erp']
        elif 'erp' in erp_df.columns:
            erp_df = erp_df[['erp']].copy()
        else:
            raise ValueError("ERP column not found in equity_risk_pr.csv")
        
        # Convert to monthly (end of month)
        erp_monthly = erp_df.resample('M').last()
        erp_monthly = erp_monthly.dropna()
        
        self.erp_data = erp_monthly
        print(f"  Loaded {len(erp_monthly)} ERP observations")
        print(f"  Date range: {erp_monthly.index.min()} to {erp_monthly.index.max()}")
        return erp_monthly
    
    def load_regime_assignments(self, use_expanding_window: bool = False) -> pd.DataFrame:
        """Load regime assignments based on model type."""
        print(f"\nLoading regime assignments for {self.regime_model}...")
        
        if use_expanding_window:
            # Use expanding window regime assignments (no look-ahead bias)
            regime_path = self.data_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_expanding_window' / 'results' / 'regime_assignments' / self.regime_model / 'regime_assignments.csv'
            print("  Using expanding window regime assignments (no look-ahead bias)")
        else:
            # Use full-sample regime assignments
            if self.regime_model == 'hmm_optimal':
                regime_path = self.data_dir / 's1_macro_vars' / 's12_regimeness' / 'results_full_sample' / 'hmm_regime_assignments' / 'regime_assignments.csv'
            elif self.regime_model == '2x2':
                regime_path = self.data_dir / 's1_macro_vars' / 's12_regimeness' / 'results_full_sample' / '2x2_regime_assignments' / 'regime_assignments.csv'
            else:
                raise ValueError(f"Unknown regime model: {self.regime_model}")
            print("  Using full-sample regime assignments")
        
        if not regime_path.exists():
            raise FileNotFoundError(f"Regime file not found: {regime_path}")
        
        regime_df = pd.read_csv(regime_path, parse_dates=['date'])
        regime_df = regime_df.set_index('date').sort_index()
        
        # Identify probability columns
        prob_cols = [col for col in regime_df.columns if 'prob' in col.lower() or col.startswith('prob_')]
        if prob_cols:
            self.regime_prob_cols = {int(col.split('_')[-1].replace('R', '')): col for col in prob_cols if 'R' in col}
        
        self.regime_data = regime_df
        print(f"  Loaded {len(regime_df)} regime observations")
        print(f"  Regimes: {sorted(regime_df['regime'].unique())}")
        if hasattr(self, 'regime_prob_cols'):
            print(f"  Probability columns: {list(self.regime_prob_cols.values())}")
        
        return regime_df
    
    def load_macro_variables(self) -> Dict[str, pd.DataFrame]:
        """Load all available macro variables from macro_processed directory."""
        print("\nLoading macro variables...")
        
        macro_vars = {}
        # Try main_project2 first, then fall back to main_project
        macro_data_dir = self.data_dir / 'data' / 'macro_processed'
        if not macro_data_dir.exists() or len(list(macro_data_dir.glob('*/'))) == 0:
            # Try main_project
            main_project_dir = self.data_dir.parent / 'main_project' / 'data' / 'macro_processed'
            if main_project_dir.exists():
                macro_data_dir = main_project_dir
                print(f"  Using macro data from: {main_project_dir}")
        
        # Define subdirectories and their variable mappings
        subdirs = {
            'ec_growth': ['gdp', 'real_gdp', 'unemployment', 'industrial_production', 
                         'retail_sales', 'tot_business_inventories'],
            'inflation': ['cpi', 'PCE_price_index', 'PPI_inflation'],
            'mkt_vol': ['nat_fin_condition_indx', '10y_2y_spread'],
            'mon_policy': ['fedfunds', 'fed_reserve_discount_rate', 
                          '10y_treasury_const_maturity_rate', 'm2_real_money_supply']
        }
        
        for subdir, var_names in subdirs.items():
            subdir_path = macro_data_dir / subdir
            if not subdir_path.exists():
                continue
            
            for var_name in var_names:
                file_path = subdir_path / f"{var_name}_processed.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        if 'date' not in df.columns:
                            continue
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        
                        # Use 'value' column if available, otherwise use first numeric column
                        if 'value' in df.columns:
                            var_df = df[['value']].copy()
                        else:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                var_df = df[[numeric_cols[0]]].copy()
                            else:
                                continue
                        
                        var_df.columns = [var_name]
                        
                        # Convert to monthly (end of month)
                        var_df_monthly = var_df.resample('M').last()
                        
                        macro_vars[var_name] = var_df_monthly
                        print(f"  Loaded {var_name}: {len(var_df_monthly)} observations")
                    except Exception as e:
                        print(f"  Warning: Could not load {var_name}: {e}")
        
        self.macro_data = macro_vars
        print(f"\n  Total macro variables loaded: {len(macro_vars)}")
        return macro_vars
    
    def combine_data(self) -> pd.DataFrame:
        """Combine ERP, regime, and macro data into a single DataFrame."""
        print("\nCombining data...")
        
        # Start with ERP data
        combined = self.erp_data.copy()
        
        # Add regime data - align dates properly
        regime_cols = ['regime', 'regime_name']
        if hasattr(self, 'regime_prob_cols'):
            regime_cols.extend(list(self.regime_prob_cols.values()))
        
        # Ensure both indices are datetime and aligned to month-end
        regime_data_aligned = self.regime_data[regime_cols].copy()
        if isinstance(regime_data_aligned.index, pd.DatetimeIndex):
            regime_data_aligned.index = regime_data_aligned.index.to_period('M').to_timestamp('M')
        
        combined = pd.merge(
            combined,
            regime_data_aligned,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Add macro variables - align dates
        for var_name, var_df in self.macro_data.items():
            var_df_aligned = var_df.copy()
            if isinstance(var_df_aligned.index, pd.DatetimeIndex):
                var_df_aligned.index = var_df_aligned.index.to_period('M').to_timestamp('M')
            
            combined = pd.merge(
                combined,
                var_df_aligned,
                left_index=True,
                right_index=True,
                how='left'
            )
        
        # Create lagged versions of macro variables (t-1 predicts t)
        print("\nCreating lagged macro variables...")
        macro_cols = list(self.macro_data.keys())
        for var in macro_cols:
            if var in combined.columns:
                combined[f'{var}_lag1'] = combined[var].shift(1)
        
        # Drop rows with missing ERP or regime
        combined = combined.dropna(subset=['erp', 'regime'])
        
        self.combined_data = combined
        print(f"  Combined data: {len(combined)} observations")
        print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
        
        return combined
    
    def create_forward_erp(self, horizon: int) -> pd.Series:
        """
        Create forward-looking ERP for forecast horizon h.
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon in months
        
        Returns:
        --------
        pd.Series
            Forward ERP series
        """
        return self.combined_data['erp'].shift(-horizon)
    
    def run_regime_regression(
        self,
        regime: int,
        variable: str,
        horizon: int = 1,
        use_probabilities: bool = True
    ) -> Optional[Dict]:
        """
        Run regression for a single variable in a single regime.
        
        Parameters:
        -----------
        regime : int
            Regime number
        variable : str
            Macro variable name (without _lag1 suffix)
        horizon : int
            Forecast horizon in months
        use_probabilities : bool
            Whether to use regime probabilities as weights
        
        Returns:
        --------
        Dict with regression results or None if insufficient data
        """
        lag_var = f'{variable}_lag1'
        
        if lag_var not in self.combined_data.columns:
            return None
        
        # Filter data for this regime
        if use_probabilities and hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
            prob_col = self.regime_prob_cols[regime]
            regime_mask = self.combined_data[prob_col] > 0.01
            regime_data = self.combined_data[regime_mask].copy()
            prob_col_name = prob_col
        else:
            regime_mask = self.combined_data['regime'] == regime
            regime_data = self.combined_data[regime_mask].copy()
            prob_col_name = None
        
        if len(regime_data) < 10:
            return None
        
        # Prepare data
        y_forward = self.create_forward_erp(horizon)
        reg_data = pd.DataFrame({
            'y': y_forward,
            'X': self.combined_data[lag_var]
        }, index=self.combined_data.index)
        
        # Filter to regime
        reg_data = reg_data[regime_mask].dropna()
        
        if len(reg_data) < 10:
            return None
        
        X = reg_data[['X']].values
        y = reg_data['y'].values
        
        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get weights if using probabilities (Mahalanobis distance-based)
        # Probabilities come from Mahalanobis distance to regime clusters
        # Higher probability = observation is closer to regime cluster center
        if use_probabilities and prob_col_name is not None:
            reg_weights = regime_data.loc[reg_data.index, prob_col_name].values
            
            if len(reg_weights) == len(reg_data):
                # Normalize weights so they sum to number of observations
                # This ensures proper statistical inference in weighted regression
                reg_weights = reg_weights / reg_weights.sum() * len(reg_weights)
            else:
                reg_weights = None
        else:
            reg_weights = None
        
        # Fit regression
        model = LinearRegression()
        if reg_weights is not None and len(reg_weights) == len(X_scaled):
            model.fit(X_scaled, y, sample_weight=reg_weights)
        else:
            model.fit(X_scaled, y)
        
        # Calculate statistics
        y_pred = model.predict(X_scaled)
        r2 = model.score(X_scaled, y)
        
        n = len(X_scaled)
        k = 1
        mse = np.mean((y - y_pred) ** 2)
        
        # Calculate standard error of coefficient
        if n > k + 1:
            var_coef = mse / np.sum((X_scaled.flatten() - X_scaled.mean()) ** 2) if np.sum((X_scaled.flatten() - X_scaled.mean()) ** 2) > 0 else 0
            se_coef = np.sqrt(var_coef) if var_coef > 0 else 0
            t_stat = model.coef_[0] / se_coef if se_coef > 0 else 0
            pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
        else:
            se_coef = 0
            t_stat = 0
            pvalue = 1.0
        
        return {
            'regime': regime,
            'variable': variable,
            'horizon': horizon,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': r2,
            't_statistic': t_stat,
            'p_value': pvalue,
            'se_coef': se_coef,
            'n_observations': n
        }
    
    def run_all_regressions(
        self,
        horizons: List[int] = [1, 3, 6, 12],
        use_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Run regressions for all variables, regimes, and horizons.
        
        Parameters:
        -----------
        horizons : List[int]
            List of forecast horizons
        use_probabilities : bool
            Whether to use regime probabilities as weights
        
        Returns:
        --------
        pd.DataFrame with all regression results
        """
        print("\n" + "="*80)
        print("Running regime-conditional regressions...")
        print("="*80)
        
        all_results = []
        regimes = sorted(self.combined_data['regime'].unique())
        macro_cols = list(self.macro_data.keys())
        
        total_runs = len(regimes) * len(macro_cols) * len(horizons)
        run_count = 0
        
        for regime in regimes:
            regime_name = self.combined_data[self.combined_data['regime'] == regime]['regime_name'].iloc[0] if 'regime_name' in self.combined_data.columns else f"Regime {regime}"
            print(f"\nRegime {regime}: {regime_name}")
            
            for variable in macro_cols:
                for horizon in horizons:
                    run_count += 1
                    if run_count % 10 == 0:
                        print(f"  Progress: {run_count}/{total_runs}")
                    
                    result = self.run_regime_regression(
                        regime=regime,
                        variable=variable,
                        horizon=horizon,
                        use_probabilities=use_probabilities
                    )
                    
                    if result:
                        result['regime_name'] = regime_name
                        all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        self.regression_results = results_df
        
        print(f"\nCompleted {len(results_df)} regressions")
        return results_df
    
    def create_coefficient_tables(self) -> Dict[str, pd.DataFrame]:
        """Create coefficient tables by horizon."""
        if self.regression_results is None or len(self.regression_results) == 0:
            raise ValueError("No regression results available. Run run_all_regressions() first.")
        
        tables = {}
        horizons = sorted(self.regression_results['horizon'].unique())
        
        for horizon in horizons:
            horizon_data = self.regression_results[self.regression_results['horizon'] == horizon].copy()
            
            # Pivot table: variables as rows, regimes as columns
            pivot = horizon_data.pivot_table(
                index='variable',
                columns='regime',
                values='coefficient',
                aggfunc='first'
            )
            
            # Add t-statistics and p-values
            t_stats = horizon_data.pivot_table(
                index='variable',
                columns='regime',
                values='t_statistic',
                aggfunc='first'
            )
            
            p_values = horizon_data.pivot_table(
                index='variable',
                columns='regime',
                values='p_value',
                aggfunc='first'
            )
            
            # Create formatted table with significance stars
            formatted_table = pivot.copy()
            for var in formatted_table.index:
                for regime in formatted_table.columns:
                    coef = pivot.loc[var, regime]
                    pval = p_values.loc[var, regime] if var in p_values.index and regime in p_values.columns else 1.0
                    
                    if pd.notna(coef) and pd.notna(pval):
                        if pval < 0.01:
                            star = '***'
                        elif pval < 0.05:
                            star = '**'
                        elif pval < 0.10:
                            star = '*'
                        else:
                            star = ''
                        formatted_table.loc[var, regime] = f"{coef:.4f}{star}"
            
            tables[f'h{horizon}_coefficients'] = pivot
            tables[f'h{horizon}_tstats'] = t_stats
            tables[f'h{horizon}_pvalues'] = p_values
            tables[f'h{horizon}_formatted'] = formatted_table
        
        self.coefficient_tables = tables
        return tables
    
    def test_coefficient_differences(self) -> pd.DataFrame:
        """
        Test if coefficients differ significantly across regimes using Wald tests.
        
        Returns:
        --------
        pd.DataFrame with test results
        """
        print("\nTesting coefficient differences across regimes...")
        
        test_results = []
        horizons = sorted(self.regression_results['horizon'].unique())
        variables = sorted(self.regression_results['variable'].unique())
        regimes = sorted(self.regime_data['regime'].unique())
        
        for horizon in horizons:
            for variable in variables:
                var_data = self.regression_results[
                    (self.regression_results['variable'] == variable) &
                    (self.regression_results['horizon'] == horizon)
                ].copy()
                
                if len(var_data) < 2:
                    continue
                
                # Get coefficients and standard errors for each regime
                coefs = {}
                ses = {}
                for _, row in var_data.iterrows():
                    regime = row['regime']
                    coefs[regime] = row['coefficient']
                    ses[regime] = row['se_coef']
                
                # Test all pairwise differences
                regime_pairs = [(r1, r2) for r1 in regimes for r2 in regimes if r1 < r2]
                
                for r1, r2 in regime_pairs:
                    if r1 not in coefs or r2 not in coefs:
                        continue
                    
                    # Wald test for difference
                    diff = coefs[r1] - coefs[r2]
                    se_diff = np.sqrt(ses[r1]**2 + ses[r2]**2) if ses[r1] > 0 and ses[r2] > 0 else np.nan
                    
                    if se_diff > 0:
                        z_stat = diff / se_diff
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        
                        test_results.append({
                            'horizon': horizon,
                            'variable': variable,
                            'regime1': r1,
                            'regime2': r2,
                            'coef1': coefs[r1],
                            'coef2': coefs[r2],
                            'difference': diff,
                            'se_difference': se_diff,
                            'z_statistic': z_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
        
        test_df = pd.DataFrame(test_results)
        self.statistical_tests = test_df
        
        if len(test_df) > 0:
            n_sig = test_df['significant'].sum()
            print(f"  Found {n_sig} significant coefficient differences out of {len(test_df)} tests")
        
        return test_df
    
    def save_results(self):
        """Save all results to CSV files."""
        print("\nSaving results...")
        
        # Save main regression results
        if self.regression_results is not None and len(self.regression_results) > 0:
            self.regression_results.to_csv(
                self.output_dir / 'regression_results_all.csv',
                index=False
            )
            print(f"  Saved: regression_results_all.csv")
        
        # Save coefficient tables
        for table_name, table_df in self.coefficient_tables.items():
            table_df.to_csv(
                self.output_dir / f'{table_name}.csv'
            )
            print(f"  Saved: {table_name}.csv")
        
        # Save statistical tests
        if self.statistical_tests is not None and len(self.statistical_tests) > 0:
            self.statistical_tests.to_csv(
                self.output_dir / 'coefficient_difference_tests.csv',
                index=False
            )
            print(f"  Saved: coefficient_difference_tests.csv")

