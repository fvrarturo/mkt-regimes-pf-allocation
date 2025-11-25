"""
Equity Risk Premium (ERP) Macro Variable Relevance Analysis

This script analyzes which macro variables are most relevant for Equity Risk Premium
(ERP = SP500 - 3m yield) conditional on different market regimes.

IMPORTANT: Regime Probabilities
The analysis accounts for regime uncertainty by using regime probabilities from the HMM model.
Instead of hard regime assignments, observations are weighted by their probability of being
in each regime. This means:
- Observations with mixed regime probabilities contribute to multiple regimes with weights
- Correlations, regressions, and feature importance are calculated using weighted methods
- This provides more robust results that account for regime uncertainty

The analysis includes:
1. Calculation of ERP from SP500 and 3m yield
2. Loading regime assignments and probabilities from HMM model
3. Loading all available macro variables
4. Regime-conditional analysis (using weighted methods):
   - Weighted correlation analysis based on regime probabilities
   - Weighted regression analysis
   - Weighted feature importance (using Random Forest with sample weights)
   - Statistical significance testing with effective sample sizes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class ERPMacroRelevanceAnalyzer:
    """
    Analyzes macro variable relevance for Equity Risk Premium conditional on regimes.
    """
    
    def __init__(
        self,
        regime_assignments_path: Path,
        macro_data_dir: Path,
        sp500_path: Path,
        yield_3m_path: Path,
        output_dir: Path
    ):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        regime_assignments_path : Path
            Path to regime_assignments.csv from HMM model
        macro_data_dir : Path
            Path to macro_processed directory
        sp500_path : Path
            Path to sp500_processed.csv
        yield_3m_path : Path
            Path to 3m_yield_processed.csv
        output_dir : Path
            Directory to save results
        """
        self.regime_assignments_path = Path(regime_assignments_path)
        self.macro_data_dir = Path(macro_data_dir)
        self.sp500_path = Path(sp500_path)
        self.yield_3m_path = Path(yield_3m_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.regime_data = None
        self.regime_prob_cols = {}  # Mapping from regime number to probability column name
        self.erp_data = None
        self.macro_data = {}
        self.combined_data = None
        self.results = {}
        
    def load_regime_assignments(self) -> pd.DataFrame:
        """
        Load regime assignments from HMM model.
        
        Includes both hard assignments and soft probabilities for each regime.
        """
        print("Loading regime assignments...")
        df = pd.read_csv(self.regime_assignments_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Extract probability columns
        prob_cols = [col for col in df.columns if col.startswith('prob_R')]
        print(f"  Found {len(prob_cols)} regime probability columns")
        
        # Create a mapping from regime number to probability column
        self.regime_prob_cols = {}
        for i in range(4):  # Assuming 4 regimes
            # Find column matching this regime
            for col in prob_cols:
                if f'R{i}' in col:
                    self.regime_prob_cols[i] = col
                    break
        
        self.regime_data = df
        print(f"  Loaded {len(df)} regime assignments")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Regimes: {sorted(df['regime'].unique())}")
        print(f"  Using regime probabilities: {list(self.regime_prob_cols.keys())}")
        return df
    
    def calculate_erp(self) -> pd.DataFrame:
        """
        Calculate Equity Risk Premium = SP500 Return - 3m yield (risk-free rate).
        
        ERP is the excess return of equities over the risk-free rate.
        This is calculated as: ERP = SP500_monthly_return - 3m_yield_monthly_rate
        
        Returns monthly ERP series.
        """
        print("\nCalculating Equity Risk Premium...")
        
        # Load SP500 data
        sp500 = pd.read_csv(self.sp500_path)
        sp500['date'] = pd.to_datetime(sp500['date'])
        sp500 = sp500[['date', 'value']].copy()
        sp500.columns = ['date', 'sp500']
        sp500 = sp500.set_index('date')
        
        # Check frequency of SP500 data
        date_diffs = sp500.index.to_series().diff().dropna()
        most_common_freq = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else None
        print(f"  SP500 data frequency: {most_common_freq}")
        
        # Convert to monthly (end of month) if not already monthly
        if date_diffs.min() < pd.Timedelta(days=25):  # Likely daily or weekly
            print("  Converting SP500 to monthly frequency...")
            sp500_monthly = sp500.resample('M').last()
        else:
            print("  SP500 already appears to be monthly")
            sp500_monthly = sp500.copy()
        
        # Calculate SP500 monthly returns
        sp500_monthly['sp500_return'] = sp500_monthly['sp500'].pct_change()
        
        # Load 3m yield data
        yield_3m = pd.read_csv(self.yield_3m_path)
        yield_3m['date'] = pd.to_datetime(yield_3m['date'])
        yield_3m = yield_3m[['date', 'value']].copy()
        yield_3m.columns = ['date', 'yield_3m']
        yield_3m = yield_3m.set_index('date')
        
        # Check frequency of 3m yield data
        yield_date_diffs = yield_3m.index.to_series().diff().dropna()
        yield_most_common_freq = yield_date_diffs.mode()[0] if len(yield_date_diffs.mode()) > 0 else None
        print(f"  3m yield data frequency: {yield_most_common_freq}")
        
        # Convert to monthly (end of month) if not already monthly
        if yield_date_diffs.min() < pd.Timedelta(days=25):  # Likely daily
            print("  Converting 3m yield to monthly frequency...")
            yield_3m_monthly = yield_3m.resample('M').last()
        else:
            print("  3m yield already appears to be monthly")
            yield_3m_monthly = yield_3m.copy()
        
        # Convert yield from percentage points to decimal (if needed)
        # Assuming yield is in percentage points (e.g., 5.0 means 5%)
        # For monthly return calculation, we need monthly rate
        # If yield is annualized, convert to monthly: monthly_rate = annual_rate / 12
        # But typically yields are already in the right units for comparison
        # We'll use the yield directly as the risk-free rate
        
        # Merge SP500 returns and 3m yield
        erp = pd.merge(
            sp500_monthly[['sp500', 'sp500_return']],
            yield_3m_monthly[['yield_3m']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Calculate ERP = SP500 Return - Risk-free rate (3m yield)
        # Note: If yield is annualized percentage (e.g., 5.0 = 5%), convert to monthly decimal
        # monthly_yield = yield_3m / 100 / 12
        # But for consistency with returns (which are already in decimal), we'll use yield/100
        # Actually, let's check: if yield is 5.0, does it mean 5% or 0.05?
        # Typically yields are stored as percentages (5.0 = 5%), so we divide by 100
        # For monthly comparison, we might need to annualize or use monthly equivalent
        # For now, assuming yield is annualized percentage, convert to monthly decimal
        erp['yield_3m_monthly'] = erp['yield_3m'] / 100 / 12  # Convert annual % to monthly decimal
        
        # Calculate ERP as excess return
        erp['erp'] = erp['sp500_return'] - erp['yield_3m_monthly']
        
        # Also keep the annualized version for reference
        erp['erp_annualized'] = erp['erp'] * 12
        
        # Calculate forward ERP return (for predictive analysis)
        erp['erp_return_forward'] = erp['erp'].shift(-1)  # Next period ERP
        
        # Drop NaN (first row will be NaN due to pct_change)
        erp = erp.dropna()
        
        self.erp_data = erp
        print(f"  ERP calculated: {len(erp)} observations")
        print(f"  Date range: {erp.index.min()} to {erp.index.max()}")
        print(f"  ERP stats (monthly): mean={erp['erp'].mean():.4f}, std={erp['erp'].std():.4f}")
        print(f"  ERP stats (annualized): mean={erp['erp_annualized'].mean():.2f}%, std={erp['erp_annualized'].std():.2f}%")
        print(f"  SP500 return stats: mean={erp['sp500_return'].mean():.4f}, std={erp['sp500_return'].std():.4f}")
        print(f"  3m yield stats: mean={erp['yield_3m'].mean():.2f}%, std={erp['yield_3m'].std():.2f}%")
        
        return erp
    
    def load_macro_variables(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available macro variables from macro_processed directory.
        
        Returns dictionary mapping variable names to DataFrames.
        """
        print("\nLoading macro variables...")
        
        macro_vars = {}
        
        # Define subdirectories and their variable mappings
        # NOTE: Excluding 'other' folder variables (sp500, 3m_yield, 2y_yield, 10y_yield, bofa_highyield_spread)
        # as they are either used for ERP calculation or should be excluded from macro analysis
        subdirs = {
            'ec_growth': ['gdp', 'real_gdp', 'unemployment', 'industrial_production', 
                         'retail_sales', 'tot_business_inventories', 
                         'export_price_index', 'import_price_index'],
            'inflation': ['cpi', 'PCE_price_index', 'PPI_inflation'],
            'mkt_vol': ['vix', 'nasdaq_vol_indx', '3month_vol_index_sp500', 
                       'nat_fin_condition_indx', '10y_2y_spread'],
            'mon_policy': ['fedfunds', 'fed_reserve_discount_rate', 
                          '10y_treasury_const_maturity_rate', 'm2_real_money_supply']
            # 'other' folder excluded: sp500, 3m_yield (used for ERP), 2y_yield, 10y_yield, bofa_highyield_spread
        }
        
        for subdir, var_names in subdirs.items():
            subdir_path = self.macro_data_dir / subdir
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
        """
        Combine ERP, regime, and macro data into a single DataFrame.
        
        Returns combined DataFrame with all variables aligned by date.
        """
        print("\nCombining data...")
        
        # Start with ERP data
        combined = self.erp_data.copy()
        
        # Add regime data (including probabilities)
        regime_cols = ['regime', 'regime_name']
        if hasattr(self, 'regime_prob_cols'):
            regime_cols.extend(list(self.regime_prob_cols.values()))
        
        combined = pd.merge(
            combined,
            self.regime_data[regime_cols],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Add macro variables
        for var_name, var_df in self.macro_data.items():
            combined = pd.merge(
                combined,
                var_df,
                left_index=True,
                right_index=True,
                how='left'
            )
        
        # Forward fill macro variables (use most recent available value)
        macro_cols = list(self.macro_data.keys())
        combined[macro_cols] = combined[macro_cols].ffill()
        
        # Drop rows with missing ERP or regime
        combined = combined.dropna(subset=['erp', 'regime'])
        
        # Drop rows where all macro variables are NaN
        combined = combined.dropna(subset=macro_cols, how='all')
        
        self.combined_data = combined
        print(f"  Combined dataset: {len(combined)} observations")
        print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
        print(f"  Variables: {len(combined.columns)}")
        print(f"  Regime distribution:")
        print(combined['regime'].value_counts().sort_index())
        
        return combined
    
    def analyze_correlations_by_regime(self, use_probabilities: bool = True) -> pd.DataFrame:
        """
        Calculate correlations between ERP and macro variables for each regime.
        
        Uses weighted correlations based on regime probabilities if use_probabilities=True.
        Otherwise uses hard regime assignments only.
        
        Parameters:
        -----------
        use_probabilities : bool
            If True, use weighted correlations based on regime probabilities.
            If False, use only hard regime assignments.
        
        Returns DataFrame with correlations by regime.
        """
        print("\nAnalyzing correlations by regime...")
        if use_probabilities and hasattr(self, 'regime_prob_cols'):
            print("  Using regime probabilities for weighted analysis")
        else:
            print("  Using hard regime assignments only")
        
        macro_cols = list(self.macro_data.keys())
        regimes = sorted(self.combined_data['regime'].unique())
        
        results = []
        
        for regime in regimes:
            if use_probabilities and hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
                # Use weighted analysis: include all observations weighted by regime probability
                prob_col = self.regime_prob_cols[regime]
                regime_data = self.combined_data.copy()
                # Filter to observations with non-zero probability for this regime
                regime_data = regime_data[regime_data[prob_col] > 0.01]  # Minimum 1% probability
                weights = regime_data[prob_col].values
                regime_name = self.combined_data[
                    self.combined_data['regime'] == regime
                ]['regime_name'].iloc[0] if len(self.combined_data[
                    self.combined_data['regime'] == regime
                ]) > 0 else f"Regime {regime}"
            else:
                # Use hard assignments only
                regime_data = self.combined_data[self.combined_data['regime'] == regime]
                weights = None
                if len(regime_data) == 0:
                    continue
                regime_name = regime_data['regime_name'].iloc[0]
            
            if len(regime_data) < 10:  # Need minimum observations
                continue
            
            for var_name in macro_cols:
                if var_name not in regime_data.columns:
                    continue
                
                # Prepare data
                corr_data = regime_data[['erp', var_name]].dropna()
                
                if len(corr_data) < 5:
                    continue
                
                # Calculate correlation
                if use_probabilities and weights is not None:
                    # Weighted correlation
                    # Get weights for non-null observations
                    corr_weights = regime_data.loc[corr_data.index, prob_col].values
                    
                    # Calculate weighted correlation
                    x = corr_data['erp'].values
                    y = corr_data[var_name].values
                    w = corr_weights / corr_weights.sum()  # Normalize weights
                    
                    # Weighted means
                    x_mean = np.average(x, weights=w)
                    y_mean = np.average(y, weights=w)
                    
                    # Weighted covariance and variances
                    cov_xy = np.average((x - x_mean) * (y - y_mean), weights=w)
                    var_x = np.average((x - x_mean) ** 2, weights=w)
                    var_y = np.average((y - y_mean) ** 2, weights=w)
                    
                    if var_x > 0 and var_y > 0:
                        corr = cov_xy / np.sqrt(var_x * var_y)
                    else:
                        corr = 0.0
                    
                    # Approximate p-value using effective sample size
                    n_eff = (np.sum(w) ** 2) / np.sum(w ** 2)  # Effective sample size
                    if n_eff > 2 and abs(corr) < 1:
                        t_stat = corr * np.sqrt((n_eff - 2) / (1 - corr ** 2))
                        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff - 2))
                    else:
                        pvalue = 1.0
                    
                    n_obs = len(corr_data)
                    weighted_n_obs = n_eff
                else:
                    # Standard unweighted correlation
                    corr, pvalue = stats.pearsonr(corr_data['erp'], corr_data[var_name])
                    n_obs = len(corr_data)
                    weighted_n_obs = n_obs
                
                results.append({
                    'regime': regime,
                    'regime_name': regime_name,
                    'variable': var_name,
                    'correlation': corr,
                    'pvalue': pvalue,
                    'n_observations': n_obs,
                    'weighted_n_observations': weighted_n_obs if use_probabilities else n_obs,
                    'abs_correlation': abs(corr),
                    'uses_probabilities': use_probabilities
                })
        
        corr_df = pd.DataFrame(results)
        self.results['correlations'] = corr_df
        
        print(f"  Calculated {len(corr_df)} correlations")
        return corr_df
    
    def analyze_regressions_by_regime(self, use_probabilities: bool = True) -> Dict:
        """
        Run regressions of ERP on macro variables for each regime.
        
        Uses weighted regressions based on regime probabilities if use_probabilities=True.
        
        Parameters:
        -----------
        use_probabilities : bool
            If True, use weighted regressions based on regime probabilities.
            If False, use only hard regime assignments.
        
        Returns dictionary with regression results.
        """
        print("\nAnalyzing regressions by regime...")
        if use_probabilities and hasattr(self, 'regime_prob_cols'):
            print("  Using regime probabilities for weighted regressions")
        else:
            print("  Using hard regime assignments only")
        
        macro_cols = list(self.macro_data.keys())
        regimes = sorted(self.combined_data['regime'].unique())
        
        regression_results = {}
        
        for regime in regimes:
            if use_probabilities and hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
                # Use weighted analysis
                prob_col = self.regime_prob_cols[regime]
                regime_data = self.combined_data.copy()
                regime_data = regime_data[regime_data[prob_col] > 0.01]
                weights = regime_data[prob_col].values
                regime_name = self.combined_data[
                    self.combined_data['regime'] == regime
                ]['regime_name'].iloc[0] if len(self.combined_data[
                    self.combined_data['regime'] == regime
                ]) > 0 else f"Regime {regime}"
            else:
                # Use hard assignments
                regime_data = self.combined_data[self.combined_data['regime'] == regime].copy()
                weights = None
                if len(regime_data) == 0:
                    continue
                regime_name = regime_data['regime_name'].iloc[0]
            
            if len(regime_data) < 10:
                continue
            
            # Prepare data
            X = regime_data[macro_cols].dropna(axis=1, how='all')
            y = regime_data['erp']
            
            # Align X and y
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            # Drop rows with any NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
            # Get weights for regression (if using probabilities)
            if use_probabilities and weights is not None:
                reg_weights = regime_data.loc[X.index, prob_col].values
            else:
                reg_weights = None
            
            # Run regression for each variable individually
            var_results = []
            
            for var_name in X.columns:
                if X[var_name].isna().all():
                    continue
                
                X_var = X_scaled[[var_name]]
                X_var_clean = X_var.dropna()
                y_var = y.loc[X_var_clean.index]
                
                if len(X_var_clean) < 5:
                    continue
                
                try:
                    # Get weights for this variable's observations
                    if use_probabilities and reg_weights is not None:
                        var_weights = regime_data.loc[X_var_clean.index, prob_col].values
                        var_weights = var_weights / var_weights.sum() * len(var_weights)  # Normalize
                    else:
                        var_weights = None
                    
                    model = LinearRegression()
                    if var_weights is not None:
                        # Weighted regression
                        model.fit(X_var_clean, y_var, sample_weight=var_weights)
                    else:
                        model.fit(X_var_clean, y_var)
                    
                    # Calculate R-squared
                    y_pred = model.predict(X_var_clean)
                    r2 = model.score(X_var_clean, y_var)
                    
                    # Calculate t-statistic and p-value
                    n = len(X_var_clean)
                    k = 1  # one predictor
                    
                    if var_weights is not None:
                        # Weighted MSE
                        mse = np.average((y_var - y_pred) ** 2, weights=var_weights)
                        # Weighted variance of X
                        x_mean = np.average(X_var_clean[var_name], weights=var_weights)
                        var_x = np.average((X_var_clean[var_name] - x_mean) ** 2, weights=var_weights)
                        # Effective sample size
                        n_eff = (np.sum(var_weights) ** 2) / np.sum(var_weights ** 2)
                        n = n_eff
                    else:
                        mse = np.mean((y_var - y_pred) ** 2)
                        var_x = np.var(X_var_clean[var_name])
                    
                    var_coef = mse / (var_x * n) if var_x > 0 and n > 0 else 0
                    se_coef = np.sqrt(var_coef) if var_coef > 0 else 0
                    t_stat = model.coef_[0] / se_coef if se_coef > 0 else 0
                    pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1)) if n > k + 1 else 1.0
                    
                    var_results.append({
                        'variable': var_name,
                        'coefficient': model.coef_[0],
                        'r_squared': r2,
                        't_statistic': t_stat,
                        'pvalue': pvalue,
                        'n_observations': len(X_var_clean),
                        'effective_n': n if var_weights is not None else len(X_var_clean),
                        'uses_probabilities': use_probabilities
                    })
                except Exception as e:
                    print(f"    Warning: Regression failed for {var_name} in regime {regime}: {e}")
            
            regression_results[regime] = {
                'regime_name': regime_name,
                'results': pd.DataFrame(var_results),
                'n_observations': len(X)
            }
        
        self.results['regressions'] = regression_results
        print(f"  Completed regressions for {len(regression_results)} regimes")
        return regression_results
    
    def analyze_feature_importance_by_regime(self, use_probabilities: bool = True) -> Dict:
        """
        Use Random Forest to calculate feature importance for ERP prediction by regime.
        
        Uses sample weights based on regime probabilities if use_probabilities=True.
        
        Parameters:
        -----------
        use_probabilities : bool
            If True, use sample weights based on regime probabilities.
            If False, use only hard regime assignments.
        
        Returns dictionary with feature importance results.
        """
        print("\nAnalyzing feature importance by regime...")
        if use_probabilities and hasattr(self, 'regime_prob_cols'):
            print("  Using regime probabilities for weighted Random Forest")
        else:
            print("  Using hard regime assignments only")
        
        macro_cols = list(self.macro_data.keys())
        regimes = sorted(self.combined_data['regime'].unique())
        
        importance_results = {}
        
        for regime in regimes:
            if use_probabilities and hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
                # Use weighted analysis
                prob_col = self.regime_prob_cols[regime]
                regime_data = self.combined_data.copy()
                regime_data = regime_data[regime_data[prob_col] > 0.01]
                weights = regime_data[prob_col].values
                regime_name = self.combined_data[
                    self.combined_data['regime'] == regime
                ]['regime_name'].iloc[0] if len(self.combined_data[
                    self.combined_data['regime'] == regime
                ]) > 0 else f"Regime {regime}"
            else:
                # Use hard assignments
                regime_data = self.combined_data[self.combined_data['regime'] == regime].copy()
                weights = None
                if len(regime_data) == 0:
                    continue
                regime_name = regime_data['regime_name'].iloc[0]
            
            if len(regime_data) < 20:  # Need more observations for RF
                continue
            
            # Prepare data
            X = regime_data[macro_cols].dropna(axis=1, how='all')
            y = regime_data['erp']
            
            # Align X and y
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            # Drop rows with any NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 20:
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
            try:
                # Get sample weights if using probabilities
                if use_probabilities and weights is not None:
                    sample_weights = regime_data.loc[X.index, prob_col].values
                else:
                    sample_weights = None
                
                # Fit Random Forest
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                if sample_weights is not None:
                    rf.fit(X_scaled, y, sample_weight=sample_weights)
                else:
                    rf.fit(X_scaled, y)
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'variable': X_scaled.columns,
                    'importance': rf.feature_importances_,
                    'n_observations': len(X)
                }).sort_values('importance', ascending=False)
                
                # Calculate R-squared
                y_pred = rf.predict(X_scaled)
                r2 = rf.score(X_scaled, y)
                
                importance_results[regime] = {
                    'regime_name': regime_name,
                    'importance': importance_df,
                    'r_squared': r2,
                    'n_observations': len(X),
                    'uses_probabilities': use_probabilities
                }
                
                print(f"  Regime {regime} ({regime_name}): R² = {r2:.3f}, n = {len(X)}")
            except Exception as e:
                print(f"  Warning: RF failed for regime {regime}: {e}")
        
        self.results['feature_importance'] = importance_results
        return importance_results
    
    def create_summary_report(self) -> pd.DataFrame:
        """
        Create a summary report ranking macro variables by relevance for each regime.
        
        Combines correlation, regression, and feature importance metrics.
        """
        print("\nCreating summary report...")
        
        macro_cols = list(self.macro_data.keys())
        regimes = sorted(self.combined_data['regime'].unique())
        
        summary_results = []
        
        for regime in regimes:
            regime_name = self.combined_data[
                self.combined_data['regime'] == regime
            ]['regime_name'].iloc[0]
            
            # Get correlations
            corr_data = self.results['correlations']
            regime_corr = corr_data[corr_data['regime'] == regime].set_index('variable')
            
            # Get regressions
            if regime in self.results['regressions']:
                reg_data = self.results['regressions'][regime]['results'].set_index('variable')
            else:
                reg_data = pd.DataFrame()
            
            # Get feature importance
            if regime in self.results['feature_importance']:
                imp_data = self.results['feature_importance'][regime]['importance'].set_index('variable')
            else:
                imp_data = pd.DataFrame()
            
            # Combine metrics for each variable
            for var_name in macro_cols:
                metrics = {
                    'regime': regime,
                    'regime_name': regime_name,
                    'variable': var_name
                }
                
                # Correlation metrics
                if var_name in regime_corr.index:
                    metrics['correlation'] = regime_corr.loc[var_name, 'correlation']
                    metrics['correlation_pvalue'] = regime_corr.loc[var_name, 'pvalue']
                    metrics['abs_correlation'] = regime_corr.loc[var_name, 'abs_correlation']
                else:
                    metrics['correlation'] = np.nan
                    metrics['correlation_pvalue'] = np.nan
                    metrics['abs_correlation'] = np.nan
                
                # Regression metrics
                if var_name in reg_data.index:
                    metrics['regression_coef'] = reg_data.loc[var_name, 'coefficient']
                    metrics['regression_r2'] = reg_data.loc[var_name, 'r_squared']
                    metrics['regression_pvalue'] = reg_data.loc[var_name, 'pvalue']
                else:
                    metrics['regression_coef'] = np.nan
                    metrics['regression_r2'] = np.nan
                    metrics['regression_pvalue'] = np.nan
                
                # Feature importance
                if var_name in imp_data.index:
                    metrics['rf_importance'] = imp_data.loc[var_name, 'importance']
                else:
                    metrics['rf_importance'] = np.nan
                
                summary_results.append(metrics)
        
        summary_df = pd.DataFrame(summary_results)
        
        # Calculate composite relevance score
        # Normalize each metric to 0-1 scale and combine
        for regime in regimes:
            regime_mask = summary_df['regime'] == regime
            
            # Normalize abs_correlation
            if summary_df.loc[regime_mask, 'abs_correlation'].notna().any():
                max_abs_corr = summary_df.loc[regime_mask, 'abs_correlation'].max()
                if max_abs_corr > 0:
                    summary_df.loc[regime_mask, 'norm_abs_corr'] = (
                        summary_df.loc[regime_mask, 'abs_correlation'] / max_abs_corr
                    )
                else:
                    summary_df.loc[regime_mask, 'norm_abs_corr'] = 0
            else:
                summary_df.loc[regime_mask, 'norm_abs_corr'] = 0
            
            # Normalize regression R²
            if summary_df.loc[regime_mask, 'regression_r2'].notna().any():
                max_r2 = summary_df.loc[regime_mask, 'regression_r2'].max()
                if max_r2 > 0:
                    summary_df.loc[regime_mask, 'norm_r2'] = (
                        summary_df.loc[regime_mask, 'regression_r2'] / max_r2
                    )
                else:
                    summary_df.loc[regime_mask, 'norm_r2'] = 0
            else:
                summary_df.loc[regime_mask, 'norm_r2'] = 0
            
            # Normalize RF importance
            if summary_df.loc[regime_mask, 'rf_importance'].notna().any():
                max_imp = summary_df.loc[regime_mask, 'rf_importance'].max()
                if max_imp > 0:
                    summary_df.loc[regime_mask, 'norm_rf_imp'] = (
                        summary_df.loc[regime_mask, 'rf_importance'] / max_imp
                    )
                else:
                    summary_df.loc[regime_mask, 'norm_rf_imp'] = 0
            else:
                summary_df.loc[regime_mask, 'norm_rf_imp'] = 0
        
        # Calculate composite score (weighted average)
        summary_df['relevance_score'] = (
            0.4 * summary_df['norm_abs_corr'].fillna(0) +
            0.3 * summary_df['norm_r2'].fillna(0) +
            0.3 * summary_df['norm_rf_imp'].fillna(0)
        )
        
        self.results['summary'] = summary_df
        return summary_df
    
    def visualize_results(self):
        """Create visualizations of the analysis results."""
        print("\nCreating visualizations...")
        
        # 1. Top variables by regime (heatmap)
        summary = self.results['summary']
        
        # Pivot for heatmap
        top_n = 10  # Top 10 variables per regime
        heatmap_data = []
        
        for regime in sorted(summary['regime'].unique()):
            regime_summary = summary[summary['regime'] == regime].nlargest(top_n, 'relevance_score')
            for _, row in regime_summary.iterrows():
                heatmap_data.append({
                    'regime': f"R{int(row['regime'])}\n{row['regime_name'].split('/')[0].strip()}",
                    'variable': row['variable'],
                    'relevance_score': row['relevance_score']
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        if len(heatmap_df) > 0:
            pivot = heatmap_df.pivot(index='variable', columns='regime', values='relevance_score')
            
            plt.figure(figsize=(14, max(8, len(pivot) * 0.4)))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Relevance Score'})
            plt.title('Top Macro Variables by Regime - Relevance Scores', fontsize=16, fontweight='bold')
            plt.xlabel('Regime', fontsize=12)
            plt.ylabel('Macro Variable', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'relevance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap by regime
        corr_data = self.results['correlations']
        if len(corr_data) > 0:
            corr_pivot = corr_data.pivot(index='variable', columns='regime', values='correlation')
            
            plt.figure(figsize=(12, max(8, len(corr_pivot) * 0.3)))
            sns.heatmap(
                corr_pivot, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r', 
                center=0,
                cbar_kws={'label': 'Correlation with ERP'}
            )
            plt.title('ERP-Macro Variable Correlations by Regime', fontsize=16, fontweight='bold')
            plt.xlabel('Regime', fontsize=12)
            plt.ylabel('Macro Variable', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Top variables per regime (bar charts)
        n_regimes = len(summary['regime'].unique())
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, regime in enumerate(sorted(summary['regime'].unique())):
            if idx >= len(axes):
                break
            
            regime_summary = summary[summary['regime'] == regime].nlargest(10, 'relevance_score')
            regime_name = regime_summary['regime_name'].iloc[0]
            
            ax = axes[idx]
            regime_summary.plot(
                x='variable',
                y='relevance_score',
                kind='barh',
                ax=ax,
                color='steelblue'
            )
            ax.set_title(f'Regime {int(regime)}: {regime_name}\nTop 10 Variables', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Relevance Score', fontsize=10)
            ax.set_ylabel('')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_variables_by_regime.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Visualizations saved")
    
    def save_results(self):
        """Save all results to CSV and JSON files."""
        print("\nSaving results...")
        
        # Save summary
        if 'summary' in self.results:
            summary_path = self.output_dir / 'erp_macro_relevance_summary.csv'
            self.results['summary'].to_csv(summary_path, index=False)
            print(f"  Saved summary: {summary_path}")
        
        # Save correlations
        if 'correlations' in self.results:
            corr_path = self.output_dir / 'erp_correlations_by_regime.csv'
            self.results['correlations'].to_csv(corr_path, index=False)
            print(f"  Saved correlations: {corr_path}")
        
        # Save regressions
        if 'regressions' in self.results:
            reg_results = []
            for regime, reg_data in self.results['regressions'].items():
                reg_df = reg_data['results'].copy()
                reg_df['regime'] = regime
                reg_df['regime_name'] = reg_data['regime_name']
                reg_results.append(reg_df)
            
            if reg_results:
                reg_df_all = pd.concat(reg_results, ignore_index=True)
                reg_path = self.output_dir / 'erp_regressions_by_regime.csv'
                reg_df_all.to_csv(reg_path, index=False)
                print(f"  Saved regressions: {reg_path}")
        
        # Save feature importance
        if 'feature_importance' in self.results:
            imp_results = []
            for regime, imp_data in self.results['feature_importance'].items():
                imp_df = imp_data['importance'].copy()
                imp_df['regime'] = regime
                imp_df['regime_name'] = imp_data['regime_name']
                imp_df['model_r2'] = imp_data['r_squared']
                imp_results.append(imp_df)
            
            if imp_results:
                imp_df_all = pd.concat(imp_results, ignore_index=True)
                imp_path = self.output_dir / 'erp_feature_importance_by_regime.csv'
                imp_df_all.to_csv(imp_path, index=False)
                print(f"  Saved feature importance: {imp_path}")
        
        # Save combined data sample
        if self.combined_data is not None:
            sample_path = self.output_dir / 'combined_data_sample.csv'
            self.combined_data.head(100).to_csv(sample_path)
            print(f"  Saved data sample: {sample_path}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 80)
        print("ERP MACRO VARIABLE RELEVANCE ANALYSIS")
        print("=" * 80)
        
        # Load data
        self.load_regime_assignments()
        self.calculate_erp()
        self.load_macro_variables()
        self.combine_data()
        
        # Run analyses (using regime probabilities by default)
        use_probs = hasattr(self, 'regime_prob_cols') and len(self.regime_prob_cols) > 0
        self.analyze_correlations_by_regime(use_probabilities=use_probs)
        self.analyze_regressions_by_regime(use_probabilities=use_probs)
        self.analyze_feature_importance_by_regime(use_probabilities=use_probs)
        self.create_summary_report()
        
        # Visualize and save
        self.visualize_results()
        self.save_results()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir}")
        print("\nKey findings:")
        
        # Print top variables per regime
        summary = self.results['summary']
        for regime in sorted(summary['regime'].unique()):
            regime_summary = summary[summary['regime'] == regime].nlargest(5, 'relevance_score')
            regime_name = regime_summary['regime_name'].iloc[0]
            print(f"\n  Regime {int(regime)}: {regime_name}")
            for _, row in regime_summary.iterrows():
                print(f"    {row['variable']:30s} (score: {row['relevance_score']:.3f})")


def main():
    """Main function to run the analysis."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    regime_assignments_path = project_root / 'test_regimes_matrix' / 'results' / 'regime_assignments.csv'
    macro_data_dir = project_root / 'data' / 'macro_processed'
    sp500_path = project_root / 'data' / 'macro_processed' / 'other' / 'sp500_processed.csv'
    yield_3m_path = project_root / 'data' / 'macro_processed' / 'other' / '3m_yield_processed.csv'
    output_dir = Path(__file__).parent / 'results'
    
    # Initialize analyzer
    analyzer = ERPMacroRelevanceAnalyzer(
        regime_assignments_path=regime_assignments_path,
        macro_data_dir=macro_data_dir,
        sp500_path=sp500_path,
        yield_3m_path=yield_3m_path,
        output_dir=output_dir
    )
    
    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

