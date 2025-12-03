#!/usr/bin/env python3
"""
Conditional Regression Analysis for All HMM Regimes

This script:
1. Loads all HMM regime assignments (all K values and variable combinations)
2. Loads all macro variables from macro_processed_full
3. Runs conditional regressions for each regime
4. Extracts statistical significance and coefficients
5. Creates visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
# path_utils is in s1_macro_vars directory
SECTION_DIR = SCRIPT_DIR.parents[2]  # Goes up: regressions -> s12_regimeness -> s1_macro_vars
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

try:
    from path_utils import get_project_root
except ImportError:
    # Fallback: construct path manually
    def get_project_root(file_path):
        """Get project root by finding main_project directory."""
        path = Path(file_path).resolve()
        while path.parent != path:
            if path.name == "main_project":
                return path
            path = path.parent
        # Fallback: assume we're in main_project/s1_macro_vars/s12_regimeness/regressions
        return Path(__file__).resolve().parent.parent.parent.parent

# Import HMM model
HMM_REGIMES_DIR = SCRIPT_DIR.parent / 'regimes' / 'HMM_regimes'
if str(HMM_REGIMES_DIR) not in sys.path:
    sys.path.insert(0, str(HMM_REGIMES_DIR))

from hmm_model import HMMRegimeModel


class ConditionalRegressionAnalyzer:
    """
    Analyzes conditional regressions for all HMM regime specifications.
    """
    
    def __init__(self, base_dir: Path, output_dir: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        base_dir : Path
            Base project directory
        output_dir : Path, optional
            Output directory for results
        """
        self.base_dir = Path(base_dir)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = SCRIPT_DIR / 'results'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.macro_data = None
        self.erp_data = None
        self.regime_assignments = {}  # Dict: (combo_name, K) -> regime assignments
        self.regression_results = []  # List of all regression results
        
        # Macro variable directories
        self.macro_dirs = {
            'ec_growth': ['export_price_index_processed 2.csv', 'import_price_index_processed 2.csv',
                         'retail_sales_processed 2.csv', 'tot_business_inventories_processed 2.csv',
                         'unemployment_processed 2.csv'],
            'inflation': ['cpi_processed 2.csv', 'PCE_price_index_processed 2.csv', 
                         'PPI_inflation_processed 2.csv'],
            'mkt_vol': ['vix_processed_monthly.csv', 'nat_fin_condition_indx_processed_monthly.csv',
                       '10y_2y_spread_processed_monthly.csv'],
            'mon_policy': ['10y_treasury_const_maturity_rate_processed 2.csv',
                          'fed_reserve_discount_rate_processed 2.csv',
                          'fedfunds_processed 2.csv', 'm2_real_money_supply_processed 2.csv']
        }
    
    def load_macro_variables(self) -> pd.DataFrame:
        """Load all macro variables from macro_processed_full."""
        print("="*80)
        print("LOADING MACRO VARIABLES")
        print("="*80)
        
        macro_data_dir = self.base_dir / 'data' / 'macro_processed_full'
        all_data = []
        
        for category, files in self.macro_dirs.items():
            print(f"\nLoading {category} variables...")
            category_dir = macro_data_dir / category
            
            for filename in files:
                file_path = category_dir / filename
                
                if not file_path.exists():
                    print(f"  ⚠️  File not found: {filename}")
                    continue
                
                try:
                    df = pd.read_csv(file_path, parse_dates=['date'])
                    
                    # Use 'value' column if available, otherwise use first numeric column
                    if 'value' in df.columns:
                        value_col = 'value'
                    else:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if not numeric_cols:
                            print(f"  ⚠️  No numeric columns in {filename}")
                            continue
                        value_col = numeric_cols[0]
                    
                    # Create variable name from filename
                    var_name = filename.replace('_processed 2.csv', '').replace('_processed_monthly.csv', '').replace('_processed.csv', '')
                    var_name = var_name.replace(' ', '_')
                    
                    # Select date and value
                    df_subset = df[['date', value_col]].copy()
                    df_subset.columns = ['date', var_name]
                    
                    # Convert to monthly if not already
                    df_subset['date'] = pd.to_datetime(df_subset['date'])
                    df_subset = df_subset.set_index('date').sort_index()
                    df_subset = df_subset.resample('ME').last()
                    df_subset = df_subset.reset_index()
                    
                    all_data.append(df_subset)
                    print(f"  ✓ {var_name}: {len(df_subset)} observations")
                    
                except Exception as e:
                    print(f"  ⚠️  Error loading {filename}: {e}")
                    continue
        
        # Merge all macro variables
        print("\nMerging all macro variables...")
        if not all_data:
            raise ValueError("No macro data loaded!")
        
        merged = all_data[0]
        for df in all_data[1:]:
            merged = pd.merge(merged, df, on='date', how='outer', suffixes=('', '_drop'))
            # Remove duplicate columns
            merged = merged.loc[:, ~merged.columns.str.endswith('_drop')]
        
        merged = merged.sort_values('date').reset_index(drop=True)
        merged = merged.dropna(subset=['date'])
        
        print(f"  ✓ Merged dataset: {len(merged)} observations")
        print(f"  ✓ Variables: {len(merged.columns) - 1} macro variables")
        print(f"  ✓ Date range: {merged['date'].min()} to {merged['date'].max()}")
        
        self.macro_data = merged
        return merged
    
    def load_erp(self) -> pd.DataFrame:
        """Load ERP data."""
        print("\n" + "="*80)
        print("LOADING ERP DATA")
        print("="*80)
        
        erp_path = self.base_dir / 'data' / 'macro_processed' / 'equity_risk_pr.csv'
        
        if not erp_path.exists():
            raise FileNotFoundError(f"ERP file not found: {erp_path}")
        
        erp_df = pd.read_csv(erp_path, parse_dates=['date'])
        erp_df['date'] = pd.to_datetime(erp_df['date'])
        
        # Use ERP column
        if 'ERP' in erp_df.columns:
            erp_df = erp_df[['date', 'ERP']].copy()
            erp_df.columns = ['date', 'erp']
        elif 'erp' in erp_df.columns:
            erp_df = erp_df[['date', 'erp']].copy()
        else:
            raise ValueError("ERP column not found")
        
        # Convert to monthly (end of month)
        erp_df = erp_df.set_index('date').sort_index()
        erp_df = erp_df.resample('ME').last()
        erp_df = erp_df.reset_index()
        
        print(f"  ✓ Loaded {len(erp_df)} ERP observations")
        print(f"  ✓ Date range: {erp_df['date'].min()} to {erp_df['date'].max()}")
        
        self.erp_data = erp_df
        return erp_df
    
    def load_hmm_regimes(self) -> Dict[Tuple[str, int], pd.DataFrame]:
        """
        Load HMM regime assignments for all combinations and K values.
        
        Returns:
        --------
        Dict mapping (combination_name, K) to regime assignments DataFrame
        """
        print("\n" + "="*80)
        print("LOADING HMM REGIME ASSIGNMENTS")
        print("="*80)
        
        # Load systematic results to get all combinations
        results_file = (self.base_dir / 's1_macro_vars' / 's12_regimeness' / 'regimes' / 
                       'HMM_regimes' / 'results_systematic' / 'all_model_results.csv')
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        results_df = pd.read_csv(results_file)
        
        # Get unique combinations and K values
        combinations = results_df[['combination', 'variables']].drop_duplicates()
        
        print(f"\nFound {len(combinations)} variable combinations")
        print(f"K values: {sorted(results_df['K'].unique())}")
        
        # Load macro data for regime detection
        macro_final_path = self.base_dir / 'data' / 'macro_final' / 'final_macro.csv'
        macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
        
        # For each combination, fit HMM for each K and get regime assignments
        regime_assignments = {}
        
        for _, combo_row in combinations.iterrows():
            combo_name = combo_row['combination']
            variables_str = combo_row['variables']
            variables = [v.strip() for v in variables_str.split(',')]
            
            # Get K values for this combination (limit to K <= 6)
            combo_results = results_df[results_df['combination'] == combo_name]
            k_values = sorted([k for k in combo_results['K'].unique() if k <= 6])
            
            print(f"\nProcessing {combo_name}...")
            print(f"  Variables: {', '.join(variables)}")
            
            for k in k_values:
                try:
                    # Prepare features
                    feature_data = macro_final[['date'] + variables].copy()
                    feature_data = feature_data.dropna()
                    
                    if len(feature_data) == 0:
                        print(f"    ⚠️  K={k}: No data")
                        continue
                    
                    # Fit HMM model
                    scaler = StandardScaler()
                    features = scaler.fit_transform(feature_data[variables].values)
                    
                    model = HMMRegimeModel(
                        n_regimes=k,
                        variables=variables,
                        random_state=42
                    )
                    model.scaler = scaler
                    model.fit(features, n_init=5)
                    
                    # Get regime probabilities (soft assignments)
                    regime_probs = model.predict_proba(features)  # Shape: (n_samples, n_regimes)
                    
                    # Create DataFrame with regime probabilities
                    regime_df = pd.DataFrame({
                        'date': feature_data['date'].values
                    })
                    
                    # Add probability columns for each regime
                    for regime_idx in range(k):
                        regime_df[f'prob_R{regime_idx}'] = regime_probs[:, regime_idx]
                    
                    # Convert to monthly (end of month)
                    regime_df['date'] = pd.to_datetime(regime_df['date'])
                    regime_df = regime_df.set_index('date').sort_index()
                    regime_df = regime_df.resample('ME').last()
                    regime_df = regime_df.reset_index()
                    
                    key = (combo_name, k)
                    regime_assignments[key] = regime_df
                    
                    print(f"    ✓ K={k}: {len(regime_df)} observations, {k} regimes with probabilities")
                    
                except Exception as e:
                    print(f"    ⚠️  K={k}: Error - {e}")
                    continue
        
        self.regime_assignments = regime_assignments
        print(f"\n✓ Loaded {len(regime_assignments)} regime specifications")
        
        return regime_assignments
    
    def run_conditional_regressions(self) -> pd.DataFrame:
        """
        Run conditional regressions for each regime using weighted regressions.
        
        Conceptual Framework:
        ---------------------
        Each regime r has its own "pure" betas (β_r) that we're trying to estimate,
        even though we never observe pure regimes. At each time t, we have regime
        probabilities (weights): w_{r,t} for r = 1, ..., K.
        
        The observed relationship at time t is a weighted combination:
            y_t = X_t * (w_{1,t} * β_1 + w_{2,t} * β_2 + ... + w_{K,t} * β_K) + ε_t
        
        To estimate the pure betas β_r, we run separate weighted regressions for each
        regime, where observations are weighted by their probability of being in that
        regime. This allows us to "isolate" each regime's pure betas:
        
        - Regime r regression: y_t = X_t * β_r + ε_t, weighted by w_{r,t}
        - Higher weights emphasize observations where regime r is more likely
        - This gives us estimates of β_r and their statistical significance
        
        Returns:
        --------
        DataFrame with all regression results
        """
        print("\n" + "="*80)
        print("RUNNING CONDITIONAL REGRESSIONS (WEIGHTED BY REGIME PROBABILITIES)")
        print("="*80)
        
        if self.macro_data is None:
            self.load_macro_variables()
        if self.erp_data is None:
            self.load_erp()
        if not self.regime_assignments:
            self.load_hmm_regimes()
        
        # Combine ERP and macro data
        combined = pd.merge(self.erp_data, self.macro_data, on='date', how='inner')
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"\nCombined dataset: {len(combined)} observations")
        print(f"Macro variables: {len(combined.columns) - 2}")  # -2 for date and erp
        
        # Get macro variable names (exclude date and erp)
        macro_vars = [col for col in combined.columns if col not in ['date', 'erp']]
        
        all_results = []
        
        # For each regime specification
        for (combo_name, k), regime_df in self.regime_assignments.items():
            print(f"\n{combo_name}, K={k}...")
            
            # Merge regime probabilities
            prob_cols = [col for col in regime_df.columns if col.startswith('prob_R')]
            merge_cols = ['date'] + prob_cols
            combined_with_regime = pd.merge(
                combined,
                regime_df[merge_cols],
                on='date',
                how='inner'
            )
            
            if len(combined_with_regime) == 0:
                print(f"  ⚠️  No overlapping dates")
                continue
            
            # Get all regimes (0 to k-1)
            all_regimes = list(range(k))
            print(f"  Running weighted regressions for {len(all_regimes)} regimes...")
            print(f"  (Estimating pure betas β_r for each regime using probability weights)")
            
            # For each regime, run weighted regression using regime probabilities as weights
            # This estimates the "pure" beta_r for regime r by emphasizing observations
            # where regime r is more likely (higher probability weight)
            for regime in all_regimes:
                prob_col = f'prob_R{regime}'
                
                if prob_col not in combined_with_regime.columns:
                    print(f"    Regime {regime}: No probability column found")
                    continue
                
                # Get regime probabilities as weights
                # These weights (w_{r,t}) determine how much each observation contributes
                # to estimating the pure beta_r for this regime
                weights = combined_with_regime[prob_col].values
                
                # Use all observations with their weights (soft assignment approach)
                # Observations with higher probability of being in this regime get more weight
                regime_data = combined_with_regime.copy()
                
                # Prepare data
                X = regime_data[macro_vars].values
                y = regime_data['erp'].values
                
                # Remove columns with all NaN
                valid_cols = ~np.isnan(X).all(axis=0)
                X = X[:, valid_cols]
                valid_var_names = [macro_vars[i] for i in range(len(macro_vars)) if valid_cols[i]]
                
                if X.shape[1] == 0:
                    print(f"    Regime {regime}: No valid variables")
                    continue
                
                # Remove rows with any NaN in X or y, or zero weight
                valid_rows = (~np.isnan(X).any(axis=1) & ~np.isnan(y) & 
                             ~np.isnan(weights) & (weights > 1e-6))
                X = X[valid_rows]
                y = y[valid_rows]
                weights_valid = weights[valid_rows]
                
                if len(y) < 10:
                    print(f"    Regime {regime}: Only {len(y)} valid observations (skipping)")
                    continue
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit weighted regression to estimate pure beta_r for this regime
                # The weights (regime probabilities) emphasize observations where this
                # regime is more likely, allowing us to estimate the regime-specific betas
                try:
                    model = LinearRegression()
                    model.fit(X_scaled, y, sample_weight=weights_valid)
                    
                    # Get predictions
                    y_pred = model.predict(X_scaled)
                    
                    # Calculate statistics (weighted)
                    n = len(y)
                    p = X_scaled.shape[1]
                    residuals = y - y_pred
                    
                    # Weighted MSE
                    weighted_mse = np.average(residuals**2, weights=weights_valid)
                    rmse = np.sqrt(weighted_mse)
                    
                    # Weighted R-squared
                    y_mean_weighted = np.average(y, weights=weights_valid)
                    ss_res_weighted = np.average(residuals**2, weights=weights_valid)
                    ss_tot_weighted = np.average((y - y_mean_weighted)**2, weights=weights_valid)
                    r_squared = 1 - (ss_res_weighted / ss_tot_weighted) if ss_tot_weighted > 0 else 0
                    
                    # Effective sample size (sum of weights)
                    n_eff = np.sum(weights_valid)
                    
                    # Average weight (for table display)
                    avg_weight = np.mean(weights_valid)
                    
                    # Standard errors and t-stats (using effective sample size)
                    if n_eff > p + 1:
                        # Weighted variance
                        var_residual = ss_res_weighted * n_eff / (n_eff - p - 1)
                        # Weighted covariance matrix
                        X_weighted = X_scaled * np.sqrt(weights_valid[:, np.newaxis])
                        var_coef = var_residual * np.linalg.inv(X_weighted.T @ X_weighted)
                        se_coef = np.sqrt(np.diag(var_coef))
                        t_stats = model.coef_ / se_coef
                        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_eff - p - 1))
                    else:
                        se_coef = np.full(p, np.nan)
                        t_stats = np.full(p, np.nan)
                        p_values = np.full(p, np.nan)
                    
                    # Store results for each variable
                    # These coefficients are estimates of the "pure" beta_r for this regime
                    for i, var_name in enumerate(valid_var_names):
                        all_results.append({
                            'combination': combo_name,
                            'K': k,
                            'regime': regime,
                            'variable': var_name,
                            'coefficient': model.coef_[i],  # Pure beta_r estimate
                            't_statistic': t_stats[i],
                            'p_value': p_values[i],
                            'n_observations': n,
                            'n_effective': n_eff,  # Effective sample size (sum of weights)
                            'avg_weight': avg_weight,  # Average regime probability
                            'r_squared': r_squared,
                            'rmse': rmse
                        })
                    
                    print(f"    Regime {regime}: n={n}, n_eff={n_eff:.1f}, avg_weight={avg_weight:.3f}, R²={r_squared:.3f}, RMSE={rmse:.4f}")
                    print(f"      → Estimated pure β_{regime} coefficients (weighted by regime probabilities)")
                    
                except Exception as e:
                    print(f"    Regime {regime}: Error - {e}")
                    continue
        
        results_df = pd.DataFrame(all_results)
        self.regression_results = results_df
        
        print(f"\n✓ Completed {len(results_df)} regressions")
        print(f"✓ Unique combinations: {results_df['combination'].nunique()}")
        print(f"✓ Unique regimes: {results_df['regime'].nunique()}")
        print(f"✓ Unique variables: {results_df['variable'].nunique()}")
        
        return results_df
    
    def save_results(self):
        """Save regression results to CSV."""
        if self.regression_results is None or len(self.regression_results) == 0:
            print("No results to save")
            return
        
        output_file = self.output_dir / 'conditional_regression_results_all.csv'
        self.regression_results.to_csv(output_file, index=False)
        print(f"\n✓ Saved results to: {output_file}")
        
        # Create summary by significance
        significant = self.regression_results[
            (self.regression_results['p_value'] < 0.05) & 
            (self.regression_results['p_value'].notna())
        ]
        
        summary_file = self.output_dir / 'significant_variables_summary.csv'
        significant_summary = significant.groupby(['combination', 'K', 'regime', 'variable']).agg({
            'coefficient': 'mean',
            't_statistic': 'mean',
            'p_value': 'mean',
            'n_observations': 'mean'
        }).reset_index()
        significant_summary.to_csv(summary_file, index=False)
        print(f"✓ Saved significant variables summary to: {summary_file}")


def main():
    """Main function."""
    base_dir = get_project_root(__file__)
    
    analyzer = ConditionalRegressionAnalyzer(base_dir)
    
    # Load data
    analyzer.load_macro_variables()
    analyzer.load_erp()
    analyzer.load_hmm_regimes()
    
    # Run regressions
    analyzer.run_conditional_regressions()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()

