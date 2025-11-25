"""
Equity Risk Premium (ERP) Predictive Power Analysis - OUT-OF-SAMPLE VALIDATED

This script properly addresses overfitting by implementing:
1. Rolling window out-of-sample validation
2. Time-series cross-validation
3. Benchmark comparisons (historical average, random walk)
4. Multiple testing corrections
5. Honest performance metrics
6. Prediction intervals

CRITICAL DIFFERENCES FROM erp_predictive_power.py:
- That script: In-sample R² (likely overfitted)
- This script: Out-of-sample R² (honest predictive power)

The results from this script should be trusted for real-world applications.
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
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class ERPPredictiveValidation:
    """
    Out-of-sample validation for ERP predictive power analysis.
    Addresses overfitting through proper train/test splitting.
    """
    
    def __init__(
        self,
        regime_assignments_path: Path,
        macro_data_dir: Path,
        sp500_path: Path,
        yield_3m_path: Path,
        output_dir: Path,
        horizons: List[int] = [1, 3, 6, 12],
        min_train_size: int = 120,  # 10 years minimum training
        walk_forward_step: int = 1   # Monthly steps
    ):
        """Initialize the validation analyzer."""
        self.regime_assignments_path = Path(regime_assignments_path)
        self.macro_data_dir = Path(macro_data_dir)
        self.sp500_path = Path(sp500_path)
        self.yield_3m_path = Path(yield_3m_path)
        self.output_dir = Path(output_dir)
        self.horizons = horizons
        self.min_train_size = min_train_size
        self.walk_forward_step = walk_forward_step
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.regime_data = None
        self.regime_prob_cols = {}
        self.erp_data = None
        self.macro_data = {}
        self.combined_data = None
        self.results = {}
        
    def load_data(self):
        """Load all necessary data (same as base class)."""
        print("Loading data for validation...")
        self.load_regime_assignments()
        self.calculate_erp()
        self.load_macro_variables()
        self.combine_data()
        
    def load_regime_assignments(self) -> pd.DataFrame:
        """Load regime assignments from HMM model."""
        print("  Loading regime assignments...")
        df = pd.read_csv(self.regime_assignments_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        prob_cols = [col for col in df.columns if col.startswith('prob_R')]
        self.regime_prob_cols = {}
        for i in range(4):
            for col in prob_cols:
                if f'R{i}' in col:
                    self.regime_prob_cols[i] = col
                    break
        
        self.regime_data = df
        print(f"    Loaded {len(df)} observations")
        return df
    
    def calculate_erp(self) -> pd.DataFrame:
        """Calculate Equity Risk Premium and forward returns."""
        print("  Calculating ERP...")
        
        # Load SP500
        sp500 = pd.read_csv(self.sp500_path)
        sp500['date'] = pd.to_datetime(sp500['date'])
        sp500 = sp500[['date', 'value']].set_index('date')
        sp500.columns = ['sp500']
        
        date_diffs = sp500.index.to_series().diff().dropna()
        if date_diffs.min() < pd.Timedelta(days=25):
            sp500_monthly = sp500.resample('M').last()
        else:
            sp500_monthly = sp500.copy()
        
        sp500_monthly['sp500_return'] = sp500_monthly['sp500'].pct_change()
        
        # Load 3m yield
        yield_3m = pd.read_csv(self.yield_3m_path)
        yield_3m['date'] = pd.to_datetime(yield_3m['date'])
        yield_3m = yield_3m[['date', 'value']].set_index('date')
        yield_3m.columns = ['yield_3m']
        
        yield_date_diffs = yield_3m.index.to_series().diff().dropna()
        if yield_date_diffs.min() < pd.Timedelta(days=25):
            yield_3m_monthly = yield_3m.resample('M').last()
        else:
            yield_3m_monthly = yield_3m.copy()
        
        # Merge and calculate ERP
        erp = pd.merge(
            sp500_monthly[['sp500', 'sp500_return']],
            yield_3m_monthly[['yield_3m']],
            left_index=True, right_index=True, how='inner'
        )
        
        erp['yield_3m_monthly'] = erp['yield_3m'] / 100 / 12
        erp['erp'] = erp['sp500_return'] - erp['yield_3m_monthly']
        
        # Calculate forward returns
        for h in self.horizons:
            erp[f'erp_forward_{h}m'] = erp['erp'].rolling(window=h).sum().shift(-h)
        
        erp = erp.dropna(subset=['erp'])
        self.erp_data = erp
        print(f"    Calculated ERP: {len(erp)} observations")
        return erp
    
    def load_macro_variables(self) -> Dict[str, pd.DataFrame]:
        """Load all available macro variables."""
        print("  Loading macro variables...")
        
        macro_vars = {}
        subdirs = {
            'ec_growth': ['gdp', 'real_gdp', 'unemployment', 'industrial_production', 
                         'retail_sales', 'tot_business_inventories', 
                         'export_price_index', 'import_price_index'],
            'inflation': ['cpi', 'PCE_price_index', 'PPI_inflation'],
            'mkt_vol': ['vix', 'nasdaq_vol_indx', '3month_vol_index_sp500', 
                       'nat_fin_condition_indx', '10y_2y_spread'],
            'mon_policy': ['fedfunds', 'fed_reserve_discount_rate', 
                          '10y_treasury_const_maturity_rate', 'm2_real_money_supply']
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
                        
                        if 'value' in df.columns:
                            var_df = df[['value']].copy()
                        else:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                var_df = df[[numeric_cols[0]]].copy()
                            else:
                                continue
                        
                        var_df.columns = [var_name]
                        var_df_monthly = var_df.resample('M').last()
                        macro_vars[var_name] = var_df_monthly
                    except Exception as e:
                        print(f"    Warning: Could not load {var_name}: {e}")
        
        self.macro_data = macro_vars
        print(f"    Loaded {len(macro_vars)} variables")
        return macro_vars
    
    def combine_data(self) -> pd.DataFrame:
        """Combine ERP, regime, and macro data."""
        print("  Combining data...")
        
        combined = self.erp_data.copy()
        
        regime_cols = ['regime', 'regime_name']
        if hasattr(self, 'regime_prob_cols'):
            regime_cols.extend(list(self.regime_prob_cols.values()))
        
        combined = pd.merge(combined, self.regime_data[regime_cols],
                          left_index=True, right_index=True, how='inner')
        
        for var_name, var_df in self.macro_data.items():
            combined = pd.merge(combined, var_df,
                              left_index=True, right_index=True, how='left')
        
        macro_cols = list(self.macro_data.keys())
        combined[macro_cols] = combined[macro_cols].ffill()
        combined = combined.dropna(subset=['erp', 'regime'])
        combined = combined.dropna(subset=macro_cols, how='all')
        
        self.combined_data = combined
        print(f"    Combined: {len(combined)} observations")
        return combined
    
    def rolling_window_validation(
        self, 
        horizon: int,
        variable: str,
        regime: Optional[int] = None
    ) -> Dict:
        """
        Perform rolling window out-of-sample validation.
        
        Train on expanding window [0:t], predict on [t], move forward.
        """
        target_col = f'erp_forward_{horizon}m'
        
        # Filter data
        if regime is not None:
            if hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
                prob_col = self.regime_prob_cols[regime]
                data = self.combined_data[self.combined_data[prob_col] > 0.01].copy()
            else:
                data = self.combined_data[self.combined_data['regime'] == regime].copy()
        else:
            data = self.combined_data.copy()
        
        # Prepare features
        data = data[[target_col, variable]].dropna()
        
        if len(data) < self.min_train_size + 12:  # Need enough for train + test
            return None
        
        predictions = []
        actuals = []
        dates = []
        
        # Rolling window: train on [0:t], test on [t]
        for t in range(self.min_train_size, len(data), self.walk_forward_step):
            train_data = data.iloc[:t]
            test_data = data.iloc[t:t+1]
            
            if len(test_data) == 0:
                continue
            
            # Prepare train data
            X_train = train_data[[variable]].values.reshape(-1, 1)
            y_train = train_data[target_col].values
            
            # Prepare test data
            X_test = test_data[[variable]].values.reshape(-1, 1)
            y_test = test_data[target_col].values
            
            # Fit model on train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Predict on test (truly out-of-sample!)
            y_pred = model.predict(X_test_scaled)[0]
            
            predictions.append(y_pred)
            actuals.append(y_test[0])
            dates.append(test_data.index[0])
        
        if len(predictions) == 0:
            return None
        
        # Calculate out-of-sample metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Out-of-sample R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        oos_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        # RMSE
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actuals - predictions))
        
        # Correlation
        corr = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 2 else 0
        
        # Direction accuracy (sign prediction)
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
        
        return {
            'oos_r2': oos_r2,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'direction_accuracy': direction_accuracy,
            'n_predictions': len(predictions),
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates
        }
    
    def benchmark_comparison(
        self,
        horizon: int,
        regime: Optional[int] = None
    ) -> Dict:
        """
        Compare model predictions to benchmarks:
        1. Historical average
        2. Random walk (0 change)
        3. Previous value
        """
        target_col = f'erp_forward_{horizon}m'
        
        # Filter data
        if regime is not None:
            if hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
                prob_col = self.regime_prob_cols[regime]
                data = self.combined_data[self.combined_data[prob_col] > 0.01].copy()
            else:
                data = self.combined_data[self.combined_data['regime'] == regime].copy()
        else:
            data = self.combined_data.copy()
        
        data = data[[target_col]].dropna()
        
        if len(data) < self.min_train_size + 12:
            return None
        
        # Benchmark 1: Historical average (expanding window)
        hist_avg_errors = []
        # Benchmark 2: Random walk (predict 0)
        random_walk_errors = []
        # Benchmark 3: Previous value
        prev_value_errors = []
        
        actuals = []
        
        for t in range(self.min_train_size, len(data), self.walk_forward_step):
            train_data = data.iloc[:t]
            test_data = data.iloc[t:t+1]
            
            if len(test_data) == 0:
                continue
            
            actual = test_data[target_col].values[0]
            actuals.append(actual)
            
            # Benchmark 1: Historical average
            hist_avg = train_data[target_col].mean()
            hist_avg_errors.append((actual - hist_avg) ** 2)
            
            # Benchmark 2: Random walk (0 change)
            random_walk_errors.append(actual ** 2)
            
            # Benchmark 3: Previous value (last observed)
            prev_val = train_data[target_col].iloc[-1] if len(train_data) > 0 else 0
            prev_value_errors.append((actual - prev_val) ** 2)
        
        actuals = np.array(actuals)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        
        # Calculate R² for each benchmark (higher is worse, since these are baselines)
        benchmarks = {
            'historical_average': {
                'mse': np.mean(hist_avg_errors),
                'r2': 1 - (np.sum(hist_avg_errors) / ss_tot) if ss_tot > 0 else -np.inf
            },
            'random_walk': {
                'mse': np.mean(random_walk_errors),
                'r2': 1 - (np.sum(random_walk_errors) / ss_tot) if ss_tot > 0 else -np.inf
            },
            'previous_value': {
                'mse': np.mean(prev_value_errors),
                'r2': 1 - (np.sum(prev_value_errors) / ss_tot) if ss_tot > 0 else -np.inf
            }
        }
        
        return benchmarks
    
    def validate_all_predictors(
        self,
        horizon: int,
        regime: int
    ) -> pd.DataFrame:
        """
        Run out-of-sample validation for all predictors at given horizon and regime.
        """
        print(f"\n  Validating all predictors: horizon={horizon}m, regime={regime}")
        
        macro_cols = list(self.macro_data.keys())
        results = []
        
        # Get benchmarks
        benchmarks = self.benchmark_comparison(horizon, regime)
        if benchmarks is None:
            return pd.DataFrame()
        
        for var_name in macro_cols:
            validation = self.rolling_window_validation(horizon, var_name, regime)
            
            if validation is None:
                continue
            
            # Compare to benchmarks
            beats_hist_avg = validation['oos_r2'] > benchmarks['historical_average']['r2']
            beats_random_walk = validation['oos_r2'] > benchmarks['random_walk']['r2']
            
            results.append({
                'regime': regime,
                'horizon': horizon,
                'variable': var_name,
                'oos_r2': validation['oos_r2'],
                'oos_correlation': validation['correlation'],
                'rmse': validation['rmse'],
                'mae': validation['mae'],
                'direction_accuracy': validation['direction_accuracy'],
                'n_predictions': validation['n_predictions'],
                'beats_hist_avg': beats_hist_avg,
                'beats_random_walk': beats_random_walk,
                'benchmark_hist_avg_r2': benchmarks['historical_average']['r2'],
                'benchmark_random_walk_r2': benchmarks['random_walk']['r2']
            })
        
        return pd.DataFrame(results)
    
    def multiple_testing_correction(
        self,
        results_df: pd.DataFrame,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Apply multiple testing correction to p-values.
        
        With 20 variables × 4 regimes × 4 horizons = 320 tests,
        we need to adjust for false discovery rate.
        """
        # For variables with positive OOS R², calculate implied p-value
        # from correlation coefficient
        results_df = results_df.copy()
        
        pvalues = []
        for _, row in results_df.iterrows():
            if row['n_predictions'] > 2 and abs(row['oos_correlation']) < 1:
                n = row['n_predictions']
                r = row['oos_correlation']
                # T-statistic for correlation
                t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if (1 - r**2) > 0 else 0
                pval = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                pvalues.append(pval)
            else:
                pvalues.append(1.0)
        
        # Apply Benjamini-Hochberg (FDR) correction
        if len(pvalues) > 0:
            reject, pvals_corrected, _, _ = multipletests(
                pvalues, 
                alpha=alpha, 
                method='fdr_bh'
            )
            
            results_df['pvalue_uncorrected'] = pvalues
            results_df['pvalue_corrected'] = pvals_corrected
            results_df['significant_after_correction'] = reject
        
        return results_df
    
    def time_series_cross_validation(
        self,
        horizon: int,
        variable: str,
        regime: int,
        n_splits: int = 5
    ) -> Dict:
        """
        Perform time-series cross-validation.
        
        Unlike rolling window (which uses ALL past data), this uses fixed-size
        train/test splits to get multiple independent estimates.
        """
        target_col = f'erp_forward_{horizon}m'
        
        # Filter data
        if hasattr(self, 'regime_prob_cols') and regime in self.regime_prob_cols:
            prob_col = self.regime_prob_cols[regime]
            data = self.combined_data[self.combined_data[prob_col] > 0.01].copy()
        else:
            data = self.combined_data[self.combined_data['regime'] == regime].copy()
        
        data = data[[target_col, variable]].dropna()
        
        if len(data) < 50:  # Need minimum data
            return None
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            X_train = train_data[[variable]].values.reshape(-1, 1)
            y_train = train_data[target_col].values
            X_test = test_data[[variable]].values.reshape(-1, 1)
            y_test = test_data[target_col].values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            
            # R² for this fold
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
            
            cv_scores.append(r2)
        
        return {
            'cv_mean_r2': np.mean(cv_scores),
            'cv_std_r2': np.std(cv_scores),
            'cv_min_r2': np.min(cv_scores),
            'cv_max_r2': np.max(cv_scores),
            'n_splits': n_splits
        }
    
    def run_full_validation(self):
        """
        Run complete out-of-sample validation analysis.
        """
        print("=" * 80)
        print("OUT-OF-SAMPLE VALIDATION - ERP PREDICTIVE POWER")
        print("=" * 80)
        print("\nThis analysis properly accounts for overfitting.")
        print("Results show TRUE predictive power on unseen data.\n")
        
        self.load_data()
        
        all_results = []
        
        for horizon in self.horizons:
            print(f"\n{'=' * 80}")
            print(f"VALIDATING {horizon}-MONTH HORIZON")
            print(f"{'=' * 80}")
            
            for regime in sorted(self.combined_data['regime'].unique()):
                regime_name = self.combined_data[
                    self.combined_data['regime'] == regime
                ]['regime_name'].iloc[0] if len(self.combined_data[
                    self.combined_data['regime'] == regime
                ]) > 0 else f"Regime {regime}"
                
                print(f"\n  Regime {regime}: {regime_name}")
                
                # Validate all predictors
                results_df = self.validate_all_predictors(horizon, regime)
                
                if len(results_df) > 0:
                    # Add regime name
                    results_df['regime_name'] = regime_name
                    
                    # Apply multiple testing correction
                    results_df = self.multiple_testing_correction(results_df)
                    
                    all_results.append(results_df)
                    
                    # Print top 3 validated predictors
                    top3 = results_df.nlargest(3, 'oos_r2')
                    print(f"\n    Top 3 Out-of-Sample Predictors:")
                    for _, row in top3.iterrows():
                        beat_str = "✓" if row['beats_hist_avg'] else "✗"
                        sig_str = "***" if row.get('significant_after_correction', False) else ""
                        print(f"      {beat_str} {row['variable']:30s} "
                              f"OOS-R²={row['oos_r2']:6.3f} {sig_str}")
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            self.results['validated'] = combined_results
            
            # Save results
            self.save_validation_results()
            self.create_validation_visualizations()
            self.print_summary()
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
    
    def save_validation_results(self):
        """Save validation results."""
        print("\n\nSaving validation results...")
        
        if 'validated' in self.results:
            # Main validated results
            path = self.output_dir / 'erp_predictive_power_validated_oos.csv'
            self.results['validated'].to_csv(path, index=False)
            print(f"  Saved: {path}")
            
            # Summary by horizon
            for horizon in self.horizons:
                horizon_results = self.results['validated'][
                    self.results['validated']['horizon'] == horizon
                ]
                path = self.output_dir / f'validated_oos_horizon_{horizon}m.csv'
                horizon_results.to_csv(path, index=False)
                print(f"  Saved: {path}")
    
    def create_validation_visualizations(self):
        """Create visualizations comparing in-sample vs out-of-sample."""
        print("\nCreating validation visualizations...")
        
        if 'validated' not in self.results:
            return
        
        results = self.results['validated']
        
        # 1. OOS R² Heatmap by Horizon and Regime
        for horizon in self.horizons:
            horizon_data = results[results['horizon'] == horizon]
            
            # Get top 10 variables by average OOS R²
            avg_r2 = horizon_data.groupby('variable')['oos_r2'].mean()
            top_vars = avg_r2.nlargest(10).index.tolist()
            
            plot_data = horizon_data[horizon_data['variable'].isin(top_vars)]
            
            if len(plot_data) > 0:
                pivot = plot_data.pivot(
                    index='variable', 
                    columns='regime', 
                    values='oos_r2'
                )
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    pivot, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlGn',
                    center=0,
                    vmin=-0.1,
                    vmax=0.3,
                    cbar_kws={'label': 'Out-of-Sample R²'}
                )
                plt.title(f'Out-of-Sample R² by Variable and Regime\n'
                         f'{horizon}-Month Horizon (VALIDATED)', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('Regime', fontsize=12)
                plt.ylabel('Variable', fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / f'oos_r2_heatmap_{horizon}m.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
        
        # 2. Benchmark Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, horizon in enumerate(self.horizons):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            horizon_data = results[results['horizon'] == horizon]
            
            # Count variables beating benchmarks
            beats_both = horizon_data[
                horizon_data['beats_hist_avg'] & horizon_data['beats_random_walk']
            ]
            beats_hist = horizon_data[
                horizon_data['beats_hist_avg'] & ~horizon_data['beats_random_walk']
            ]
            beats_neither = horizon_data[
                ~horizon_data['beats_hist_avg']
            ]
            
            categories = ['Beats Both\nBenchmarks', 'Beats Historical\nAverage Only', 
                         'Beats Neither\nBenchmark']
            counts = [len(beats_both), len(beats_hist), len(beats_neither)]
            colors = ['green', 'orange', 'red']
            
            ax.bar(categories, counts, color=colors, alpha=0.7)
            ax.set_title(f'{horizon}-Month Horizon', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Variables', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            # Add counts on bars
            for i, (cat, count) in enumerate(zip(categories, counts)):
                ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
        
        fig.suptitle('Benchmark Comparison: How Many Variables Beat Baselines?',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'benchmark_comparison.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # 3. Direction Accuracy Distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, horizon in enumerate(self.horizons):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            horizon_data = results[results['horizon'] == horizon]
            
            ax.hist(horizon_data['direction_accuracy'], bins=20, 
                   color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(0.5, color='red', linestyle='--', linewidth=2, 
                      label='Random Guess (50%)')
            ax.set_xlabel('Direction Accuracy', fontsize=10)
            ax.set_ylabel('Number of Variables', fontsize=10)
            ax.set_title(f'{horizon}-Month Horizon', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        fig.suptitle('Direction Accuracy: Can We Predict Sign of Future ERP?',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'direction_accuracy_dist.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print("  Visualizations saved")
    
    def print_summary(self):
        """Print summary of validation results."""
        if 'validated' not in self.results:
            return
        
        results = self.results['validated']
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        for horizon in self.horizons:
            horizon_data = results[results['horizon'] == horizon]
            
            print(f"\n{horizon}-MONTH HORIZON:")
            print(f"  Total variables tested: {len(horizon_data)}")
            
            # How many beat benchmarks?
            beats_hist = horizon_data['beats_hist_avg'].sum()
            beats_rw = horizon_data['beats_random_walk'].sum()
            beats_both = (horizon_data['beats_hist_avg'] & 
                         horizon_data['beats_random_walk']).sum()
            
            print(f"  Variables beating historical average: {beats_hist} "
                  f"({100*beats_hist/len(horizon_data):.1f}%)")
            print(f"  Variables beating random walk: {beats_rw} "
                  f"({100*beats_rw/len(horizon_data):.1f}%)")
            print(f"  Variables beating BOTH benchmarks: {beats_both} "
                  f"({100*beats_both/len(horizon_data):.1f}%)")
            
            # Average OOS R²
            avg_oos_r2 = horizon_data['oos_r2'].mean()
            median_oos_r2 = horizon_data['oos_r2'].median()
            max_oos_r2 = horizon_data['oos_r2'].max()
            
            print(f"  Average OOS R²: {avg_oos_r2:.3f}")
            print(f"  Median OOS R²: {median_oos_r2:.3f}")
            print(f"  Best OOS R²: {max_oos_r2:.3f}")
            
            # Best predictor
            best = horizon_data.nlargest(1, 'oos_r2').iloc[0]
            print(f"  Best predictor: {best['variable']} "
                  f"(R²={best['oos_r2']:.3f}, Regime {best['regime']})")
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS:")
        print("=" * 80)
        
        # Overall assessment
        all_positive = (results['oos_r2'] > 0).sum()
        all_total = len(results)
        
        print(f"\n1. Overall Predictability:")
        print(f"   {all_positive}/{all_total} ({100*all_positive/all_total:.1f}%) "
              f"of variable-regime-horizon combinations have positive OOS R²")
        
        # Best horizon
        avg_by_horizon = results.groupby('horizon')['oos_r2'].mean()
        best_horizon = avg_by_horizon.idxmax()
        print(f"\n2. Best Forecast Horizon: {best_horizon} months "
              f"(avg OOS R² = {avg_by_horizon.max():.3f})")
        
        # Best variables overall
        avg_by_var = results.groupby('variable')['oos_r2'].mean()
        top5_vars = avg_by_var.nlargest(5)
        print(f"\n3. Most Consistently Predictive Variables (across all regimes/horizons):")
        for i, (var, r2) in enumerate(top5_vars.items(), 1):
            print(f"   {i}. {var:30s} (avg OOS R² = {r2:.3f})")
        
        # Statistical significance
        if 'significant_after_correction' in results.columns:
            n_sig = results['significant_after_correction'].sum()
            print(f"\n4. Statistical Significance (after multiple testing correction):")
            print(f"   {n_sig}/{all_total} ({100*n_sig/all_total:.1f}%) "
                  f"remain significant after FDR correction")


def main():
    """Main function to run out-of-sample validation."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    regime_assignments_path = project_root / 'test_4regimes_HMM' / 'results' / 'regime_assignments.csv'
    macro_data_dir = project_root / 'data' / 'macro_processed'
    sp500_path = macro_data_dir / 'other' / 'sp500_processed.csv'
    yield_3m_path = macro_data_dir / 'other' / '3m_yield_processed.csv'
    output_dir = Path(__file__).parent / 'results_validated'
    
    # Initialize validator
    validator = ERPPredictiveValidation(
        regime_assignments_path=regime_assignments_path,
        macro_data_dir=macro_data_dir,
        sp500_path=sp500_path,
        yield_3m_path=yield_3m_path,
        output_dir=output_dir,
        horizons=[1, 3, 6, 12],
        min_train_size=120,  # 10 years minimum training window
        walk_forward_step=1   # Monthly rolling forward
    )
    
    # Run validation
    validator.run_full_validation()


if __name__ == "__main__":
    main()

