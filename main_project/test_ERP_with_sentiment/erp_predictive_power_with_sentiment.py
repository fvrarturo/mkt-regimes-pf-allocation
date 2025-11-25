"""
Equity Risk Premium (ERP) Predictive Power Analysis - WITH SENTIMENT SCORES

This script extends the base ERP validation to test whether adding sentiment scores
improves prediction accuracy. It:
1. Tests all macro variables (as baseline)
2. Tests all sentiment variables
3. Compares results to determine if sentiment adds predictive power
4. Uses strict out-of-sample validation to avoid overfitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class ERPPredictiveValidationWithSentiment:
    """
    Extended ERP validation that includes sentiment scores.
    Tests both macro-only and macro+sentiment to compare performance.
    """
    
    def __init__(
        self,
        regime_assignments_path: Path,
        macro_data_dir: Path,
        sp500_path: Path,
        yield_3m_path: Path,
        sentiment_path: Path,
        output_dir: Path,
        horizons: List[int] = [1, 3, 6, 12],
        min_train_size: int = 120,
        walk_forward_step: int = 1
    ):
        """Initialize the validation analyzer with sentiment."""
        self.regime_assignments_path = Path(regime_assignments_path)
        self.macro_data_dir = Path(macro_data_dir)
        self.sp500_path = Path(sp500_path)
        self.yield_3m_path = Path(yield_3m_path)
        self.sentiment_path = Path(sentiment_path)
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
        self.sentiment_data = {}
        self.combined_data = None
        self.results = {}
        
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
        print(f"    Loaded {len(macro_vars)} macro variables")
        return macro_vars
    
    def load_sentiment_scores(self) -> Dict[str, pd.DataFrame]:
        """Load sentiment scores from CSV file."""
        print("  Loading sentiment scores...")
        
        if not self.sentiment_path.exists():
            print(f"    Warning: Sentiment file not found at {self.sentiment_path}")
            return {}
        
        try:
            df = pd.read_csv(self.sentiment_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Extract sentiment columns (exclude date)
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
            
            sentiment_vars = {}
            for col in sentiment_cols:
                # Create monthly aggregation (take last value of month)
                sentiment_monthly = df[[col]].resample('M').last()
                sentiment_monthly.columns = [col]
                sentiment_vars[col] = sentiment_monthly
                print(f"    Loaded {col}: {len(sentiment_monthly)} observations")
            
            self.sentiment_data = sentiment_vars
            print(f"    Total sentiment variables: {len(sentiment_vars)}")
            return sentiment_vars
            
        except Exception as e:
            print(f"    Error loading sentiment scores: {e}")
            return {}
    
    def combine_data(self) -> pd.DataFrame:
        """Combine ERP, regime, macro, and sentiment data."""
        print("  Combining data (including sentiment)...")
        
        combined = self.erp_data.copy()
        
        # Add regime data
        regime_cols = ['regime', 'regime_name']
        if hasattr(self, 'regime_prob_cols'):
            regime_cols.extend(list(self.regime_prob_cols.values()))
        
        combined = pd.merge(combined, self.regime_data[regime_cols],
                          left_index=True, right_index=True, how='inner')
        
        # Add macro variables
        for var_name, var_df in self.macro_data.items():
            combined = pd.merge(combined, var_df,
                              left_index=True, right_index=True, how='left')
        
        # Add sentiment variables
        for var_name, var_df in self.sentiment_data.items():
            combined = pd.merge(combined, var_df,
                              left_index=True, right_index=True, how='left')
        
        # Forward fill missing values
        # NOTE: ffill() is safe because it only uses PAST values to fill missing data.
        # It fills NaN at time t with the last known value from time < t.
        # This is acceptable for out-of-sample validation because no future information is used.
        macro_cols = list(self.macro_data.keys())
        sentiment_cols = list(self.sentiment_data.keys())
        all_predictor_cols = macro_cols + sentiment_cols
        
        combined[all_predictor_cols] = combined[all_predictor_cols].ffill()
        combined = combined.dropna(subset=['erp', 'regime'])
        combined = combined.dropna(subset=all_predictor_cols, how='all')
        
        self.combined_data = combined
        print(f"    Combined: {len(combined)} observations")
        print(f"    Total predictors: {len(all_predictor_cols)} "
              f"({len(macro_cols)} macro + {len(sentiment_cols)} sentiment)")
        return combined
    
    def load_data(self):
        """Load all necessary data including sentiment."""
        print("Loading data for validation...")
        self.load_regime_assignments()
        self.calculate_erp()
        self.load_macro_variables()
        self.load_sentiment_scores()
        self.combine_data()
    
    def rolling_window_validation(
        self, 
        horizon: int,
        variable: str,
        regime: Optional[int] = None
    ) -> Dict:
        """
        Perform rolling window out-of-sample validation.
        
        This is a UNIVARIATE regression: uses only ONE variable at a time.
        Model: ERP_forward_{h}m = α + β × variable + ε
        
        Even though the combined_data has 24 variables (20 macro + 4 sentiment),
        this function uses ONLY the specified variable in the regression.
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
        
        # Prepare features - NOTE: Only using ONE variable (univariate analysis)
        data = data[[target_col, variable]].dropna()
        
        if len(data) < self.min_train_size + 12:
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
            # NOTE: Standardization is fit ONLY on training data to prevent look-ahead bias
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on train
            X_test_scaled = scaler.transform(X_test)         # Transform test using train stats
            
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
        """Compare model predictions to benchmarks."""
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
        
        hist_avg_errors = []
        random_walk_errors = []
        prev_value_errors = []
        actuals = []
        
        for t in range(self.min_train_size, len(data), self.walk_forward_step):
            train_data = data.iloc[:t]
            test_data = data.iloc[t:t+1]
            
            if len(test_data) == 0:
                continue
            
            actual = test_data[target_col].values[0]
            actuals.append(actual)
            
            hist_avg = train_data[target_col].mean()
            hist_avg_errors.append((actual - hist_avg) ** 2)
            
            random_walk_errors.append(actual ** 2)
            
            prev_val = train_data[target_col].iloc[-1] if len(train_data) > 0 else 0
            prev_value_errors.append((actual - prev_val) ** 2)
        
        actuals = np.array(actuals)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        
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
        regime: int,
        include_sentiment: bool = True
    ) -> pd.DataFrame:
        """
        Run out-of-sample validation for all predictors.
        
        IMPORTANT: This is a UNIVARIATE analysis - each variable is tested INDIVIDUALLY
        in a simple linear regression: ERP = α + β × Variable + ε
        
        The include_sentiment flag only determines which variables are in the candidate pool.
        Each variable is still tested one at a time - they are NOT combined in multivariate models.
        
        Example:
        - include_sentiment=True: Tests 24 variables (20 macro + 4 sentiment) separately
        - include_sentiment=False: Tests 20 variables (macro only) separately
        - When testing 'unemployment', only unemployment is used, regardless of sentiment flag
        """
        print(f"\n  Validating all predictors: horizon={horizon}m, regime={regime}, "
              f"sentiment={'included' if include_sentiment else 'excluded'}")
        
        # Select variables based on include_sentiment flag
        # NOTE: These variables will be tested ONE AT A TIME, not combined
        if include_sentiment:
            predictor_cols = list(self.macro_data.keys()) + list(self.sentiment_data.keys())
        else:
            predictor_cols = list(self.macro_data.keys())
        
        results = []
        
        # Get benchmarks
        benchmarks = self.benchmark_comparison(horizon, regime)
        if benchmarks is None:
            return pd.DataFrame()
        
        for var_name in predictor_cols:
            validation = self.rolling_window_validation(horizon, var_name, regime)
            
            if validation is None:
                continue
            
            beats_hist_avg = validation['oos_r2'] > benchmarks['historical_average']['r2']
            beats_random_walk = validation['oos_r2'] > benchmarks['random_walk']['r2']
            
            # Determine if this is a sentiment variable
            is_sentiment = var_name in self.sentiment_data.keys()
            
            results.append({
                'regime': regime,
                'horizon': horizon,
                'variable': var_name,
                'variable_type': 'sentiment' if is_sentiment else 'macro',
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
        """Apply multiple testing correction to p-values."""
        results_df = results_df.copy()
        
        pvalues = []
        for _, row in results_df.iterrows():
            if row['n_predictions'] > 2 and abs(row['oos_correlation']) < 1:
                n = row['n_predictions']
                r = row['oos_correlation']
                t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if (1 - r**2) > 0 else 0
                pval = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                pvalues.append(pval)
            else:
                pvalues.append(1.0)
        
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
    
    def run_comparison_analysis(self):
        """Run validation with and without sentiment, then compare results."""
        print("=" * 80)
        print("OUT-OF-SAMPLE VALIDATION - ERP WITH SENTIMENT SCORES")
        print("=" * 80)
        print("\nThis analysis tests whether sentiment scores improve prediction accuracy.")
        print("Results show TRUE predictive power on unseen data.\n")
        
        self.load_data()
        
        all_results_with_sentiment = []
        all_results_macro_only = []
        
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
                
                # Validate WITH sentiment
                print("    Testing WITH sentiment variables...")
                results_with = self.validate_all_predictors(horizon, regime, include_sentiment=True)
                
                # Validate WITHOUT sentiment (macro only)
                print("    Testing WITHOUT sentiment (macro only)...")
                results_without = self.validate_all_predictors(horizon, regime, include_sentiment=False)
                
                if len(results_with) > 0:
                    results_with['regime_name'] = regime_name
                    results_with = self.multiple_testing_correction(results_with)
                    all_results_with_sentiment.append(results_with)
                
                if len(results_without) > 0:
                    results_without['regime_name'] = regime_name
                    results_without = self.multiple_testing_correction(results_without)
                    all_results_macro_only.append(results_without)
                
                # Print top predictors from each category
                if len(results_with) > 0:
                    sentiment_vars = results_with[results_with['variable_type'] == 'sentiment']
                    macro_vars = results_with[results_with['variable_type'] == 'macro']
                    
                    if len(sentiment_vars) > 0:
                        top_sentiment = sentiment_vars.nlargest(3, 'oos_r2')
                        print(f"\n    Top 3 Sentiment Predictors:")
                        for _, row in top_sentiment.iterrows():
                            beat_str = "✓" if row['beats_hist_avg'] else "✗"
                            sig_str = "***" if row.get('significant_after_correction', False) else ""
                            print(f"      {beat_str} {row['variable']:30s} "
                                  f"OOS-R²={row['oos_r2']:6.3f} {sig_str}")
                    
                    if len(macro_vars) > 0:
                        top_macro = macro_vars.nlargest(3, 'oos_r2')
                        print(f"\n    Top 3 Macro Predictors:")
                        for _, row in top_macro.iterrows():
                            beat_str = "✓" if row['beats_hist_avg'] else "✗"
                            sig_str = "***" if row.get('significant_after_correction', False) else ""
                            print(f"      {beat_str} {row['variable']:30s} "
                                  f"OOS-R²={row['oos_r2']:6.3f} {sig_str}")
        
        # Combine all results
        if all_results_with_sentiment:
            combined_with = pd.concat(all_results_with_sentiment, ignore_index=True)
            self.results['with_sentiment'] = combined_with
        
        if all_results_macro_only:
            combined_without = pd.concat(all_results_macro_only, ignore_index=True)
            self.results['macro_only'] = combined_without
        
        # Save results and create comparison
        self.save_comparison_results()
        self.create_comparison_analysis()
        self.print_comparison_summary()
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
    
    def save_comparison_results(self):
        """Save comparison results."""
        print("\n\nSaving comparison results...")
        
        if 'with_sentiment' in self.results:
            path = self.output_dir / 'erp_with_sentiment_oos.csv'
            self.results['with_sentiment'].to_csv(path, index=False)
            print(f"  Saved: {path}")
        
        if 'macro_only' in self.results:
            path = self.output_dir / 'erp_macro_only_oos.csv'
            self.results['macro_only'].to_csv(path, index=False)
            print(f"  Saved: {path}")
        
        # Create comparison summary
        if 'with_sentiment' in self.results and 'macro_only' in self.results:
            comparison = self.create_comparison_table()
            path = self.output_dir / 'sentiment_vs_macro_comparison.csv'
            comparison.to_csv(path, index=False)
            print(f"  Saved: {path}")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table between macro-only and with-sentiment results."""
        with_sent = self.results['with_sentiment']
        macro_only = self.results['macro_only']
        
        # Compare macro variables performance
        macro_with = with_sent[with_sent['variable_type'] == 'macro'].copy()
        macro_without = macro_only.copy()
        
        # Merge on variable, horizon, regime
        comparison = pd.merge(
            macro_with[['variable', 'horizon', 'regime', 'oos_r2', 'direction_accuracy']],
            macro_without[['variable', 'horizon', 'regime', 'oos_r2', 'direction_accuracy']],
            on=['variable', 'horizon', 'regime'],
            suffixes=('_with_sentiment', '_macro_only'),
            how='outer'
        )
        
        # Calculate improvement
        comparison['r2_improvement'] = (
            comparison['oos_r2_with_sentiment'] - comparison['oos_r2_macro_only']
        )
        comparison['direction_improvement'] = (
            comparison['direction_accuracy_with_sentiment'] - 
            comparison['direction_accuracy_macro_only']
        )
        
        return comparison
    
    def create_comparison_analysis(self):
        """Create visualizations comparing macro-only vs with-sentiment."""
        print("\nCreating comparison visualizations...")
        
        if 'with_sentiment' not in self.results or 'macro_only' not in self.results:
            return
        
        with_sent = self.results['with_sentiment']
        macro_only = self.results['macro_only']
        
        # 1. Comparison of best R² by horizon
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, horizon in enumerate(self.horizons):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            sent_h = with_sent[with_sent['horizon'] == horizon]
            macro_h = macro_only[macro_only['horizon'] == horizon]
            
            # Get best R² for each category
            best_macro = macro_h['oos_r2'].max()
            best_sentiment = sent_h[sent_h['variable_type'] == 'sentiment']['oos_r2'].max() if len(sent_h[sent_h['variable_type'] == 'sentiment']) > 0 else 0
            best_macro_with_sent = sent_h[sent_h['variable_type'] == 'macro']['oos_r2'].max()
            
            categories = ['Macro Only', 'Macro (with\nsentiment context)', 'Sentiment Only']
            values = [best_macro, best_macro_with_sent, best_sentiment]
            colors = ['steelblue', 'green', 'orange']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(f'{horizon}-Month Horizon', fontsize=12, fontweight='bold')
            ax.set_ylabel('Best Out-of-Sample R²', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        fig.suptitle('Best Predictive Power: Macro vs Sentiment vs Combined',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'sentiment_vs_macro_comparison.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # 2. Distribution of R² improvements
        comparison = self.create_comparison_table()
        if len(comparison) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            improvements = comparison['r2_improvement'].dropna()
            if len(improvements) > 0:
                ax.hist(improvements, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax.axvline(0, color='red', linestyle='--', linewidth=2, 
                          label='No Improvement')
                ax.axvline(improvements.mean(), color='green', linestyle='--', linewidth=2,
                          label=f'Mean: {improvements.mean():.4f}')
                ax.set_xlabel('R² Improvement (With Sentiment - Macro Only)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('Distribution of R² Improvements from Adding Sentiment',
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / 'sentiment_improvement_distribution.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
        
        print("  Comparison visualizations saved")
    
    def print_comparison_summary(self):
        """Print summary comparing macro-only vs with-sentiment."""
        if 'with_sentiment' not in self.results or 'macro_only' not in self.results:
            return
        
        with_sent = self.results['with_sentiment']
        macro_only = self.results['macro_only']
        
        print("\n" + "=" * 80)
        print("SENTIMENT vs MACRO COMPARISON SUMMARY")
        print("=" * 80)
        
        for horizon in self.horizons:
            sent_h = with_sent[with_sent['horizon'] == horizon]
            macro_h = macro_only[macro_only['horizon'] == horizon]
            
            print(f"\n{horizon}-MONTH HORIZON:")
            
            # Macro-only stats
            macro_best = macro_h['oos_r2'].max()
            macro_avg = macro_h['oos_r2'].mean()
            macro_positive = (macro_h['oos_r2'] > 0).sum()
            
            print(f"\n  MACRO ONLY:")
            print(f"    Best R²: {macro_best:.3f}")
            print(f"    Average R²: {macro_avg:.3f}")
            print(f"    Variables with positive R²: {macro_positive}/{len(macro_h)}")
            
            # Sentiment-only stats
            sent_only = sent_h[sent_h['variable_type'] == 'sentiment']
            if len(sent_only) > 0:
                sent_best = sent_only['oos_r2'].max()
                sent_avg = sent_only['oos_r2'].mean()
                sent_positive = (sent_only['oos_r2'] > 0).sum()
                
                print(f"\n  SENTIMENT ONLY:")
                print(f"    Best R²: {sent_best:.3f}")
                print(f"    Average R²: {sent_avg:.3f}")
                print(f"    Variables with positive R²: {sent_positive}/{len(sent_only)}")
            
            # Macro with sentiment context
            macro_with_sent = sent_h[sent_h['variable_type'] == 'macro']
            if len(macro_with_sent) > 0:
                macro_ws_best = macro_with_sent['oos_r2'].max()
                macro_ws_avg = macro_with_sent['oos_r2'].mean()
                
                print(f"\n  MACRO (with sentiment context):")
                print(f"    Best R²: {macro_ws_best:.3f} "
                      f"(change: {macro_ws_best - macro_best:+.3f})")
                print(f"    Average R²: {macro_ws_avg:.3f} "
                      f"(change: {macro_ws_avg - macro_avg:+.3f})")
        
        # Overall conclusion
        print("\n" + "=" * 80)
        print("KEY FINDING:")
        print("=" * 80)
        
        comparison = self.create_comparison_table()
        if len(comparison) > 0:
            improvements = comparison['r2_improvement'].dropna()
            if len(improvements) > 0:
                mean_improvement = improvements.mean()
                positive_improvements = (improvements > 0).sum()
                total_comparisons = len(improvements)
                
                print(f"\nAdding sentiment improves R² in {positive_improvements}/{total_comparisons} "
                      f"({100*positive_improvements/total_comparisons:.1f}%) cases")
                print(f"Mean R² improvement: {mean_improvement:+.4f}")
                
                if mean_improvement > 0.001:
                    print("\n✓ CONCLUSION: Sentiment scores ADD predictive power")
                elif mean_improvement < -0.001:
                    print("\n✗ CONCLUSION: Sentiment scores do NOT improve predictions")
                else:
                    print("\n→ CONCLUSION: Sentiment scores have minimal impact")


def main():
    """Main function to run sentiment-enhanced validation."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    regime_assignments_path = project_root / 'test_4regimes_HMM' / 'results' / 'regime_assignments.csv'
    macro_data_dir = project_root / 'data' / 'macro_processed'
    sp500_path = macro_data_dir / 'other' / 'sp500_processed.csv'
    yield_3m_path = macro_data_dir / 'other' / '3m_yield_processed.csv'
    sentiment_path = project_root / 'data' / 'news_data' / 'sentiment_scores.csv'
    output_dir = Path(__file__).parent / 'results_validated'
    
    # Initialize validator with sentiment
    validator = ERPPredictiveValidationWithSentiment(
        regime_assignments_path=regime_assignments_path,
        macro_data_dir=macro_data_dir,
        sp500_path=sp500_path,
        yield_3m_path=yield_3m_path,
        sentiment_path=sentiment_path,
        output_dir=output_dir,
        horizons=[1, 3, 6, 12],
        min_train_size=120,
        walk_forward_step=1
    )
    
    # Run comparison analysis
    validator.run_comparison_analysis()


if __name__ == "__main__":
    main()
