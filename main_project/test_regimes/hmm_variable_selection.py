"""
Automated Variable Selection for HMM Regime Detection

This script automatically selects the most important macro and sentiment variables
for regime detection using Hidden Markov Models, penalizing for model complexity
and retaining only essential variables.

Author: Automated HMM Variable Selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hmmlearn import hmm
import json
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class HMMVariableSelector:
    """
    Automated variable selection for HMM regime detection using penalized
    model selection criteria (BIC, AIC) and regime quality metrics.
    """
    
    def __init__(
        self,
        data_dir: Path,
        sentiment_path: Optional[Path] = None,
        n_regimes: int = 4,
        min_vars: int = 4,
        max_vars: int = 8,
        max_combinations: int = 480
    ):
        """
        Initialize the HMM variable selector.
        
        Parameters:
        -----------
        data_dir : Path
            Path to macro_processed directory
        sentiment_path : Optional[Path]
            Path to sentiment_scores.csv file
        n_regimes : int
            Number of regimes to detect (default: 3)
        min_vars : int
            Minimum number of variables in a subset (default: 2)
        max_vars : int
            Maximum number of variables in a subset (default: 8)
        max_combinations : int
            Maximum number of variable combinations to test (default: 100)
        """
        self.data_dir = Path(data_dir)
        self.sentiment_path = Path(sentiment_path) if sentiment_path else None
        self.n_regimes = n_regimes
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.max_combinations = max_combinations
        
        # Category to column mapping (most frequent series)
        self.category_to_column = {
            'inflation': 'pct_change_mom',
            'ec_growth': 'pct_change_mom',
            'mkt_vol': 'pct_change_mom',
            'mon_policy': 'pct_change_mom',
            'other': 'pct_change_mom'
        }
        
        # Sentiment category mapping
        self.sentiment_categories = {
            'inflation_sentiment': 'inflation',
            'ec_growth_sentiment': 'ec_growth',
            'monetary_policy_sentiment': 'mon_policy',
            'market_vol_sentiment': 'mkt_vol'
        }
        
        self.macro_data = {}
        self.sentiment_data = None
        self.combined_data = None
        self.results = []
        self.best_model = None
        self.best_variables = None
        
    def load_macro_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load macro data from all categories, selecting the most frequent
        series (e.g., pct_change_mom for inflation).
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping variable names to DataFrames
        """
        print("Loading macro data...")
        macro_data = {}
        
        categories = ['ec_growth', 'inflation', 'mkt_vol', 'mon_policy', 'other']
        
        for category in categories:
            category_dir = self.data_dir / category
            if not category_dir.exists():
                print(f"  Warning: {category} directory not found")
                continue
                
            # Get all CSV files in category
            csv_files = list(category_dir.glob('*_processed.csv'))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, parse_dates=['date'])
                    
                    # Select the most frequent series for this category
                    column = self.category_to_column.get(category, 'pct_change_mom')
                    
                    if column not in df.columns:
                        # Try alternative columns
                        if 'pct_change_yoy' in df.columns:
                            column = 'pct_change_yoy'
                        elif 'value' in df.columns:
                            column = 'value'
                        else:
                            print(f"  Warning: No suitable column found in {csv_file.name}")
                            continue
                    
                    # Extract date and selected column
                    var_name = csv_file.stem.replace('_processed', '')
                    var_df = df[['date', column]].copy()
                    var_df.columns = ['date', var_name]
                    var_df = var_df.dropna()
                    
                    if len(var_df) > 0:
                        macro_data[var_name] = var_df
                        print(f"  Loaded: {var_name} ({len(var_df)} observations)")
                        
                except Exception as e:
                    print(f"  Error loading {csv_file.name}: {e}")
                    continue
        
        self.macro_data = macro_data
        print(f"\nLoaded {len(macro_data)} macro variables")
        return macro_data
    
    def load_sentiment_data(self) -> Optional[pd.DataFrame]:
        """
        Load sentiment data if available.
        
        Returns:
        --------
        Optional[pd.DataFrame]
            DataFrame with sentiment scores or None if not available
        """
        if self.sentiment_path is None or not self.sentiment_path.exists():
            print("\nSentiment data not found. Continuing with macro data only.")
            return None
        
        try:
            df = pd.read_csv(self.sentiment_path, parse_dates=['date'])
            print(f"\nLoaded sentiment data: {len(df)} observations")
            print(f"  Columns: {list(df.columns)}")
            self.sentiment_data = df
            return df
        except Exception as e:
            print(f"\nError loading sentiment data: {e}")
            return None
    
    def prepare_combined_data(self, resample_freq: str = 'M') -> pd.DataFrame:
        """
        Merge all macro and sentiment data into a single DataFrame.
        Resamples all data to a common frequency (default: monthly).
        
        Parameters:
        -----------
        resample_freq : str
            Frequency to resample to (default: 'M' for monthly)
            Options: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
        
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all variables aligned by date
        """
        print("\nPreparing combined dataset...")
        print(f"  Resampling all data to {resample_freq} frequency...")
        
        # Start with first macro variable
        if not self.macro_data:
            raise ValueError("No macro data loaded")
        
        # Resample each variable to common frequency
        resampled_data = {}
        for var_name, var_df in self.macro_data.items():
            var_df = var_df.copy()
            var_df['date'] = pd.to_datetime(var_df['date'])
            var_df = var_df.set_index('date')
            
            # Resample to common frequency (take last value in each period)
            var_resampled = var_df.resample(resample_freq).last()
            var_resampled = var_resampled.reset_index()
            var_resampled = var_resampled.dropna()
            
            if len(var_resampled) > 0:
                resampled_data[var_name] = var_resampled
                print(f"    {var_name}: {len(var_df)} -> {len(var_resampled)} observations")
        
        if not resampled_data:
            raise ValueError("No data after resampling")
        
        # Merge all resampled data
        combined = None
        for var_name, var_df in resampled_data.items():
            if combined is None:
                combined = var_df.copy()
            else:
                combined = pd.merge(combined, var_df, on='date', how='inner')
        
        # Add sentiment data if available
        if self.sentiment_data is not None:
            sentiment_df = self.sentiment_data.copy()
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            sentiment_df = sentiment_df.set_index('date')
            
            # Resample sentiment data
            sentiment_resampled = sentiment_df.resample(resample_freq).last()
            sentiment_resampled = sentiment_resampled.reset_index()
            
            sentiment_cols = [col for col in sentiment_resampled.columns 
                            if col != 'date' and 'sentiment' in col]
            for col in sentiment_cols:
                combined = pd.merge(
                    combined,
                    sentiment_resampled[['date', col]],
                    on='date',
                    how='inner'
                )
        
        # Sort by date and remove rows with any missing values
        combined = combined.sort_values('date').reset_index(drop=True)
        combined = combined.dropna()
        
        print(f"\n  Combined dataset: {len(combined)} observations")
        print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"  Variables: {len(combined.columns) - 1}")
        
        self.combined_data = combined
        return combined
    
    def calculate_regime_stability(self, states: np.ndarray) -> float:
        """
        Calculate regime stability as the average duration of regimes.
        
        Parameters:
        -----------
        states : np.ndarray
            Array of regime states
            
        Returns:
        --------
        float
            Average regime duration
        """
        if len(states) == 0:
            return 0.0
        
        durations = []
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_state = states[i]
                current_duration = 1
        
        durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def fit_hmm_and_evaluate(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> Dict:
        """
        Fit HMM and calculate evaluation metrics.
        
        Parameters:
        -----------
        data : np.ndarray
            Data matrix (n_samples, n_features)
        variable_names : List[str]
            List of variable names in this subset
            
        Returns:
        --------
        Dict
            Dictionary with model metrics
        """
        if len(data) < self.n_regimes * 10:
            return None
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit HMM
        try:
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            model.fit(data_scaled)
            
            # Get states
            states = model.predict(data_scaled)
            
            # Calculate log-likelihood
            log_likelihood = model.score(data_scaled)
            
            # Calculate AIC and BIC
            # Parameter count for Gaussian HMM with full covariance:
            # - Transition matrix: n_regimes * (n_regimes - 1) (rows sum to 1)
            # - Initial probabilities: n_regimes - 1 (sum to 1)
            # - Means: n_regimes * n_features
            # - Covariances: n_regimes * n_features * (n_features + 1) / 2 (symmetric matrix)
            n_features = len(variable_names)
            n_params = (
                self.n_regimes * (self.n_regimes - 1) +  # transition matrix
                (self.n_regimes - 1) +                   # initial probabilities
                self.n_regimes * n_features +            # means
                self.n_regimes * n_features * (n_features + 1) / 2  # covariances
            )
            n_samples = len(data)
            
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_samples) * n_params
            
            # Calculate silhouette score
            try:
                silhouette = silhouette_score(data_scaled, states)
            except:
                silhouette = -1.0
            
            # Calculate regime stability
            stability = self.calculate_regime_stability(states)
            
            # Calculate regime separation (inter-regime distance)
            regime_means = []
            for r in range(self.n_regimes):
                regime_data = data_scaled[states == r]
                if len(regime_data) > 0:
                    regime_means.append(np.mean(regime_data, axis=0))
            
            separation = 0.0
            if len(regime_means) > 1:
                for i in range(len(regime_means)):
                    for j in range(i + 1, len(regime_means)):
                        separation += np.linalg.norm(regime_means[i] - regime_means[j])
                separation /= (len(regime_means) * (len(regime_means) - 1) / 2)
            
            return {
                'variables': variable_names,
                'n_variables': len(variable_names),
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'silhouette': silhouette,
                'stability': stability,
                'separation': separation,
                'model': model,
                'states': states,
                'scaler': scaler,
                'n_samples': n_samples
            }
            
        except Exception as e:
            print(f"    Error fitting HMM: {e}")
            return None
    
    def generate_variable_subsets(self) -> List[List[str]]:
        """
        Generate variable subsets to test.
        
        Returns:
        --------
        List[List[str]]
            List of variable name lists to test
        """
        all_variables = [col for col in self.combined_data.columns if col != 'date']
        
        subsets = []
        
        # Generate all combinations within size constraints
        for r in range(self.min_vars, min(self.max_vars + 1, len(all_variables) + 1)):
            for combo in combinations(all_variables, r):
                subsets.append(list(combo))
                
                # Limit total combinations
                if len(subsets) >= self.max_combinations:
                    break
            
            if len(subsets) >= self.max_combinations:
                break
        
        print(f"\nGenerated {len(subsets)} variable subsets to test")
        return subsets
    
    def select_best_variables(self) -> Tuple[List[str], Dict]:
        """
        Iteratively test variable subsets and select the best one.
        
        Returns:
        --------
        Tuple[List[str], Dict]
            Best variable subset and its evaluation results
        """
        print("\n" + "="*80)
        print("Testing Variable Subsets")
        print("="*80)
        
        subsets = self.generate_variable_subsets()
        results = []
        
        for i, subset in enumerate(subsets):
            print(f"\n[{i+1}/{len(subsets)}] Testing: {', '.join(subset)}")
            
            # Extract data for this subset
            data = self.combined_data[subset].values
            
            # Fit HMM and evaluate
            result = self.fit_hmm_and_evaluate(data, subset)
            
            if result is not None:
                results.append(result)
                print(f"  BIC: {result['bic']:.2f}, AIC: {result['aic']:.2f}, "
                      f"Silhouette: {result['silhouette']:.3f}, "
                      f"Stability: {result['stability']:.2f}")
        
        if not results:
            raise ValueError("No valid HMM models fitted")
        
        # Select best model based on BIC (penalizes complexity)
        best_result = min(results, key=lambda x: x['bic'])
        
        print("\n" + "="*80)
        print("Best Model Selected (Lowest BIC)")
        print("="*80)
        print(f"Variables: {', '.join(best_result['variables'])}")
        print(f"BIC: {best_result['bic']:.2f}")
        print(f"AIC: {best_result['aic']:.2f}")
        print(f"Silhouette Score: {best_result['silhouette']:.3f}")
        print(f"Regime Stability: {best_result['stability']:.2f}")
        print(f"Regime Separation: {best_result['separation']:.3f}")
        
        self.results = results
        self.best_model = best_result['model']
        self.best_variables = best_result['variables']
        
        return best_result['variables'], best_result
    
    def create_results_table(self) -> pd.DataFrame:
        """
        Create a results table with all tested subsets and their metrics.
        
        Returns:
        --------
        pd.DataFrame
            Results table
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            rows.append({
                'variables': ', '.join(result['variables']),
                'n_variables': result['n_variables'],
                'bic': result['bic'],
                'aic': result['aic'],
                'log_likelihood': result['log_likelihood'],
                'silhouette': result['silhouette'],
                'stability': result['stability'],
                'separation': result['separation'],
                'n_samples': result['n_samples']
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('bic')
        
        return df
    
    def plot_regime_transitions(
        self,
        output_dir: Path,
        economic_events: Optional[Dict[str, str]] = None
    ):
        """
        Plot regime transitions over time with optional economic event markers.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save plots
        economic_events : Optional[Dict[str, str]]
            Dictionary mapping dates to event descriptions
        """
        if self.best_model is None or self.best_variables is None:
            print("No best model to plot")
            return
        
        # Get data for best variables
        data = self.combined_data[self.best_variables].values
        dates = self.combined_data['date'].values
        
        # Standardize and predict
        best_result = next(r for r in self.results if r['variables'] == self.best_variables)
        scaler = best_result['scaler']
        data_scaled = scaler.transform(data)
        states = self.best_model.predict(data_scaled)
        
        # Create figure
        fig, axes = plt.subplots(len(self.best_variables) + 1, 1, figsize=(16, 4 * (len(self.best_variables) + 1)))
        
        if len(self.best_variables) == 1:
            axes = [axes]
        
        # Plot each variable
        for i, var_name in enumerate(self.best_variables):
            ax = axes[i]
            ax.plot(dates, data[:, i], alpha=0.6, linewidth=1)
            
            # Color by regime
            for regime in range(self.n_regimes):
                mask = states == regime
                ax.scatter(dates[mask], data[mask, i], 
                          alpha=0.5, s=20, label=f'Regime {regime}')
            
            ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add economic events if provided
            if economic_events:
                for event_date, event_desc in economic_events.items():
                    try:
                        event_dt = pd.to_datetime(event_date)
                        if dates[0] <= event_dt <= dates[-1]:
                            ax.axvline(event_dt, color='red', linestyle='--', 
                                     alpha=0.5, linewidth=1)
                    except:
                        pass
        
        # Plot regime states
        ax = axes[-1]
        ax.plot(dates, states, linewidth=2, alpha=0.7)
        ax.fill_between(dates, states, alpha=0.3)
        ax.set_title('Regime States Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Regime', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_yticks(range(self.n_regimes))
        ax.grid(True, alpha=0.3)
        
        # Add economic events
        if economic_events:
            for event_date, event_desc in economic_events.items():
                try:
                    event_dt = pd.to_datetime(event_date)
                    if dates[0] <= event_dt <= dates[-1]:
                        ax.axvline(event_dt, color='red', linestyle='--', 
                                 alpha=0.7, linewidth=2, label=event_desc)
                except:
                    pass
            ax.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / 'regime_transitions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved regime transitions plot: {output_path}")
        plt.close()
    
    def plot_model_comparison(self, output_dir: Path):
        """
        Plot comparison of models (BIC vs AIC, etc.).
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save plots
        """
        if not self.results:
            return
        
        results_df = self.create_results_table()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # BIC vs AIC
        ax = axes[0, 0]
        scatter = ax.scatter(results_df['aic'], results_df['bic'], 
                           c=results_df['n_variables'], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black', linewidth=1)
        ax.set_xlabel('AIC', fontsize=12)
        ax.set_ylabel('BIC', fontsize=12)
        ax.set_title('AIC vs BIC (Color = Number of Variables)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Number of Variables')
        
        # Mark best model
        best_idx = results_df['bic'].idxmin()
        ax.scatter(results_df.loc[best_idx, 'aic'], 
                  results_df.loc[best_idx, 'bic'],
                  s=300, marker='*', color='red', edgecolors='black', 
                  linewidth=2, zorder=10, label='Best (Lowest BIC)')
        ax.legend()
        
        # BIC vs Silhouette
        ax = axes[0, 1]
        scatter = ax.scatter(results_df['bic'], results_df['silhouette'],
                           c=results_df['n_variables'], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black', linewidth=1)
        ax.set_xlabel('BIC', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('BIC vs Silhouette Score', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Number of Variables')
        ax.scatter(results_df.loc[best_idx, 'bic'],
                  results_df.loc[best_idx, 'silhouette'],
                  s=300, marker='*', color='red', edgecolors='black',
                  linewidth=2, zorder=10)
        
        # BIC vs Stability
        ax = axes[1, 0]
        scatter = ax.scatter(results_df['bic'], results_df['stability'],
                           c=results_df['n_variables'], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black', linewidth=1)
        ax.set_xlabel('BIC', fontsize=12)
        ax.set_ylabel('Regime Stability (Avg Duration)', fontsize=12)
        ax.set_title('BIC vs Regime Stability', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Number of Variables')
        ax.scatter(results_df.loc[best_idx, 'bic'],
                  results_df.loc[best_idx, 'stability'],
                  s=300, marker='*', color='red', edgecolors='black',
                  linewidth=2, zorder=10)
        
        # Number of Variables vs BIC
        ax = axes[1, 1]
        for n_vars in sorted(results_df['n_variables'].unique()):
            subset = results_df[results_df['n_variables'] == n_vars]
            ax.scatter(subset['n_variables'], subset['bic'],
                      s=100, alpha=0.6, label=f'{n_vars} variables')
        ax.set_xlabel('Number of Variables', fontsize=12)
        ax.set_ylabel('BIC', fontsize=12)
        ax.set_title('Model Complexity vs BIC', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.scatter(results_df.loc[best_idx, 'n_variables'],
                  results_df.loc[best_idx, 'bic'],
                  s=300, marker='*', color='red', edgecolors='black',
                  linewidth=2, zorder=10)
        
        plt.tight_layout()
        output_path = output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot: {output_path}")
        plt.close()
    
    def save_results(self, output_dir: Path):
        """
        Save all results to files.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results table
        results_df = self.create_results_table()
        results_path = output_dir / 'hmm_selection_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved results table: {results_path}")
        
        # Save best model summary
        if self.best_variables:
            best_result = next(r for r in self.results if r['variables'] == self.best_variables)
            summary = {
                'best_variables': self.best_variables,
                'n_variables': len(self.best_variables),
                'bic': float(best_result['bic']),
                'aic': float(best_result['aic']),
                'log_likelihood': float(best_result['log_likelihood']),
                'silhouette': float(best_result['silhouette']),
                'stability': float(best_result['stability']),
                'separation': float(best_result['separation']),
                'n_regimes': self.n_regimes,
                'n_samples': int(best_result['n_samples'])
            }
            
            summary_path = output_dir / 'best_model_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved best model summary: {summary_path}")
    
    def run_full_analysis(
        self,
        output_dir: Path,
        economic_events: Optional[Dict[str, str]] = None
    ):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save all outputs
        economic_events : Optional[Dict[str, str]]
            Dictionary mapping dates to event descriptions
        """
        print("="*80)
        print("HMM Variable Selection Analysis")
        print("="*80)
        
        # Load data
        self.load_macro_data()
        self.load_sentiment_data()
        self.prepare_combined_data()
        
        # Select best variables
        best_vars, best_result = self.select_best_variables()
        
        # Create outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_results(output_dir)
        self.plot_regime_transitions(output_dir, economic_events)
        self.plot_model_comparison(output_dir)
        
        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)
        print(f"\nBest variables: {', '.join(best_vars)}")
        print(f"Results saved to: {output_dir}")


def get_default_economic_events() -> Dict[str, str]:
    """
    Get default economic event dates for visualization.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping dates to event descriptions
    """
    return {
        '2008-09-15': 'Lehman Brothers Bankruptcy',
        '2008-10-03': 'TARP Enactment',
        '2009-03-09': 'Market Bottom',
        '2010-05-06': 'Flash Crash',
        '2011-08-05': 'US Credit Downgrade',
        '2015-08-24': 'China Devaluation',
        '2016-11-08': 'US Election',
        '2020-03-23': 'COVID-19 Lockdown',
        '2020-03-27': 'CARES Act',
        '2022-02-24': 'Russia-Ukraine War',
        '2022-03-16': 'Fed Rate Hike Cycle Start'
    }


def main():
    """Main execution function."""
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data' / 'macro_processed'
    sentiment_path = project_dir / 'initial_test' / 'llm_text' / 'sentiment_scores.csv'
    output_dir = script_dir / 'results'
    
    # Initialize selector
    selector = HMMVariableSelector(
        data_dir=data_dir,
        sentiment_path=sentiment_path if sentiment_path.exists() else None,
        n_regimes=3,
        min_vars=2,
        max_vars=8,
        max_combinations=480
    )
    
    # Get economic events
    economic_events = get_default_economic_events()
    
    # Run analysis
    selector.run_full_analysis(
        output_dir=output_dir,
        economic_events=economic_events
    )


if __name__ == '__main__':
    main()

