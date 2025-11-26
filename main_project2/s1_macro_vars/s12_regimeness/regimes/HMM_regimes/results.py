"""
Results processing and statistical tests for HMM regime analysis.

This module handles:
- Regime characterization tables
- Statistical tests (t-tests, ANOVA) for ERP across regimes
- Model selection metrics (AIC/BIC) for different K values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HMMResults:
    """Results processing and statistical testing for HMM regimes."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize results processor.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_regime_statistics(
        self,
        data: pd.DataFrame,
        regime_states: np.ndarray,
        regime_characteristics: Dict[int, Dict],
        erp_col: str = 'erp'
    ) -> pd.DataFrame:
        """
        Compute comprehensive statistics for each regime.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined data with macro factors and ERP
        regime_states : np.ndarray
            Regime assignments
        regime_characteristics : Dict[int, Dict]
            Regime characteristics from HMM model
        erp_col : str
            Name of ERP column
        
        Returns:
        --------
        pd.DataFrame: Regime statistics table
        """
        stats_list = []
        
        for regime_id in range(len(regime_characteristics)):
            if regime_id not in regime_characteristics:
                continue
            
            regime_mask = regime_states == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            chars = regime_characteristics[regime_id]
            
            # Calculate ERP statistics
            erp_values = regime_data[erp_col].dropna()
            
            # Calculate detailed Growth statistics
            growth_values = regime_data['growth_factor'].dropna()
            growth_stats = {
                'growth_mean': growth_values.mean() if len(growth_values) > 0 else np.nan,
                'growth_std': growth_values.std() if len(growth_values) > 0 else np.nan,
                'growth_min': growth_values.min() if len(growth_values) > 0 else np.nan,
                'growth_max': growth_values.max() if len(growth_values) > 0 else np.nan,
                'growth_median': growth_values.median() if len(growth_values) > 0 else np.nan,
                'growth_p25': growth_values.quantile(0.25) if len(growth_values) > 0 else np.nan,
                'growth_p75': growth_values.quantile(0.75) if len(growth_values) > 0 else np.nan,
                'growth_skew': growth_values.skew() if len(growth_values) > 0 else np.nan,
            }
            
            # Calculate detailed Inflation statistics
            inflation_values = regime_data['inflation_factor'].dropna()
            inflation_stats = {
                'inflation_mean': inflation_values.mean() if len(inflation_values) > 0 else np.nan,
                'inflation_std': inflation_values.std() if len(inflation_values) > 0 else np.nan,
                'inflation_min': inflation_values.min() if len(inflation_values) > 0 else np.nan,
                'inflation_max': inflation_values.max() if len(inflation_values) > 0 else np.nan,
                'inflation_median': inflation_values.median() if len(inflation_values) > 0 else np.nan,
                'inflation_p25': inflation_values.quantile(0.25) if len(inflation_values) > 0 else np.nan,
                'inflation_p75': inflation_values.quantile(0.75) if len(inflation_values) > 0 else np.nan,
                'inflation_skew': inflation_values.skew() if len(inflation_values) > 0 else np.nan,
            }
            
            stats_dict = {
                'regime_id': regime_id,
                'regime_name': chars.get('name', f'Regime {regime_id}'),
                'n_observations': len(regime_data),
                'pct_of_total': chars.get('pct_of_total', 0),
                
                # Macro averages (for backward compatibility)
                'avg_growth': chars.get('avg_growth', np.nan),
                'avg_inflation': chars.get('avg_inflation', np.nan),
                'avg_policy': chars.get('avg_policy', np.nan),
                'avg_volatility': chars.get('avg_volatility', np.nan),
                
                # Detailed Growth statistics
                **growth_stats,
                
                # Detailed Inflation statistics
                **inflation_stats,
                
                # ERP statistics
                'avg_erp': erp_values.mean() if len(erp_values) > 0 else np.nan,
                'std_erp': erp_values.std() if len(erp_values) > 0 else np.nan,
                'erp_skew': erp_values.skew() if len(erp_values) > 0 else np.nan,
                'erp_kurtosis': erp_values.kurtosis() if len(erp_values) > 0 else np.nan,
                
                # Tail statistics
                'erp_min': erp_values.min() if len(erp_values) > 0 else np.nan,
                'erp_max': erp_values.max() if len(erp_values) > 0 else np.nan,
                'erp_p5': erp_values.quantile(0.05) if len(erp_values) > 0 else np.nan,
                'erp_p95': erp_values.quantile(0.95) if len(erp_values) > 0 else np.nan,
                
                # Volatility (if available)
                'erp_volatility': regime_data.get('erp_volatility', pd.Series()).mean() 
                                 if 'erp_volatility' in regime_data.columns else np.nan,
                
                # Date range
                'start_date': chars.get('date_range', (None, None))[0],
                'end_date': chars.get('date_range', (None, None))[1]
            }
            
            stats_list.append(stats_dict)
        
        stats_df = pd.DataFrame(stats_list)
        
        # Save to CSV
        stats_file = self.output_dir / 'regime_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"    Saved regime statistics to {stats_file.name}")
        
        return stats_df
    
    def perform_erp_tests(
        self,
        data: pd.DataFrame,
        regime_states: np.ndarray,
        regime_names: Dict[int, str],
        erp_col: str = 'erp'
    ) -> Dict:
        """
        Perform t-tests and ANOVA for ERP across regimes.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined data with ERP
        regime_states : np.ndarray
            Regime assignments
        regime_names : Dict[int, str]
            Mapping of regime_id to name
        erp_col : str
            Name of ERP column
        
        Returns:
        --------
        Dict with test results
        """
        print("  Performing statistical tests on ERP across regimes...")
        
        # Prepare ERP data by regime
        erp_by_regime = {}
        for regime_id in np.unique(regime_states):
            regime_mask = regime_states == regime_id
            erp_values = data.loc[regime_mask, erp_col].dropna().values
            if len(erp_values) > 0:
                erp_by_regime[regime_id] = erp_values
        
        if len(erp_by_regime) < 2:
            print("    WARNING: Need at least 2 regimes with data for tests")
            return {}
        
        results = {}
        
        # Pairwise t-tests
        print("    Performing pairwise t-tests...")
        regime_ids = sorted(erp_by_regime.keys())
        ttest_results = []
        
        for i, regime1 in enumerate(regime_ids):
            for regime2 in regime_ids[i+1:]:
                erp1 = erp_by_regime[regime1]
                erp2 = erp_by_regime[regime2]
                
                tstat, pvalue = stats.ttest_ind(erp1, erp2)
                
                ttest_results.append({
                    'regime1': regime_names.get(regime1, f'R{regime1}'),
                    'regime2': regime_names.get(regime2, f'R{regime2}'),
                    'mean1': np.mean(erp1),
                    'mean2': np.mean(erp2),
                    'mean_diff': np.mean(erp1) - np.mean(erp2),
                    't_statistic': tstat,
                    'p_value': pvalue,
                    'significant': pvalue < 0.05
                })
        
        ttest_df = pd.DataFrame(ttest_results)
        results['pairwise_ttests'] = ttest_df
        
        # Save t-test results
        ttest_file = self.output_dir / 'pairwise_ttests_erp.csv'
        ttest_df.to_csv(ttest_file, index=False)
        print(f"      Saved pairwise t-tests to {ttest_file.name}")
        
        # ANOVA F-test
        print("    Performing ANOVA F-test...")
        fstat, pvalue_anova = stats.f_oneway(*[erp_by_regime[r] for r in regime_ids])
        
        results['anova'] = {
            'f_statistic': fstat,
            'p_value': pvalue_anova,
            'significant': pvalue_anova < 0.05,
            'n_regimes': len(regime_ids)
        }
        
        # Save ANOVA results
        anova_file = self.output_dir / 'anova_results_erp.txt'
        with open(anova_file, 'w') as f:
            f.write("ANOVA F-Test for ERP Across HMM Regimes\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"F-statistic: {fstat:.4f}\n")
            f.write(f"P-value: {pvalue_anova:.6f}\n")
            f.write(f"Significant at 5% level: {pvalue_anova < 0.05}\n")
            f.write(f"\nNumber of regimes tested: {len(regime_ids)}\n")
            f.write(f"Total observations: {sum(len(erp_by_regime[r]) for r in regime_ids)}\n")
        
        print(f"      Saved ANOVA results to {anova_file.name}")
        
        return results
    
    def compare_model_selection(
        self,
        model_metrics: Dict[int, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare AIC/BIC for different K values (2, 3, 4).
        
        Parameters:
        -----------
        model_metrics : Dict[int, Dict[str, float]]
            Dictionary mapping K (n_regimes) to metrics dict
        
        Returns:
        --------
        pd.DataFrame: Model selection comparison table
        """
        comparison_list = []
        
        for k, metrics in model_metrics.items():
            comparison_list.append({
                'n_regimes': k,
                'AIC': metrics.get('AIC', np.nan),
                'BIC': metrics.get('BIC', np.nan),
                'log_likelihood': metrics.get('log_likelihood', np.nan),
                'n_params': metrics.get('n_params', np.nan),
                'n_samples': metrics.get('n_samples', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_list).sort_values('n_regimes')
        
        # Add relative metrics (best = lowest AIC/BIC)
        if len(comparison_df) > 0:
            comparison_df['AIC_rank'] = comparison_df['AIC'].rank(ascending=True)
            comparison_df['BIC_rank'] = comparison_df['BIC'].rank(ascending=True)
            comparison_df['best_by_AIC'] = comparison_df['AIC'] == comparison_df['AIC'].min()
            comparison_df['best_by_BIC'] = comparison_df['BIC'] == comparison_df['BIC'].min()
        
        # Save to CSV
        comparison_file = self.output_dir / 'model_selection_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"    Saved model selection comparison to {comparison_file.name}")
        
        return comparison_df
    
    def create_summary_table(
        self,
        regime_stats: pd.DataFrame,
        model_metrics: Dict[str, float],
        anova_results: Dict
    ) -> pd.DataFrame:
        """
        Create summary table combining regime statistics and model metrics.
        
        Parameters:
        -----------
        regime_stats : pd.DataFrame
            Regime statistics table
        model_metrics : Dict[str, float]
            Model metrics (AIC, BIC, etc.)
        anova_results : Dict
            ANOVA test results
        
        Returns:
        --------
        pd.DataFrame: Summary table
        """
        summary_data = {
            'Model_Metrics': [
                'AIC',
                'BIC',
                'Log-Likelihood',
                'N_Parameters',
                'N_Samples'
            ],
            'Values': [
                model_metrics.get('AIC', np.nan),
                model_metrics.get('BIC', np.nan),
                model_metrics.get('log_likelihood', np.nan),
                model_metrics.get('n_params', np.nan),
                model_metrics.get('n_samples', np.nan)
            ]
        }
        
        if anova_results:
            summary_data['Model_Metrics'].extend([
                'ANOVA_F_Statistic',
                'ANOVA_P_Value',
                'ANOVA_Significant'
            ])
            summary_data['Values'].extend([
                anova_results.get('f_statistic', np.nan),
                anova_results.get('p_value', np.nan),
                anova_results.get('significant', False)
            ])
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / 'model_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"    Saved model summary to {summary_file.name}")
        
        return summary_df

