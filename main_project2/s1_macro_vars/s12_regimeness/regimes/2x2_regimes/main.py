#!/usr/bin/env python3
"""
Main script for 2x2 Growth × Inflation Regime Classification

This script:
1. Loads macro data (growth and inflation factors)
2. Loads pre-calculated ERP (Equity Risk Premium) from equity_risk_pr.csv
3. Classifies periods into 4 regimes
4. Computes regime statistics
5. Performs statistical tests
6. Generates outputs (plots and tables)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from regime_definitions import RegimeDefinitions
from plotting import RegimePlotter
import warnings
warnings.filterwarnings('ignore')


class TwoByTwoRegimeAnalyzer:
    """
    Analyzer for 2x2 Growth × Inflation regime classification.
    """
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path = None,
        threshold_method: str = 'median'
    ):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        data_dir : Path
            Path to data directory (should contain macro_final/ and macro_processed/)
        output_dir : Path, optional
            Path to output directory for results. If None, uses results/ subdirectory.
        threshold_method : str
            Method for determining thresholds ('median', 'zero', 'mean')
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.threshold_method = threshold_method
        self.regime_def = RegimeDefinitions(threshold_method=threshold_method)
        self.plotter = RegimePlotter(self.output_dir)
        
        # Data storage
        self.macro_data = None
        self.erp_data = None
        self.combined_data = None
        self.regime_assignments = None
        self.regime_stats = None
        
    def load_macro_data(self) -> pd.DataFrame:
        """Load macro factors from final_macro.csv."""
        print("Loading macro data...")
        
        macro_file = self.data_dir / 'macro_final' / 'final_macro.csv'
        if not macro_file.exists():
            raise FileNotFoundError(f"Macro file not found: {macro_file}")
        
        df = pd.read_csv(macro_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"  Loaded {len(df)} observations")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        self.macro_data = df
        return df
    
    def load_erp(self) -> pd.DataFrame:
        """
        Load pre-calculated Equity Risk Premium (ERP) from equity_risk_pr.csv.
        
        The ERP file contains:
        - date: Date of observation
        - ERP: Equity Risk Premium (stock_return - risk_free_return)
        - stock_return: SP500 monthly return
        - risk_free_return: 3m yield monthly rate
        """
        print("Loading ERP data...")
        
        # Try multiple possible locations for ERP file
        erp_file = None
        for path in [
            self.data_dir / 'macro_processed' / 'equity_risk_pr.csv',
            self.data_dir.parent / 'main_project' / 'data' / 'macro_processed' / 'equity_risk_pr.csv'
        ]:
            if path.exists():
                erp_file = path
                break
        
        if erp_file is None:
            raise FileNotFoundError(
                f"ERP file (equity_risk_pr.csv) not found. "
                f"Searched in: {self.data_dir / 'macro_processed'}"
            )
        
        # Load ERP data
        erp_df = pd.read_csv(erp_file)
        erp_df['date'] = pd.to_datetime(erp_df['date'])
        erp_df = erp_df.sort_values('date').reset_index(drop=True)
        
        # Ensure ERP column exists
        if 'ERP' not in erp_df.columns:
            raise ValueError("ERP column not found in equity_risk_pr.csv")
        
        # Rename ERP column to lowercase for consistency
        erp_df = erp_df.rename(columns={'ERP': 'erp'})
        
        # Calculate ERP volatility (rolling 12-month std)
        erp_df = erp_df.sort_values('date')
        erp_df['erp_volatility'] = erp_df['erp'].rolling(window=12).std()
        
        # Drop rows with missing ERP
        erp_df = erp_df.dropna(subset=['erp']).reset_index(drop=True)
        
        print(f"  Loaded ERP for {len(erp_df)} observations")
        print(f"  Date range: {erp_df['date'].min()} to {erp_df['date'].max()}")
        print(f"  ERP mean: {erp_df['erp'].mean():.4f}, std: {erp_df['erp'].std():.4f}")
        
        self.erp_data = erp_df
        return erp_df
    
    def combine_data(self) -> pd.DataFrame:
        """Combine macro data with ERP data."""
        print("Combining macro and ERP data...")
        
        if self.macro_data is None:
            self.load_macro_data()
        if self.erp_data is None:
            self.load_erp()
        
        # Ensure dates are in same format (monthly, end of month)
        macro_data = self.macro_data.copy()
        erp_data = self.erp_data.copy()
        
        # Convert dates to monthly period end for better matching
        macro_data['date_month'] = pd.to_datetime(macro_data['date']).dt.to_period('M').dt.end_time
        erp_data['date_month'] = pd.to_datetime(erp_data['date']).dt.to_period('M').dt.end_time
        
        # Merge on monthly date
        combined = pd.merge(
            macro_data,
            erp_data[['date_month', 'erp', 'erp_volatility']],
            left_on='date_month',
            right_on='date_month',
            how='inner'
        )
        
        # Use original date from macro data
        combined = combined.drop(columns=['date_month'])
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"  Combined dataset: {len(combined)} observations")
        if len(combined) == 0:
            print("  WARNING: No overlapping dates found!")
            print(f"    Macro date range: {macro_data['date'].min()} to {macro_data['date'].max()}")
            print(f"    ERP date range: {erp_data['date'].min()} to {erp_data['date'].max()}")
        
        self.combined_data = combined
        return combined
    
    def classify_regimes(self) -> pd.Series:
        """Classify all periods into regimes."""
        print("Classifying regimes...")
        
        if self.combined_data is None:
            self.combine_data()
        
        # Classify using regime definitions
        regimes = self.regime_def.classify_dataframe(
            self.combined_data,
            growth_col='growth_factor',
            inflation_col='inflation_factor'
        )
        
        self.combined_data['regime'] = regimes
        self.regime_assignments = regimes
        
        # Print regime distribution
        regime_counts = regimes.value_counts().sort_index()
        print("\n  Regime distribution:")
        for regime_id, count in regime_counts.items():
            pct = count / len(regimes) * 100
            name = self.regime_def.get_regime_short_name(regime_id)
            print(f"    {name} (R{regime_id}): {count} periods ({pct:.1f}%)")
        
        print(f"\n  Growth threshold: {self.regime_def.growth_threshold:.4f}")
        print(f"  Inflation threshold: {self.regime_def.inflation_threshold:.4f}")
        
        return regimes
    
    def compute_regime_statistics(self) -> pd.DataFrame:
        """Compute statistics for each regime."""
        print("Computing regime statistics...")
        
        if self.regime_assignments is None:
            self.classify_regimes()
        
        stats_list = []
        
        for regime_id in range(4):
            regime_mask = self.combined_data['regime'] == regime_id
            regime_data = self.combined_data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Macro statistics
            stats_dict = {
                'regime_id': regime_id,
                'regime_name': self.regime_def.get_regime_short_name(regime_id),
                'n_observations': len(regime_data),
                'pct_of_total': len(regime_data) / len(self.combined_data) * 100,
                
                # Growth and Inflation
                'avg_growth': regime_data['growth_factor'].mean(),
                'std_growth': regime_data['growth_factor'].std(),
                'avg_inflation': regime_data['inflation_factor'].mean(),
                'std_inflation': regime_data['inflation_factor'].std(),
                
                # Policy and Volatility
                'avg_policy': regime_data['monetary_policy_factor'].mean(),
                'avg_volatility': regime_data['market_volatility_factor'].mean(),
                
                # ERP statistics
                'avg_erp': regime_data['erp'].mean(),
                'std_erp': regime_data['erp'].std(),
                'erp_volatility': regime_data['erp_volatility'].mean(),
                'erp_skew': regime_data['erp'].skew(),
                'erp_kurtosis': regime_data['erp'].kurtosis(),
                
                # Tail statistics
                'erp_min': regime_data['erp'].min(),
                'erp_max': regime_data['erp'].max(),
                'erp_p5': regime_data['erp'].quantile(0.05),
                'erp_p95': regime_data['erp'].quantile(0.95),
                
                # Date range
                'start_date': regime_data['date'].min(),
                'end_date': regime_data['date'].max()
            }
            
            stats_list.append(stats_dict)
        
        stats_df = pd.DataFrame(stats_list)
        self.regime_stats = stats_df
        
        # Save to CSV
        stats_file = self.output_dir / 'regime_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"  Saved regime statistics to {stats_file}")
        
        return stats_df
    
    def perform_statistical_tests(self) -> Dict:
        """Perform t-tests and ANOVA for ERP across regimes."""
        print("Performing statistical tests...")
        
        if self.regime_assignments is None:
            self.classify_regimes()
        
        # Prepare ERP data by regime
        erp_by_regime = {}
        for regime_id in range(4):
            regime_mask = self.combined_data['regime'] == regime_id
            erp_by_regime[regime_id] = self.combined_data.loc[regime_mask, 'erp'].values
        
        # Remove empty regimes
        erp_by_regime = {k: v for k, v in erp_by_regime.items() if len(v) > 0}
        
        results = {}
        
        # Pairwise t-tests
        print("  Performing pairwise t-tests...")
        regime_ids = sorted(erp_by_regime.keys())
        ttest_results = []
        
        for i, regime1 in enumerate(regime_ids):
            for regime2 in regime_ids[i+1:]:
                erp1 = erp_by_regime[regime1]
                erp2 = erp_by_regime[regime2]
                
                tstat, pvalue = stats.ttest_ind(erp1, erp2)
                
                ttest_results.append({
                    'regime1': self.regime_def.get_regime_short_name(regime1),
                    'regime2': self.regime_def.get_regime_short_name(regime2),
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
        ttest_file = self.output_dir / 'pairwise_ttests.csv'
        ttest_df.to_csv(ttest_file, index=False)
        print(f"    Saved pairwise t-tests to {ttest_file}")
        
        # ANOVA F-test
        print("  Performing ANOVA F-test...")
        all_erp = []
        all_regimes = []
        
        for regime_id, erp_values in erp_by_regime.items():
            all_erp.extend(erp_values)
            all_regimes.extend([regime_id] * len(erp_values))
        
        fstat, pvalue_anova = stats.f_oneway(*[erp_by_regime[r] for r in regime_ids])
        
        results['anova'] = {
            'f_statistic': fstat,
            'p_value': pvalue_anova,
            'significant': pvalue_anova < 0.05,
            'n_regimes': len(regime_ids)
        }
        
        # Save ANOVA results
        anova_file = self.output_dir / 'anova_results.txt'
        with open(anova_file, 'w') as f:
            f.write("ANOVA F-Test for ERP Across Regimes\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"F-statistic: {fstat:.4f}\n")
            f.write(f"P-value: {pvalue_anova:.6f}\n")
            f.write(f"Significant at 5% level: {pvalue_anova < 0.05}\n")
            f.write(f"\nNumber of regimes tested: {len(regime_ids)}\n")
            f.write(f"Total observations: {len(all_erp)}\n")
        
        print(f"    Saved ANOVA results to {anova_file}")
        
        return results
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("2x2 REGIME CLASSIFICATION ANALYSIS")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}\n")
        
        # Load and combine data
        self.combine_data()
        
        # Classify regimes
        self.classify_regimes()
        
        # Compute statistics
        self.compute_regime_statistics()
        
        # Perform statistical tests
        self.perform_statistical_tests()
        
        # Generate plots
        print("\nGenerating plots...")
        self.plotter.plot_scatter(self.combined_data, self.regime_def)
        self.plotter.plot_boxplots(self.combined_data, self.regime_def)
        self.plotter.plot_time_series(self.combined_data, self.regime_def)
        
        # Save regime assignments
        assignments_file = self.output_dir / 'regime_assignments.csv'
        output_df = self.combined_data[['date', 'regime', 'growth_factor', 
                                       'inflation_factor', 'erp']].copy()
        output_df['regime_name'] = output_df['regime'].map(
            lambda x: self.regime_def.get_regime_short_name(x)
        )
        output_df.to_csv(assignments_file, index=False)
        print(f"  Saved regime assignments to {assignments_file}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main execution function."""
    # Set up paths
    # Script is at: main_project2/s1_macro_vars/s12_regimeness/2x2_regimes/main.py
    # Data is at: main_project2/data/
    script_dir = Path(__file__).parent.absolute()
    # Path structure: main_project2/s1_macro_vars/s12_regimeness/2x2_regimes/
    # Go up 4 levels: 2x2_regimes -> s12_regimeness -> s1_macro_vars -> main_project2
    main_project2_dir = script_dir.parent.parent.parent.parent / 'main_project2'
    data_dir = main_project2_dir / 'data'
    output_dir = script_dir / 'results'
    
    # Verify data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Initialize analyzer
    analyzer = TwoByTwoRegimeAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        threshold_method='median'
    )
    
    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

