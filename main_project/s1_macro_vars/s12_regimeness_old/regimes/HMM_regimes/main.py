#!/usr/bin/env python3
"""
Main script for HMM Regime Detection with 4 Macro Variables

This script:
1. Loads macro data (4 factors: growth, inflation, policy, volatility)
2. Loads pre-calculated ERP (Equity Risk Premium)
3. Fits HMM models with K = 2, 3, 4 regimes
4. Selects best K based on AIC/BIC
5. Extracts regime assignments and probabilities
6. Computes regime statistics
7. Performs statistical tests
8. Generates outputs (plots and tables)
9. Compares HMM regimes vs simple 4 quadrants
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory and shared utilities to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REGIMES_DIR = SCRIPT_DIR.parent
SECTION_DIR = SCRIPT_DIR.parents[2]  # s1_macro_vars
for path in (SCRIPT_DIR, REGIMES_DIR, SECTION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from hmm_model import HMMRegimeModel
from plotting import HMMPlotter
from results import HMMResults
from path_utils import get_data_dir

# Import 2x2 regime definitions for comparison
two_by_two_path = SCRIPT_DIR.parent / '2x2_regimes'
two_by_two_str = str(two_by_two_path)
if two_by_two_str not in sys.path:
    sys.path.insert(0, two_by_two_str)
try:
    from regime_definitions import RegimeDefinitions
except ImportError:
    print("Warning: Could not import 2x2 regime definitions. Comparison will be skipped.")
    RegimeDefinitions = None


class HMMRegimeAnalyzer:
    """
    Main analyzer for HMM regime detection with 4 macro variables.
    """
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path = None,
        n_regimes: int = None  # If None, will test K=2,3,4 and select best
    ):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        data_dir : Path
            Path to data directory (should contain macro_final/ and macro_processed/)
        output_dir : Path, optional
            Path to output directory for results. If None, uses results/ subdirectory.
        n_regimes : int, optional
            Number of regimes. If None, will test K=2,3,4 and select best by BIC.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_regimes = n_regimes
        
        # Initialize components
        self.plotter = HMMPlotter(self.output_dir)
        self.results_processor = HMMResults(self.output_dir)
        
        # Data storage
        self.macro_data = None
        self.erp_data = None
        self.combined_data = None
        self.hmm_model = None
        self.regime_states = None
        self.regime_probs = None
        self.regime_characteristics = None
        self.model_metrics = {}
        
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
        """
        Combine macro data with ERP data.
        
        NOTE: ERP is NOT used as an input feature to the HMM model.
        The HMM only uses the 4 macro variables (growth, inflation, policy, volatility).
        ERP is combined here for POST-HOC analysis:
        - To characterize each regime (average ERP per regime)
        - To perform statistical tests (compare ERP means across regimes)
        - To create regime statistics tables
        """
        print("Combining macro and ERP data...")
        print("  NOTE: ERP is used for post-hoc analysis only, not as HMM input feature")
        
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
        # ERP is merged for analysis purposes, not for model fitting
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
        
        # Drop rows with missing macro factors
        # NOTE: We drop on macro factors only - ERP can be missing (it's not used in HMM)
        macro_cols = [
            'growth_factor',
            'inflation_factor',
            'monetary_policy_factor',
            'market_volatility_factor'
        ]
        combined = combined.dropna(subset=macro_cols).reset_index(drop=True)
        
        print(f"  Combined dataset: {len(combined)} observations")
        if len(combined) == 0:
            raise ValueError("No overlapping dates found after merging!")
        
        self.combined_data = combined
        return combined
    
    def select_best_k(self, k_values: List[int] = [2, 3, 4]) -> int:
        """
        Test different K values and select best based on BIC.
        
        Parameters:
        -----------
        k_values : List[int]
            List of K values to test
        
        Returns:
        --------
        int: Best K value
        """
        print("\n" + "=" * 80)
        print("MODEL SELECTION: Testing different K values")
        print("=" * 80)
        
        if self.combined_data is None:
            self.combine_data()
        
        # Prepare features once
        temp_model = HMMRegimeModel(n_regimes=3)  # Temporary for feature prep
        features = temp_model.prepare_features(self.combined_data, fit_scaler=True)
        
        # Test each K value
        for k in k_values:
            print(f"\nTesting K = {k}...")
            model = HMMRegimeModel(n_regimes=k)
            model.scaler = temp_model.scaler  # Use same scaler
            
            try:
                model.fit(features, n_init=5)
                metrics = model.calculate_model_metrics(features)
                self.model_metrics[k] = metrics
                
                print(f"  AIC: {metrics['AIC']:.2f}")
                print(f"  BIC: {metrics['BIC']:.2f}")
                print(f"  Log-likelihood: {metrics['log_likelihood']:.2f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        # Select best K by BIC (lower is better)
        if len(self.model_metrics) == 0:
            raise RuntimeError("Failed to fit any models!")
        
        best_k = min(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['BIC'])
        
        print(f"\n{'='*80}")
        print(f"Best K = {best_k} (lowest BIC: {self.model_metrics[best_k]['BIC']:.2f})")
        print(f"{'='*80}\n")
        
        # Save model selection comparison
        self.results_processor.compare_model_selection(self.model_metrics)
        
        return best_k
    
    def fit_hmm(self) -> HMMRegimeModel:
        """Fit HMM model with selected or specified K."""
        print("\n" + "=" * 80)
        print("FITTING HMM MODEL")
        print("=" * 80)
        
        if self.combined_data is None:
            self.combine_data()
        
        # Select K if not specified
        if self.n_regimes is None:
            self.n_regimes = self.select_best_k()
        
        # Fit final model
        print(f"\nFitting final HMM model with K = {self.n_regimes}...")
        self.hmm_model = HMMRegimeModel(n_regimes=self.n_regimes)
        features = self.hmm_model.prepare_features(self.combined_data, fit_scaler=True)
        self.hmm_model.fit(features, n_init=10)
        
        # Get regime assignments and probabilities
        self.regime_states = self.hmm_model.predict(features)
        self.regime_probs = self.hmm_model.predict_proba(features)
        transition_matrix = self.hmm_model.get_transition_matrix()
        
        # Calculate model metrics
        final_metrics = self.hmm_model.calculate_model_metrics(features)
        self.model_metrics[self.n_regimes] = final_metrics
        
        print(f"\nModel Metrics:")
        print(f"  AIC: {final_metrics['AIC']:.2f}")
        print(f"  BIC: {final_metrics['BIC']:.2f}")
        print(f"  Log-likelihood: {final_metrics['log_likelihood']:.2f}")
        print(f"  N parameters: {final_metrics['n_params']}")
        
        # Interpret regimes
        print("\nInterpreting regimes...")
        self.regime_characteristics = self.hmm_model.interpret_regimes(
            self.combined_data,
            self.regime_states
        )
        
        for regime_id, chars in self.regime_characteristics.items():
            print(f"  Regime {regime_id}: {chars['name']}")
            print(f"    Observations: {chars['n_observations']} ({chars['pct_of_total']:.1f}%)")
        
        return self.hmm_model
    
    def generate_outputs(self):
        """Generate all outputs: plots, tables, and statistical tests."""
        print("\n" + "=" * 80)
        print("GENERATING OUTPUTS")
        print("=" * 80)
        
        if self.regime_states is None:
            raise ValueError("Must fit model first. Call fit_hmm() method.")
        
        # Get regime names
        regime_names = {
            regime_id: chars['name']
            for regime_id, chars in self.regime_characteristics.items()
        }
        
        # 1. Regime statistics
        print("\n1. Computing regime statistics...")
        regime_stats = self.results_processor.compute_regime_statistics(
            self.combined_data,
            self.regime_states,
            self.regime_characteristics
        )
        
        # 2. Statistical tests
        print("\n2. Performing statistical tests...")
        anova_results = self.results_processor.perform_erp_tests(
            self.combined_data,
            self.regime_states,
            regime_names
        )
        
        # 3. Model summary
        print("\n3. Creating model summary...")
        final_metrics = self.model_metrics.get(self.n_regimes, {})
        self.results_processor.create_summary_table(
            regime_stats,
            final_metrics,
            anova_results.get('anova', {}) if anova_results else {}
        )
        
        # 4. Plots
        print("\n4. Generating plots...")
        
        # Regime probabilities over time
        self.plotter.plot_regime_probabilities(
            self.combined_data['date'],
            self.regime_probs,
            regime_names
        )
        
        # Transition matrix
        transition_matrix = self.hmm_model.get_transition_matrix()
        self.plotter.plot_transition_matrix(
            transition_matrix,
            regime_names
        )
        
        # Regime assignments over time
        self.plotter.plot_regime_time_series(
            self.combined_data['date'],
            self.regime_states,
            regime_names
        )
        
        # Regime interpretation plots
        self.plotter.plot_regime_interpretation(
            self.combined_data,
            self.regime_states,
            self.regime_characteristics
        )
        
        # 5. Comparison with 2x2 quadrants
        if RegimeDefinitions is not None:
            print("\n5. Comparing HMM regimes vs 2x2 quadrants...")
            try:
                # Create 2x2 regime definitions
                quad_def = RegimeDefinitions(threshold_method='median')
                quadrant_regimes = quad_def.classify_dataframe(
                    self.combined_data,
                    growth_col='growth_factor',
                    inflation_col='inflation_factor'
                )
                
                # Get quadrant names
                quadrant_names = {
                    i: quad_def.get_regime_short_name(i)
                    for i in range(4)
                }
                
                # Create comparison plot
                self.plotter.plot_regime_comparison(
                    self.regime_states,
                    quadrant_regimes.values,
                    regime_names,
                    quadrant_names
                )
            except Exception as e:
                print(f"    Warning: Could not create comparison: {e}")
        else:
            print("\n5. Skipping 2x2 quadrant comparison (regime_definitions not available)")
        
        # 6. Save regime assignments
        print("\n6. Saving regime assignments...")
        assignments_df = pd.DataFrame({
            'date': self.combined_data['date'],
            'regime': self.regime_states,
            'regime_name': pd.Series(self.regime_states).map(regime_names),
            'growth_factor': self.combined_data['growth_factor'],
            'inflation_factor': self.combined_data['inflation_factor'],
            'monetary_policy_factor': self.combined_data['monetary_policy_factor'],
            'market_volatility_factor': self.combined_data['market_volatility_factor'],
            'erp': self.combined_data['erp']
        })
        
        # Add probability columns
        for i in range(self.n_regimes):
            if i in regime_names:
                prob_col_name = f'prob_R{i}'
                assignments_df[prob_col_name] = self.regime_probs[:, i]
        
        assignments_file = self.output_dir / 'regime_assignments.csv'
        assignments_df.to_csv(assignments_file, index=False)
        print(f"    Saved regime assignments to {assignments_file.name}")
        
        # Save transition matrix
        transmat_df = pd.DataFrame(
            transition_matrix,
            index=[regime_names.get(i, f'R{i}') for i in range(self.n_regimes)],
            columns=[regime_names.get(i, f'R{i}') for i in range(self.n_regimes)]
        )
        transmat_file = self.output_dir / 'transition_matrix.csv'
        transmat_df.to_csv(transmat_file)
        print(f"    Saved transition matrix to {transmat_file.name}")
        
        print("\n" + "=" * 80)
        print("OUTPUT GENERATION COMPLETE")
        print("=" * 80)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("HMM REGIME DETECTION ANALYSIS")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}\n")
        
        # Load and combine data
        self.combine_data()
        
        # Fit HMM model
        self.fit_hmm()
        
        # Generate outputs
        self.generate_outputs()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).resolve().parent
    data_dir = get_data_dir(__file__)
    output_dir = script_dir / 'results_4vars'
    
    # Verify data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Initialize analyzer
    # Set n_regimes=None to automatically select best K (2, 3, or 4)
    analyzer = HMMRegimeAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        n_regimes=None  # Will test K=2,3,4 and select best
    )
    
    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
