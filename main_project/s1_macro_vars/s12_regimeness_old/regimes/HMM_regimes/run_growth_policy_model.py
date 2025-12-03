#!/usr/bin/env python3
"""
Run HMM analysis with Growth + Policy variables (Optimal Model).

This model achieves the best statistical fit (lowest BIC) and provides
clear economic interpretation linking growth fundamentals to policy-driven
valuation effects.
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
SECTION_DIR = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REGIMES_DIR, SECTION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from hmm_model import HMMRegimeModel
from plotting import HMMPlotter
from results import HMMResults
from path_utils import get_data_dir


class GrowthPolicyHMMAnalyzer:
    """
    HMM analyzer using Growth + Policy variables (optimal model).
    """
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path = None,
        n_regimes: int = None
    ):
        """Initialize analyzer with Growth + Policy variables only."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'results_2vars_optimal'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_regimes = n_regimes
        self.variables = ['growth_factor', 'monetary_policy_factor']
        
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
        self.macro_data = df
        return df
    
    def load_erp(self) -> pd.DataFrame:
        """Load pre-calculated Equity Risk Premium (ERP)."""
        print("Loading ERP data...")
        
        erp_file = None
        for path in [
            self.data_dir / 'macro_processed' / 'equity_risk_pr.csv',
            self.data_dir.parent / 'main_project' / 'data' / 'macro_processed' / 'equity_risk_pr.csv'
        ]:
            if path.exists():
                erp_file = path
                break
        
        if erp_file is None:
            raise FileNotFoundError(f"ERP file not found")
        
        erp_df = pd.read_csv(erp_file)
        erp_df['date'] = pd.to_datetime(erp_df['date'])
        erp_df = erp_df.rename(columns={'ERP': 'erp'})
        erp_df['erp_volatility'] = erp_df['erp'].rolling(window=12).std()
        erp_df = erp_df.dropna(subset=['erp']).reset_index(drop=True)
        
        print(f"  Loaded ERP for {len(erp_df)} observations")
        self.erp_data = erp_df
        return erp_df
    
    def combine_data(self) -> pd.DataFrame:
        """Combine macro and ERP data."""
        print("Combining macro and ERP data...")
        print(f"  Using variables: {', '.join(self.variables)}")
        
        if self.macro_data is None:
            self.load_macro_data()
        if self.erp_data is None:
            self.load_erp()
        
        macro_data = self.macro_data.copy()
        erp_data = self.erp_data.copy()
        
        macro_data['date_month'] = pd.to_datetime(macro_data['date']).dt.to_period('M').dt.end_time
        erp_data['date_month'] = pd.to_datetime(erp_data['date']).dt.to_period('M').dt.end_time
        
        combined = pd.merge(
            macro_data,
            erp_data[['date_month', 'erp', 'erp_volatility']],
            left_on='date_month',
            right_on='date_month',
            how='inner'
        )
        combined = combined.drop(columns=['date_month'])
        combined = combined.sort_values('date').reset_index(drop=True)
        
        # Drop rows with missing variables
        combined = combined.dropna(subset=self.variables).reset_index(drop=True)
        
        print(f"  Combined dataset: {len(combined)} observations")
        self.combined_data = combined
        return combined
    
    def select_best_k(self, k_values: List[int] = [2, 3, 4]) -> int:
        """Test different K values and select best based on BIC."""
        print("\n" + "=" * 80)
        print("MODEL SELECTION: Testing different K values")
        print(f"Variables: {', '.join(self.variables)}")
        print("=" * 80)
        
        if self.combined_data is None:
            self.combine_data()
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(self.combined_data[self.variables].values)
        
        for k in k_values:
            print(f"\nTesting K = {k}...")
            model = HMMRegimeModel(n_regimes=k)
            model.scaler = scaler
            
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
        
        if len(self.model_metrics) == 0:
            raise RuntimeError("Failed to fit any models!")
        
        best_k = min(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['BIC'])
        
        print(f"\n{'='*80}")
        print(f"Best K = {best_k} (lowest BIC: {self.model_metrics[best_k]['BIC']:.2f})")
        print(f"{'='*80}\n")
        
        self.results_processor.compare_model_selection(self.model_metrics)
        return best_k
    
    def fit_hmm(self) -> HMMRegimeModel:
        """Fit HMM model with selected or specified K."""
        print("\n" + "=" * 80)
        print("FITTING HMM MODEL (Growth + Policy - Optimal)")
        print("=" * 80)
        
        if self.combined_data is None:
            self.combine_data()
        
        if self.n_regimes is None:
            self.n_regimes = self.select_best_k()
        
        print(f"\nFitting final HMM model with K = {self.n_regimes}...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(self.combined_data[self.variables].values)
        
        self.hmm_model = HMMRegimeModel(n_regimes=self.n_regimes)
        self.hmm_model.scaler = scaler
        self.hmm_model.fit(features, n_init=10)
        
        self.regime_states = self.hmm_model.predict(features)
        self.regime_probs = self.hmm_model.predict_proba(features)
        
        final_metrics = self.hmm_model.calculate_model_metrics(features)
        self.model_metrics[self.n_regimes] = final_metrics
        
        print(f"\nModel Metrics:")
        print(f"  AIC: {final_metrics['AIC']:.2f}")
        print(f"  BIC: {final_metrics['BIC']:.2f}")
        print(f"  Log-likelihood: {final_metrics['log_likelihood']:.2f}")
        
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
        """Generate all outputs."""
        print("\n" + "=" * 80)
        print("GENERATING OUTPUTS")
        print("=" * 80)
        
        if self.regime_states is None:
            raise ValueError("Must fit model first.")
        
        regime_names = {
            regime_id: chars['name']
            for regime_id, chars in self.regime_characteristics.items()
        }
        
        # Regime statistics
        print("\n1. Computing regime statistics...")
        regime_stats = self.results_processor.compute_regime_statistics(
            self.combined_data,
            self.regime_states,
            self.regime_characteristics
        )
        
        # Statistical tests
        print("\n2. Performing statistical tests...")
        anova_results = self.results_processor.perform_erp_tests(
            self.combined_data,
            self.regime_states,
            regime_names
        )
        
        # Model summary
        print("\n3. Creating model summary...")
        final_metrics = self.model_metrics.get(self.n_regimes, {})
        self.results_processor.create_summary_table(
            regime_stats,
            final_metrics,
            anova_results.get('anova', {}) if anova_results else {}
        )
        
        # Plots
        print("\n4. Generating plots...")
        self.plotter.plot_regime_probabilities(
            self.combined_data['date'],
            self.regime_probs,
            regime_names
        )
        
        transition_matrix = self.hmm_model.get_transition_matrix()
        self.plotter.plot_transition_matrix(transition_matrix, regime_names)
        self.plotter.plot_regime_time_series(
            self.combined_data['date'],
            self.regime_states,
            regime_names
        )
        self.plotter.plot_regime_interpretation(
            self.combined_data,
            self.regime_states,
            self.regime_characteristics
        )
        
        # Save assignments
        print("\n5. Saving regime assignments...")
        assignments_df = pd.DataFrame({
            'date': self.combined_data['date'],
            'regime': self.regime_states,
            'regime_name': pd.Series(self.regime_states).map(regime_names),
            **{var: self.combined_data[var] for var in self.variables},
            'erp': self.combined_data['erp']
        })
        
        for i in range(self.n_regimes):
            if i in regime_names:
                assignments_df[f'prob_R{i}'] = self.regime_probs[:, i]
        
        assignments_df.to_csv(self.output_dir / 'regime_assignments.csv', index=False)
        
        transmat_df = pd.DataFrame(
            transition_matrix,
            index=[regime_names.get(i, f'R{i}') for i in range(self.n_regimes)],
            columns=[regime_names.get(i, f'R{i}') for i in range(self.n_regimes)]
        )
        transmat_df.to_csv(self.output_dir / 'transition_matrix.csv')
        
        print("\n" + "=" * 80)
        print("OUTPUT GENERATION COMPLETE")
        print("=" * 80)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("HMM REGIME DETECTION - GROWTH + POLICY MODEL (OPTIMAL)")
        print("Variables: Growth + Policy (best statistical fit)")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}\n")
        
        self.combine_data()
        self.fit_hmm()
        self.generate_outputs()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main execution function."""
    script_dir = Path(__file__).resolve().parent
    data_dir = get_data_dir(__file__)
    output_dir = script_dir / 'results_2vars_optimal'
    
    analyzer = GrowthPolicyHMMAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        n_regimes=None
    )
    
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
