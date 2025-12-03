#!/usr/bin/env python3
"""
Quick test script - runs analysis on a subset of regimes for faster testing.
"""

import sys
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parents[2]
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

try:
    from path_utils import get_project_root
except ImportError:
    def get_project_root(file_path):
        path = Path(file_path).resolve()
        while path.parent != path:
            if path.name == "main_project":
                return path
            path = path.parent
        return Path(__file__).resolve().parent.parent.parent.parent

from conditional_regression_all_regimes import ConditionalRegressionAnalyzer
import pandas as pd

# Modify the load_hmm_regimes method to only test a subset
def test_load_hmm_regimes_quick(self):
    """Quick test version - only test 2 combinations and K=2,3,4"""
    print("\n" + "="*80)
    print("LOADING HMM REGIME ASSIGNMENTS (QUICK TEST MODE)")
    print("="*80)
    
    # Load systematic results
    results_file = (self.base_dir / 's1_macro_vars' / 's12_regimeness' / 'regimes' / 
                   'HMM_regimes' / 'results_systematic' / 'all_model_results.csv')
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    results_df = pd.read_csv(results_file)
    
    # Test ALL combinations but with fewer K values and initializations for speed
    combinations = results_df[['combination', 'variables']].drop_duplicates()
    k_values = [2, 3, 4, 5]  # Test K=2,3,4,5 (fewer than full range for speed)
    
    print(f"\nTEST MODE: Testing {len(combinations)} combinations, K={k_values}")
    
    # Load macro data
    macro_final_path = self.base_dir / 'data' / 'macro_final' / 'final_macro.csv'
    macro_final = pd.read_csv(macro_final_path, parse_dates=['date'])
    
    regime_assignments = {}
    
    for _, combo_row in combinations.iterrows():
        combo_name = combo_row['combination']
        variables_str = combo_row['variables']
        variables = [v.strip() for v in variables_str.split(',')]
        
        print(f"\nProcessing {combo_name}...")
        print(f"  Variables: {', '.join(variables)}")
        
        for k in k_values:
            try:
                feature_data = macro_final[['date'] + variables].copy()
                feature_data = feature_data.dropna()
                
                if len(feature_data) == 0:
                    continue
                
                from sklearn.preprocessing import StandardScaler
                from hmm_model import HMMRegimeModel
                
                scaler = StandardScaler()
                features = scaler.fit_transform(feature_data[variables].values)
                
                model = HMMRegimeModel(
                    n_regimes=k,
                    variables=variables,
                    random_state=42
                )
                model.scaler = scaler
                model.fit(features, n_init=3)  # Fewer initializations for speed (full uses 5)
                
                # Get regime probabilities (soft assignments)
                regime_probs = model.predict_proba(features)  # Shape: (n_samples, n_regimes)
                
                # Create DataFrame with regime probabilities
                regime_df = pd.DataFrame({
                    'date': feature_data['date'].values
                })
                
                # Add probability columns for each regime
                for regime_idx in range(k):
                    regime_df[f'prob_R{regime_idx}'] = regime_probs[:, regime_idx]
                
                regime_df['date'] = pd.to_datetime(regime_df['date'])
                regime_df = regime_df.set_index('date').sort_index()
                regime_df = regime_df.resample('ME').last()
                regime_df = regime_df.reset_index()
                
                key = (combo_name, k)
                regime_assignments[key] = regime_df
                
                print(f"    ✓ K={k}: {len(regime_df)} observations")
                
            except Exception as e:
                print(f"    ⚠️  K={k}: Error - {e}")
                continue
    
    self.regime_assignments = regime_assignments
    print(f"\n✓ Loaded {len(regime_assignments)} regime specifications (TEST MODE)")
    return regime_assignments


def main():
    """Quick test."""
    print("="*80)
    print("QUICK TEST: CONDITIONAL REGRESSION ANALYSIS")
    print("="*80)
    print("Testing with subset of regimes for faster execution")
    
    base_dir = get_project_root(__file__)
    output_dir = SCRIPT_DIR / 'results' / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = ConditionalRegressionAnalyzer(base_dir, output_dir)
    
    # Monkey patch the load_hmm_regimes method
    analyzer.load_hmm_regimes = lambda: test_load_hmm_regimes_quick(analyzer)
    
    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    analyzer.load_macro_variables()
    analyzer.load_erp()
    analyzer.load_hmm_regimes()
    
    # Run regressions
    print("\n" + "="*80)
    print("STEP 2: RUNNING CONDITIONAL REGRESSIONS")
    print("="*80)
    results_df = analyzer.run_conditional_regressions()
    
    # Save results
    print("\n" + "="*80)
    print("STEP 3: SAVING RESULTS")
    print("="*80)
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("QUICK TEST COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nSample results:")
    print(results_df.head(20))


if __name__ == "__main__":
    main()

