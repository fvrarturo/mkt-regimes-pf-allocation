"""
Main Entry Point for Expanding Window Regime-Conditional Regression Analysis

This script orchestrates the complete analysis:
1. Detects regimes using expanding windows (no look-ahead bias)
2. Runs regime-conditional regressions using these regime assignments
3. Creates comprehensive visualizations and summaries
4. Saves all results to results/
"""

import sys
import pandas as pd
from pathlib import Path

# Add Section 1 root to sys.path for shared helpers
SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parents[2]
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

from path_utils import get_project_root

# Add current directory and subdirectories to path
base_path = SCRIPT_DIR
regression_path = base_path / 'regression'
regimes_path = base_path / 'regimes'

# Add paths in specific order to avoid conflicts
for path in (regression_path, regimes_path, base_path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Import from specific modules using full paths to avoid conflicts
import importlib.util

# Import regime detection
spec_regime = importlib.util.spec_from_file_location(
    "regime_detection_expanding_window",
    regimes_path / "regime_detection_expanding_window.py"
)
regime_module = importlib.util.module_from_spec(spec_regime)
spec_regime.loader.exec_module(regime_module)
ExpandingWindowRegimeDetector = regime_module.ExpandingWindowRegimeDetector

# Import regression modules
spec_regression = importlib.util.spec_from_file_location(
    "conditional_regression",
    regression_path / "conditional_regression.py"
)
regression_module = importlib.util.module_from_spec(spec_regression)
spec_regression.loader.exec_module(regression_module)
RegimeConditionalRegressor = regression_module.RegimeConditionalRegressor

spec_plotting = importlib.util.spec_from_file_location(
    "plotting",
    regression_path / "plotting.py"
)
plotting_module = importlib.util.module_from_spec(spec_plotting)
spec_plotting.loader.exec_module(plotting_module)
create_all_plots = plotting_module.create_all_plots

spec_summary = importlib.util.spec_from_file_location(
    "summary",
    regression_path / "summary.py"
)
summary_module = importlib.util.module_from_spec(spec_summary)
spec_summary.loader.exec_module(summary_module)
create_summary = summary_module.create_summary
create_statistics_summary = summary_module.create_statistics_summary


def main():
    """Run complete analysis with expanding window regime detection."""
    print("="*80)
    print("COMPLETE ANALYSIS: NO LOOK-AHEAD BIAS")
    print("="*80)
    print("\nThis analysis uses expanding window regime detection to avoid look-ahead bias.")
    print("For each time t, regimes are detected using only data from start to t.\n")
    
    # Set up paths
    script_dir = SCRIPT_DIR
    base_dir = get_project_root(__file__)
    
    # Run analysis for Hard Threshold method only (Mahalanobis results already in results_mahalanobis/)
    analyses = [
        {
            'name': 'Hard Thresholds',
            'use_mahalanobis': False,
            'use_probabilities': False,
            'results_folder': 'results_hard_threshold_new',
            'regime_folder': 'results_hard_threshold_new/regime_assignments'
        }
    ]
    
    for analysis_config in analyses:
        print("\n" + "="*80)
        print(f"ANALYSIS: {analysis_config['name'].upper()}")
        print("="*80)
        
        # Step 1: Detect regimes using expanding windows
        print("\n" + "="*80)
        print("STEP 1: EXPANDING WINDOW REGIME DETECTION")
        print("="*80)
        
        models = ['hmm_optimal', '2x2']
        
        for model in models:
            print(f"\n{'='*80}")
            print(f"Detecting regimes: {model.upper()} ({analysis_config['name']})")
            print(f"{'='*80}")
            
            # Set output directory for regime assignments
            regime_output_dir = base_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_expanding_window' / analysis_config['regime_folder'] / model
            
            detector = ExpandingWindowRegimeDetector(
                data_dir=base_dir,
                regime_model=model,
                min_window_size=24,  # 2 years minimum
                output_dir=regime_output_dir
            )
            
            # Run detection (only 2x2 uses use_mahalanobis parameter)
            if model == '2x2':
                detector.run_detection(use_mahalanobis=analysis_config['use_mahalanobis'])
            else:
                detector.run_detection()  # HMM doesn't use Mahalanobis
            
            # Save results
            detector.save_regime_assignments()
            
            print(f"\n✓ Completed regime detection for {model}")
        
        # Step 2: Run regression analysis with expanding window regimes
        print("\n" + "="*80)
        print("STEP 2: REGIME-CONDITIONAL REGRESSION ANALYSIS")
        print("="*80)
        print(f"\nUsing expanding window regime assignments ({analysis_config['name']})\n")
        
        for model in models:
            print(f"\n{'='*80}")
            print(f"Analyzing: {model.upper()} ({analysis_config['name']})")
            print(f"{'='*80}")
            
            # Set output directory for regression results
            regression_output_dir = base_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_expanding_window' / analysis_config['results_folder'] / 'regression_conditional' / model
            
            # Initialize regressor with expanding window flag
            regressor = RegimeConditionalRegressor(
                data_dir=base_dir,
                regime_model=model,
                use_expanding_window=True,  # Use expanding window regimes
                output_dir=regression_output_dir
            )
            
            # Load data - need to specify the correct regime folder
            regressor.load_erp()
            
            # Override the regime path to use the correct folder
            regime_path = base_dir / 's1_macro_vars' / 's12_regimeness' / 'regressions_expanding_window' / analysis_config['regime_folder'] / model / 'regime_assignments.csv'
            if not regime_path.exists():
                raise FileNotFoundError(f"Regime file not found: {regime_path}")
            
            regime_df = pd.read_csv(regime_path, parse_dates=['date'])
            regime_df = regime_df.set_index('date').sort_index()
            
            # Identify probability columns
            prob_cols = [col for col in regime_df.columns if 'prob' in col.lower() or col.startswith('prob_')]
            if prob_cols:
                regressor.regime_prob_cols = {int(col.split('_')[-1].replace('R', '')): col for col in prob_cols if 'R' in col}
            
            regressor.regime_data = regime_df
            print(f"  Loaded {len(regime_df)} regime observations from {regime_path}")
            print(f"  Regimes: {sorted(regime_df['regime'].unique())}")
            if hasattr(regressor, 'regime_prob_cols'):
                print(f"  Probability columns: {list(regressor.regime_prob_cols.values())}")
            
            regressor.load_macro_variables()
            regressor.combine_data()
            
            # Run regressions
            regressor.run_all_regressions(
                horizons=[1, 3, 6, 12], 
                use_probabilities=analysis_config['use_probabilities']
            )
            
            # Create tables and tests
            regressor.create_coefficient_tables()
            regressor.test_coefficient_differences()
            
            # Save results
            regressor.save_results()
            
            # Create visualizations
            create_all_plots(regressor)
            
            # Create summaries (statistics_summary now includes all summary functions)
            create_summary(regressor)
            create_statistics_summary(regressor)  # This now creates all summary files
            
            print(f"\n✓ Analysis complete for {model}!")
            print(f"  Results saved to: {regressor.output_dir}")
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE: {analysis_config['name'].upper()}")
        print(f"{'='*80}")
        print(f"\nResults saved to: {analysis_config['results_folder']}/")
    
    print("\n" + "="*80)
    print("HARD THRESHOLD ANALYSIS COMPLETE (NO LOOK-AHEAD BIAS)!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results_hard_threshold_new/ (Hard threshold method - above/below median)")
    print("\nNote: Mahalanobis results remain in results_mahalanobis/ (unchanged)")
    print("\nKey differences from full-sample analysis:")
    print("  - Regime detection uses only past data (expanding windows)")
    print("  - No look-ahead bias in regime assignments")
    print("  - Results are truly predictive (not just descriptive)")
    print("  - Hard threshold: Simple above/below median classification (no Mahalanobis distance)")
    print("  - Hard assignments: Only observations with exact regime match are used")
    print("\nGenerated outputs:")
    print("  - Regression results (CSV)")
    print("  - Coefficient tables (CSV)")
    print("  - Statistical tests (CSV)")
    print("  - Comprehensive visualizations (PNG)")
    print("  - Summary reports (MD, CSV)")


if __name__ == "__main__":
    main()
