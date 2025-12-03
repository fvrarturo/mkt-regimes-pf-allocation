#!/usr/bin/env python3
"""
Main script for conditional regression analysis across all HMM regimes.

This script:
1. Loads all macro variables from macro_processed_full
2. Loads all HMM regime assignments (all K and variable combinations)
3. Runs conditional regressions for each regime
4. Creates visualizations
"""

import sys
from pathlib import Path

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
from conditional_regression_all_regimes import ConditionalRegressionAnalyzer
from create_coefficient_table import create_coefficient_table
import pandas as pd

# Optional imports (only if files exist)
try:
    from visualize_regression_results import create_all_visualizations
    HAS_VISUALIZE = True
except ImportError:
    HAS_VISUALIZE = False

try:
    from plot_3d_coefficients import create_3d_coefficient_plot
    HAS_3D_PLOT = True
except ImportError:
    HAS_3D_PLOT = False

try:
    from plot_top_combinations import create_top_combinations_ranking
    HAS_TOP_COMBINATIONS = True
except ImportError:
    HAS_TOP_COMBINATIONS = False


def main():
    """Main function."""
    print("="*80)
    print("CONDITIONAL REGRESSION ANALYSIS FOR ALL HMM REGIMES")
    print("="*80)
    
    base_dir = get_project_root(__file__)
    output_dir = SCRIPT_DIR / 'results'
    
    # Initialize analyzer
    analyzer = ConditionalRegressionAnalyzer(base_dir, output_dir)
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    analyzer.load_macro_variables()
    analyzer.load_erp()
    analyzer.load_hmm_regimes()
    
    # Step 2: Run regressions
    print("\n" + "="*80)
    print("STEP 2: RUNNING CONDITIONAL REGRESSIONS")
    print("="*80)
    results_df = analyzer.run_conditional_regressions()
    
    # Step 3: Save results
    print("\n" + "="*80)
    print("STEP 3: SAVING RESULTS")
    print("="*80)
    analyzer.save_results()
    
    # Step 4: Create visualizations
    print("\n" + "="*80)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("="*80)
    
    results_file = output_dir / 'conditional_regression_results_all.csv'
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        
        # Create comprehensive coefficient table
        print("\n4a. Creating comprehensive coefficient table...")
        create_coefficient_table(results_df, output_dir, significance_threshold=0.05)
    else:
        print("⚠️  Results file not found, skipping visualizations")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - conditional_regression_results_all.csv")
    print(f"  - significant_variables_summary.csv")
    print(f"  - Various visualization PNG files")


if __name__ == "__main__":
    main()

