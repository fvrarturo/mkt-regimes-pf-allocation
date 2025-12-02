"""
Master orchestration script for Section 1: Macro Variables Analysis

This script runs the complete analysis pipeline to identify which macro variables
have the most predictability on Equity Risk Premium (ERP) using:
1. Full-sample regressions
2. Regime-dependent conditional regressions (2x2 and HMM)
3. Extremeness-based analysis

As per codex_instructions.md, the goal is to find the regime/extremeness definition
that shows the clearest patterns in terms of significance of linear regression
coefficients for macro variables.

Execution order:
1. Full-sample regression analysis (baseline)
2. Regime detection (2x2 Growth×Inflation and HMM Growth+Policy)
3. Regime model comparison
4. Expanding-window conditional regressions (no look-ahead bias)
5. Extremeness analysis (Isolation Forest and PCA Distance)
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_utils import get_project_root


def run_script(script_path: Path, description: str, cwd: Optional[Path] = None, env: Optional[dict] = None) -> bool:
    """
    Run a Python script and return True if successful.
    
    Parameters:
    -----------
    script_path : Path
        Path to the Python script to run
    description : str
        Description of what the script does (for logging)
    cwd : Path, optional
        Working directory for the script
    env : dict, optional
        Environment variables to set (will be merged with current env)
    
    Returns:
    --------
    bool
        True if script executed successfully, False otherwise
    """
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Script: {script_path}")
    
    if not script_path.exists():
        print(f"ERROR: Script not found at {script_path}")
        return False
    
    try:
        # Prepare environment
        script_env = os.environ.copy()
        if env:
            script_env.update(env)
        
        # Ensure PYTHONPATH includes section directory for path_utils imports
        section_dir = get_project_root(script_path) / "s1_macro_vars"
        pythonpath = script_env.get("PYTHONPATH", "")
        if pythonpath:
            pythonpath = f"{section_dir}{os.pathsep}{pythonpath}"
        else:
            pythonpath = str(section_dir)
        script_env["PYTHONPATH"] = pythonpath
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd) if cwd else str(script_path.parent),
            env=script_env,
            check=True,
            capture_output=False  # Show output in real-time
        )
        print(f"\n✓ Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {description}")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error running {description}")
        print(f"Error: {str(e)}")
        return False


def main():
    """Main orchestration function."""
    print("="*80)
    print("SECTION 1: MACRO VARIABLES ANALYSIS - MASTER PIPELINE")
    print("="*80)
    print("\nThis script runs the complete analysis to identify which macro variables")
    print("have the most predictability on Equity Risk Premium (ERP).")
    print("\nAnalysis components:")
    print("  1. Full-sample regressions (baseline)")
    print("  2. Regime detection (2x2 and HMM optimal)")
    print("  3. Regime model comparison")
    print("  4. Expanding-window conditional regressions")
    print("  5. Extremeness analysis")
    print("="*80)
    
    # Get project root
    base_dir = get_project_root(__file__)
    section_dir = base_dir / "s1_macro_vars"
    
    print(f"\nProject root: {base_dir}")
    print(f"Section directory: {section_dir}")
    
    # Track success/failure
    results = {}
    
    # ========================================================================
    # STEP 1: Full-Sample Regression Analysis
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 1: FULL-SAMPLE REGRESSION ANALYSIS")
    print("#"*80)
    
    full_sample_main = section_dir / "s11_full_sample" / "main.py"
    results['full_sample'] = run_script(
        full_sample_main,
        "Full-Sample Regression Analysis",
        cwd=section_dir
    )
    
    if not results['full_sample']:
        print("\n⚠️  Warning: Full-sample analysis failed. Continuing with other analyses...")
    
    # ========================================================================
    # STEP 2: Regime Detection
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 2: REGIME DETECTION")
    print("#"*80)
    
    # 2a. 2x2 Growth × Inflation Regimes
    print("\n--- 2a. 2x2 Growth × Inflation Regimes ---")
    regime_2x2_main = section_dir / "s12_regimeness" / "regimes" / "2x2_regimes" / "main.py"
    results['regime_2x2'] = run_script(
        regime_2x2_main,
        "2x2 Growth × Inflation Regime Detection",
        cwd=section_dir
    )
    
    # 2b. HMM Optimal (Growth + Policy)
    print("\n--- 2b. HMM Optimal (Growth + Policy) ---")
    hmm_optimal_main = section_dir / "s12_regimeness" / "regimes" / "HMM_regimes" / "run_growth_policy_model.py"
    results['hmm_optimal'] = run_script(
        hmm_optimal_main,
        "HMM Optimal Model (Growth + Policy)",
        cwd=section_dir
    )
    
    if not results['regime_2x2'] or not results['hmm_optimal']:
        print("\n⚠️  Warning: Some regime detection failed. Conditional regressions may not work.")
    
    # ========================================================================
    # STEP 3: Regime Model Comparison
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 3: REGIME MODEL COMPARISON")
    print("#"*80)
    
    compare_models = section_dir / "s12_regimeness" / "compare_models.py"
    results['compare_models'] = run_script(
        compare_models,
        "Compare 2x2 vs HMM Optimal Models",
        cwd=section_dir
    )
    
    if not results['compare_models']:
        print("\n⚠️  Warning: Model comparison failed. Continuing...")
    
    # ========================================================================
    # STEP 4: Expanding-Window Conditional Regressions
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 4: EXPANDING-WINDOW CONDITIONAL REGRESSIONS")
    print("#"*80)
    print("\nThis step runs regime-conditional regressions using expanding windows")
    print("to avoid look-ahead bias. This is critical for the forecasting pipeline.")
    
    expanding_window_main = section_dir / "s12_regimeness" / "regressions_expanding_window" / "main.py"
    results['expanding_window'] = run_script(
        expanding_window_main,
        "Expanding-Window Conditional Regressions",
        cwd=section_dir
    )
    
    if not results['expanding_window']:
        print("\n⚠️  Warning: Expanding-window regressions failed.")
        print("This is critical for the forecasting pipeline.")
    
    # ========================================================================
    # STEP 5: Extremeness Analysis
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 5: EXTREMENESS ANALYSIS")
    print("#"*80)
    
    extremeness_main = section_dir / "s13_extremeness" / "initial_relevance" / "main.py"
    results['extremeness'] = run_script(
        extremeness_main,
        "Extremeness Models Analysis (Isolation Forest & PCA Distance)",
        cwd=section_dir
    )
    
    if not results['extremeness']:
        print("\n⚠️  Warning: Extremeness analysis failed.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    print("\nComponent Results:")
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {component:20s}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n" + "="*80)
        print("✓ ALL ANALYSES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey outputs generated:")
        print("\n1. Full-Sample Analysis:")
        print("   - s11_full_sample/results/regression_results_all_horizons.csv")
        print("   - s11_full_sample/results/variable_importance_ranking.csv")
        
        print("\n2. Regime Analysis:")
        print("   - s12_regimeness/results/regime_comparison_summary.csv")
        print("   - s12_regimeness/regimes/2x2_regimes/results/regime_statistics.csv")
        print("   - s12_regimeness/regimes/HMM_regimes/results_2vars_optimal/regime_statistics.csv")
        print("   - s12_regimeness/regressions_expanding_window/results/.../regression_conditional/...")
        
        print("\n3. Extremeness Analysis:")
        print("   - s13_extremeness/results/statistical_tests.csv")
        print("   - s13_extremeness/results/extremeness_model_summary.csv")
        print("   - s13_extremeness/results/*_erp_statistics.csv")
        
        print("\n4. Model Comparison:")
        print("   - COMPARISON_2X2_VS_HMM_OPTIMAL.md (at repo root)")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("="*80)
        print("1. Review regime_comparison_summary.csv to identify best regime model")
        print("2. Review statistical_tests.csv to identify best extremeness model")
        print("3. Use expanding-window conditional regression results for forecasting")
        print("4. Proceed to Part 2: Forecasting macro variables")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  SOME ANALYSES FAILED")
        print("="*80)
        print("\nPlease review the errors above and fix issues before proceeding.")
        print("The pipeline requires all components to complete successfully")
        print("for accurate identification of predictive macro variables.")
        print("="*80)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

