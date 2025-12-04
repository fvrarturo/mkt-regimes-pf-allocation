"""
Master orchestration script for Section 2: Macro Forecasting

This script runs the complete forecasting pipeline to predict Growth (Industrial Production)
and Inflation using multiple models:
1. TVP-VAR (Time-Varying Parameter VAR)
2. XGBoost (Macro-only and Macro+Sentiment)
3. LSTM (Long Short-Term Memory)
4. MIDAS TVP-VAR (with daily oil prices)
5. Cross-model comparison and evaluation

As per goals.md, the goal is to forecast Growth and Inflation at horizons h ∈ {1, 3, 6} months
and compare model performance using RMSE, MAE, and statistical tests.

Execution order:
1. TVP-VAR forecasting (baseline)
2. XGBoost forecasting (macro-only and macro+sentiment)
3. LSTM forecasting
4. MIDAS TVP-VAR forecasting
5. Cross-model comparison and evaluation
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

# Try to import path_utils if available
try:
    from path_utils import get_project_root
except ImportError:
    # Fallback if path_utils not available
    def get_project_root(file_path):
        """Get project root by finding main_project directory."""
        path = Path(file_path).resolve()
        while path.parent != path:
            if path.name == "main_project":
                return path.parent
            path = path.parent
        # Fallback: assume we're in main_project/s2_forecasts
        return Path(__file__).resolve().parent.parent.parent


def run_script(
    script_path: Path, 
    description: str, 
    cwd: Optional[Path] = None, 
    env: Optional[dict] = None
) -> bool:
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
        try:
            section_dir = get_project_root(script_path) / "s2_forecasts"
            pythonpath = script_env.get("PYTHONPATH", "")
            if pythonpath:
                pythonpath = f"{section_dir}{os.pathsep}{pythonpath}"
            else:
                pythonpath = str(section_dir)
            script_env["PYTHONPATH"] = pythonpath
        except Exception:
            # If path_utils fails, just use current directory
            pass
        
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
    print("SECTION 2: MACRO FORECASTING - MASTER PIPELINE")
    print("="*80)
    print("\nThis script runs the complete forecasting pipeline to predict")
    print("Growth (Industrial Production) and Inflation using multiple models.")
    print("\nForecast Models:")
    print("  1. TVP-VAR (Time-Varying Parameter VAR)")
    print("  2. XGBoost (Macro-only and Macro+Sentiment)")
    print("  3. LSTM (Long Short-Term Memory)")
    print("  4. MIDAS TVP-VAR (with daily oil prices)")
    print("  5. Cross-model comparison and evaluation")
    print("="*80)
    
    # Get section directory - this script is in main_project/s2_forecasts/main.py
    script_path = Path(__file__).resolve()
    # Script is at: .../main_project/s2_forecasts/main.py
    # So section_dir is: .../main_project/s2_forecasts
    section_dir = script_path.parent
    
    # Base dir is main_project (parent of section_dir)
    base_dir = section_dir.parent
    
    print(f"\nProject root: {base_dir}")
    print(f"Section directory: {section_dir}")
    
    # Track success/failure
    results = {}
    
    # ========================================================================
    # STEP 1: TVP-VAR Forecasting
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 1: TVP-VAR FORECASTING")
    print("#"*80)
    print("\nBaseline model: 4-variable TVP-VAR (growth, inflation, policy, volatility)")
    print("Forecast horizons: 1, 3, 6 months")
    
    tvpvar_main = section_dir / "s21_macro" / "main.py"
    results['tvpvar'] = run_script(
        tvpvar_main,
        "TVP-VAR Forecasting (Baseline)",
        cwd=section_dir
    )
    
    if not results['tvpvar']:
        print("\n⚠️  Warning: TVP-VAR forecasting failed. Continuing with other models...")
    
    # ========================================================================
    # STEP 2: XGBoost Forecasting
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 2: XGBOOST FORECASTING")
    print("#"*80)
    print("\nGradient boosting models:")
    print("  - Macro-only features")
    print("  - Macro + Sentiment features")
    print("Forecast horizons: 1, 3, 6 months")
    
    xgboost_main = section_dir / "s22_ml_based" / "main.py"
    results['xgboost'] = run_script(
        xgboost_main,
        "XGBoost Forecasting (Macro-only and Macro+Sentiment)",
        cwd=section_dir
    )
    
    if not results['xgboost']:
        print("\n⚠️  Warning: XGBoost forecasting failed. Continuing with other models...")
    
    # ========================================================================
    # STEP 3: LSTM Forecasting
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 3: LSTM FORECASTING")
    print("#"*80)
    print("\nLong Short-Term Memory neural network:")
    print("  - Multivariate sequence model")
    print("  - Joint prediction of Growth and Inflation")
    print("Forecast horizons: 1, 3, 6 months")
    
    lstm_main = section_dir / "s22_ml_based" / "lstm_main.py"
    results['lstm'] = run_script(
        lstm_main,
        "LSTM Forecasting (Neural Network)",
        cwd=section_dir
    )
    
    if not results['lstm']:
        print("\n⚠️  Warning: LSTM forecasting failed. Continuing with other models...")
    
    # ========================================================================
    # STEP 4: MIDAS TVP-VAR Forecasting
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 4: MIDAS TVP-VAR FORECASTING")
    print("#"*80)
    print("\nMIDAS-augmented TVP-VAR:")
    print("  - Combines monthly macro factors with daily oil prices")
    print("  - Exponential aggregation of high-frequency data")
    print("Forecast horizons: 1, 3, 6 months")
    
    midas_main = section_dir / "s23_Midas" / "main_midas.py"
    results['midas'] = run_script(
        midas_main,
        "MIDAS TVP-VAR Forecasting (with Oil Prices)",
        cwd=section_dir
    )
    
    if not results['midas']:
        print("\n⚠️  Warning: MIDAS TVP-VAR forecasting failed. Continuing with comparison...")
    
    # ========================================================================
    # STEP 5: Cross-Model Comparison
    # ========================================================================
    print("\n\n" + "#"*80)
    print("# STEP 5: CROSS-MODEL COMPARISON")
    print("#"*80)
    print("\nThis step compares all models and generates:")
    print("  - Performance comparison tables (RMSE, MAE)")
    print("  - Statistical tests (Diebold-Mariano)")
    print("  - Visualizations and summary reports")
    
    comparison_main = section_dir / "cross_comparison" / "main.py"
    results['comparison'] = run_script(
        comparison_main,
        "Cross-Model Comparison and Evaluation",
        cwd=section_dir
    )
    
    if not results['comparison']:
        print("\n⚠️  Warning: Cross-model comparison failed.")
        print("Individual model results are still available.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    print("\nModel Results:")
    model_names = {
        'tvpvar': 'TVP-VAR',
        'xgboost': 'XGBoost',
        'lstm': 'LSTM',
        'midas': 'MIDAS TVP-VAR',
        'comparison': 'Cross-Model Comparison'
    }
    
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        model_name = model_names.get(component, component)
        print(f"  {model_name:30s}: {status}")
    
    all_success = all(results.values())
    at_least_one_model = any([
        results.get('tvpvar', False),
        results.get('xgboost', False),
        results.get('lstm', False),
        results.get('midas', False)
    ])
    
    if all_success:
        print("\n" + "="*80)
        print("✓ ALL FORECASTING MODELS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey outputs generated:")
        
        print("\n1. TVP-VAR Results:")
        print("   - s21_macro/results/growth_forecast_metrics.csv")
        print("   - s21_macro/results/inflation_forecast_metrics.csv")
        print("   - s21_macro/results/forecast_performance_table.csv")
        
        print("\n2. XGBoost Results:")
        print("   - s22_ml_based/results/xgboost/growth_factor_metrics_xgboost.csv")
        print("   - s22_ml_based/results/xgboost/inflation_factor_metrics_xgboost.csv")
        print("   - s22_ml_based/results/xgboost/feature_importance_*.png")
        
        print("\n3. LSTM Results:")
        print("   - s22_ml_based/results/lstm/growth_factor_metrics_lstm.csv")
        print("   - s22_ml_based/results/lstm/inflation_factor_metrics_lstm.csv")
        print("   - s22_ml_based/results/lstm/learning_curve_*.png")
        
        print("\n4. MIDAS TVP-VAR Results:")
        print("   - s23_Midas/results_midas/growth_forecast_metrics.csv")
        print("   - s23_Midas/results_midas/inflation_forecast_metrics.csv")
        
        print("\n5. Cross-Model Comparison:")
        print("   - cross_comparison/results/performance_comparison_table.csv")
        print("   - cross_comparison/results/model_comparison_report.md")
        print("   - cross_comparison/results/performance_*.png")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("="*80)
        print("1. Review model_comparison_report.md for best model identification")
        print("2. Check performance_comparison_table.csv for RMSE/MAE by horizon")
        print("3. Review Diebold-Mariano test results for statistical significance")
        print("4. Use best-performing model for trading strategy evaluation (Section 3)")
        print("="*80)
    elif at_least_one_model:
        print("\n" + "="*80)
        print("⚠️  PARTIAL SUCCESS - SOME MODELS COMPLETED")
        print("="*80)
        print("\nAt least one forecasting model completed successfully.")
        print("Review individual model results in their respective result directories.")
        print("\nFailed components:")
        for component, success in results.items():
            if not success:
                model_name = model_names.get(component, component)
                print(f"  - {model_name}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ ALL FORECASTING MODELS FAILED")
        print("="*80)
        print("\nPlease review the errors above and fix issues before proceeding.")
        print("The forecasting pipeline requires at least one model to complete")
        print("successfully for trading strategy evaluation.")
        print("="*80)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

