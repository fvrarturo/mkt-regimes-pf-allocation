"""
Example usage of HMM Variable Selection

This script demonstrates how to use the HMMVariableSelector class
with custom parameters and configurations.
"""

from pathlib import Path
from hmm_variable_selection import HMMVariableSelector, get_default_economic_events

def main():
    """Example usage."""
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data' / 'macro_processed'
    
    # Try to find sentiment data (optional)
    sentiment_path = project_dir / 'initial_test' / 'llm_text' / 'sentiment_scores.csv'
    if not sentiment_path.exists():
        print("Note: sentiment_scores.csv not found. Running with macro data only.")
        sentiment_path = None
    
    # Output directory
    output_dir = script_dir / 'results'
    
    # Example 1: Basic usage with default parameters
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    selector = HMMVariableSelector(
        data_dir=data_dir,
        sentiment_path=sentiment_path,
        n_regimes=3,          # Detect 3 regimes (e.g., expansion, recession, recovery)
        min_vars=2,           # At least 2 variables
        max_vars=6,           # At most 6 variables
        max_combinations=50   # Test up to 50 combinations
    )
    
    # Get default economic events
    economic_events = get_default_economic_events()
    
    # Run analysis
    selector.run_full_analysis(
        output_dir=output_dir,
        economic_events=economic_events
    )
    
    # Example 2: Custom configuration for more regimes
    print("\n" + "="*80)
    print("Example 2: Custom Configuration (4 Regimes)")
    print("="*80)
    
    selector2 = HMMVariableSelector(
        data_dir=data_dir,
        sentiment_path=sentiment_path,
        n_regimes=4,          # 4 regimes (e.g., expansion, slowdown, recession, recovery)
        min_vars=3,
        max_vars=8,
        max_combinations=100
    )
    
    # Load data
    selector2.load_macro_data()
    selector2.load_sentiment_data()
    selector2.prepare_combined_data()
    
    # Select best variables
    best_vars, best_result = selector2.select_best_variables()
    
    # Print summary
    print(f"\nBest variables: {', '.join(best_vars)}")
    print(f"BIC: {best_result['bic']:.2f}")
    print(f"Silhouette: {best_result['silhouette']:.3f}")
    
    # Example 3: Access results programmatically
    print("\n" + "="*80)
    print("Example 3: Accessing Results")
    print("="*80)
    
    results_df = selector2.create_results_table()
    print(f"\nTop 5 models by BIC:")
    print(results_df.head(5)[['variables', 'n_variables', 'bic', 'aic', 'silhouette']])
    
    # Get best model details
    print(f"\nBest model details:")
    print(f"  Variables: {', '.join(best_result['variables'])}")
    print(f"  Number of variables: {best_result['n_variables']}")
    print(f"  BIC: {best_result['bic']:.2f}")
    print(f"  AIC: {best_result['aic']:.2f}")
    print(f"  Log-likelihood: {best_result['log_likelihood']:.2f}")
    print(f"  Silhouette score: {best_result['silhouette']:.3f}")
    print(f"  Regime stability: {best_result['stability']:.2f}")
    print(f"  Regime separation: {best_result['separation']:.3f}")
    print(f"  Number of samples: {best_result['n_samples']}")


if __name__ == '__main__':
    main()

