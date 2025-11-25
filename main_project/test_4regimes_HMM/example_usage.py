"""
Example usage of the HMM Regime Detection system.

This script demonstrates how to use the RegimeDetectionHMM class to:
1. Load macro and sentiment data
2. Combine them with specified weights
3. Detect regimes using HMM
4. Analyze results
"""

from pathlib import Path
from regime_detection_hmm import RegimeDetectionHMM
import pandas as pd

def main():
    """Example usage of the regime detection system."""
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    macro_dir = project_root / 'data' / 'macro_processed' / 'selection'
    sentiment_path = project_root / 'data' / 'news_data' / 'sentiment_scores.csv'
    output_dir = Path(__file__).parent / 'results'
    
    print("=" * 80)
    print("HMM REGIME DETECTION - EXAMPLE USAGE")
    print("=" * 80)
    
    # Initialize detector with custom weights
    detector = RegimeDetectionHMM(
        macro_dir=macro_dir,
        sentiment_path=sentiment_path,
        macro_weight=0.4,      # 40% weight for macro factors
        sentiment_weight=0.6,  # 60% weight for sentiment
        n_regimes=4            # 4 regimes: High/Low Growth × High/Low Inflation
    )
    
    # Run full analysis
    print("\nRunning full analysis...")
    results = detector.run_full_analysis(output_dir=output_dir)
    
    # Access results
    print("\n" + "=" * 80)
    print("ACCESSING RESULTS")
    print("=" * 80)
    
    # Model metrics
    print("\nModel Metrics:")
    print(f"  AIC: {results['model_metrics']['AIC']:.2f}")
    print(f"  BIC: {results['model_metrics']['BIC']:.2f}")
    print(f"  Log-likelihood: {results['model_metrics']['log_likelihood']:.2f}")
    print(f"  Number of parameters: {results['model_metrics']['n_params']}")
    
    # Validation results
    print("\nValidation Results:")
    val_results = results['validation_results']
    print(f"  Average train score: {np.mean(val_results['train_scores']):.2f}")
    print(f"  Average test score: {np.mean(val_results['test_scores']):.2f}")
    print(f"  Average train BIC: {np.mean(val_results['train_bics']):.2f}")
    print(f"  Average test BIC: {np.mean(val_results['test_bics']):.2f}")
    
    # Regime characteristics
    print("\nRegime Characteristics:")
    for regime, chars in results['regime_characteristics'].items():
        print(f"\n  Regime {regime}: {chars['name']}")
        print(f"    Observations: {chars['n_observations']} ({chars['pct_of_total']:.1f}%)")
        print(f"    Average Growth: {chars['avg_growth']:.4f}")
        print(f"    Average Inflation: {chars['avg_inflation']:.4f}")
        print(f"    Date range: {chars['date_range'][0]} to {chars['date_range'][1]}")
    
    # Transition matrix
    print("\nTransition Matrix (probability of switching from row to column):")
    transmat = pd.DataFrame(
        results['transition_matrix'],
        index=[f'Regime {i}' for i in range(detector.n_regimes)],
        columns=[f'Regime {i}' for i in range(detector.n_regimes)]
    )
    print(transmat.round(3))
    
    # Load regime assignments
    print("\n" + "=" * 80)
    print("LOADING SAVED RESULTS")
    print("=" * 80)
    
    assignments_file = output_dir / 'regime_assignments.csv'
    if assignments_file.exists():
        assignments = pd.read_csv(assignments_file)
        assignments['date'] = pd.to_datetime(assignments['date'])
        
        print(f"\nLoaded {len(assignments)} regime assignments")
        print("\nFirst 10 rows:")
        print(assignments.head(10))
        
        print("\nLast 10 rows:")
        print(assignments.tail(10))
        
        print("\nRegime distribution:")
        print(assignments['regime'].value_counts().sort_index())
    
    print(f"\n{'=' * 80}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import numpy as np
    main()

