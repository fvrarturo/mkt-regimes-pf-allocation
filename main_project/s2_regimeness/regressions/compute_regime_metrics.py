"""
Compute R², adjusted R², and RMSE for each regime in 2x2 and HMM models.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_regime_metrics():
    """Compute metrics for each regime in both models."""
    
    results_dir = Path(__file__).parent / 'results'
    
    # Load conditional regression results
    print("Loading conditional regression results...")
    two_by_two_file = results_dir / 'conditional_regression_2x2_regimes.csv'
    hmm_file = results_dir / 'conditional_regression_hmm_inflation_market_volatility_k4.csv'
    
    two_by_two_df = pd.read_csv(two_by_two_file)
    hmm_df = pd.read_csv(hmm_file)
    
    print(f"\n2x2 regimes: {len(two_by_two_df)} rows")
    print(f"HMM regimes: {len(hmm_df)} rows")
    
    # Extract metrics for each regime
    results = []
    
    # Process 2x2 regimes
    print("\n" + "="*80)
    print("2x2 REGIMES METRICS")
    print("="*80)
    
    for regime in sorted(two_by_two_df['regime'].unique()):
        regime_data = two_by_two_df[two_by_two_df['regime'] == regime]
        
        # Get metrics (they should be the same for all variables in a regime)
        r_squared = regime_data['r_squared'].iloc[0]
        rmse = regime_data['rmse'].iloc[0]
        n_effective = regime_data['n_effective'].iloc[0]
        n_observations = regime_data['n_observations'].iloc[0]
        avg_weight = regime_data['avg_weight'].iloc[0]
        
        # Count number of variables (features)
        n_variables = len(regime_data)
        
        # Compute adjusted R²
        # adj_r² = 1 - (1 - R²) * (n - 1) / (n - k - 1)
        # where k is the number of predictors
        if n_effective > n_variables + 1:
            adj_r_squared = 1 - (1 - r_squared) * (n_effective - 1) / (n_effective - n_variables - 1)
        else:
            adj_r_squared = r_squared
        
        results.append({
            'Model': '2x2 Regimes',
            'Regime': int(regime),
            'R²': r_squared,
            'Adj R²': adj_r_squared,
            'RMSE': rmse,
            'N_effective': n_effective,
            'N_observations': n_observations,
            'N_variables': n_variables,
            'Avg_weight': avg_weight
        })
        
        print(f"\nRegime {regime}:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Adj R² = {adj_r_squared:.4f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  N_effective = {n_effective:.1f}")
        print(f"  N_observations = {n_observations}")
        print(f"  N_variables = {n_variables}")
        print(f"  Avg_weight = {avg_weight:.3f}")
    
    # Process HMM regimes
    print("\n" + "="*80)
    print("HMM INFLATION_MARKET_VOLATILITY K=4 REGIMES METRICS")
    print("="*80)
    
    for regime in sorted(hmm_df['regime'].unique()):
        regime_data = hmm_df[hmm_df['regime'] == regime]
        
        # Get metrics (they should be the same for all variables in a regime)
        r_squared = regime_data['r_squared'].iloc[0]
        rmse = regime_data['rmse'].iloc[0]
        n_effective = regime_data['n_effective'].iloc[0]
        n_observations = regime_data['n_observations'].iloc[0]
        avg_weight = regime_data['avg_weight'].iloc[0]
        
        # Count number of variables (features)
        n_variables = len(regime_data)
        
        # Compute adjusted R²
        if n_effective > n_variables + 1:
            adj_r_squared = 1 - (1 - r_squared) * (n_effective - 1) / (n_effective - n_variables - 1)
        else:
            adj_r_squared = r_squared
        
        results.append({
            'Model': 'HMM Inflation_Market_Volatility K=4',
            'Regime': int(regime),
            'R²': r_squared,
            'Adj R²': adj_r_squared,
            'RMSE': rmse,
            'N_effective': n_effective,
            'N_observations': n_observations,
            'N_variables': n_variables,
            'Avg_weight': avg_weight
        })
        
        print(f"\nRegime {regime}:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Adj R² = {adj_r_squared:.4f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  N_effective = {n_effective:.1f}")
        print(f"  N_observations = {n_observations}")
        print(f"  N_variables = {n_variables}")
        print(f"  Avg_weight = {avg_weight:.3f}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = results_dir / 'regime_metrics_summary.csv'
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved results to: {output_file}")
    
    return summary_df


if __name__ == "__main__":
    compute_regime_metrics()

