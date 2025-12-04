"""
Diagnostic script to understand why HMM and 2x2 strategies are so similar.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "regimes" / "HMM_regimes"))
sys.path.insert(0, str(Path(__file__).parent.parent / "regimes" / "2x2_regimes"))

from hmm_model import HMMRegimeModel
from regime_definitions import RegimeDefinitions
from data_loader import load_all_macro_variables
from hmm_forecasts import load_hmm_model_and_coefficients, get_regime_probabilities
from two_by_two_forecasts import load_2x2_regime_definitions_and_coefficients, get_hard_regime_assignment

def main():
    # base_dir should be s2_regimeness (same as main.py)
    base_dir = Path(__file__).parent.parent
    
    # Load macro data
    print("Loading macro data...")
    all_macro_df = load_all_macro_variables(base_dir=base_dir)
    
    # Load HMM model and coefficients
    print("\nLoading HMM model...")
    hmm_model, hmm_coefficients, macro_df = load_hmm_model_and_coefficients(
        base_dir=base_dir
    )
    
    # Load 2x2 regime definitions
    print("Loading 2x2 regime definitions...")
    regime_def, two_by_two_coefficients, _ = load_2x2_regime_definitions_and_coefficients(
        base_dir=base_dir
    )
    
    # Get actual growth and inflation
    growth_actual = macro_df["growth_factor"]
    inflation_actual = macro_df["inflation_factor"]
    
    # Get HMM regime probabilities
    print("\nComputing HMM regime probabilities...")
    hmm_probs = get_regime_probabilities(hmm_model, growth_actual, inflation_actual)
    
    # Get 2x2 hard regime assignments
    print("Computing 2x2 regime assignments...")
    two_by_two_regimes = get_hard_regime_assignment(regime_def, growth_actual, inflation_actual)
    
    # Align dates
    common_dates = hmm_probs.index.intersection(two_by_two_regimes.index)
    hmm_probs_aligned = hmm_probs.reindex(common_dates)
    two_by_two_regimes_aligned = two_by_two_regimes.reindex(common_dates)
    
    # 1. Check HMM probability concentration
    print("\n" + "="*80)
    print("1. HMM REGIME PROBABILITY CONCENTRATION")
    print("="*80)
    
    # Find dominant regime for each period
    dominant_regimes = hmm_probs_aligned.idxmax(axis=1).str.replace('prob_R', '').astype(int)
    
    # Check how often probabilities are highly concentrated (>0.8 in one regime)
    high_concentration = (hmm_probs_aligned.max(axis=1) > 0.8).sum()
    print(f"\nPeriods with >80% probability in one regime: {high_concentration} / {len(hmm_probs_aligned)} ({100*high_concentration/len(hmm_probs_aligned):.1f}%)")
    
    # Average max probability
    avg_max_prob = hmm_probs_aligned.max(axis=1).mean()
    print(f"Average maximum probability: {avg_max_prob:.3f}")
    
    # Distribution of dominant regimes
    print("\nDistribution of dominant HMM regimes:")
    print(dominant_regimes.value_counts().sort_index())
    
    # 2. Compare HMM dominant regime vs 2x2 regime
    print("\n" + "="*80)
    print("2. HMM DOMINANT REGIME vs 2x2 REGIME ALIGNMENT")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'hmm_dominant': dominant_regimes,
        '2x2_regime': two_by_two_regimes_aligned
    })
    
    matches = (comparison_df['hmm_dominant'] == comparison_df['2x2_regime']).sum()
    print(f"\nPeriods where HMM dominant regime matches 2x2 regime: {matches} / {len(comparison_df)} ({100*matches/len(comparison_df):.1f}%)")
    
    print("\nCross-tabulation:")
    print(pd.crosstab(comparison_df['hmm_dominant'], comparison_df['2x2_regime'], margins=True))
    
    # 3. Compare coefficients across regimes
    print("\n" + "="*80)
    print("3. COEFFICIENT SIMILARITY ACROSS REGIMES")
    print("="*80)
    
    # Get common variables
    all_vars = set()
    for regime_coefs in hmm_coefficients.values():
        all_vars.update(regime_coefs.keys())
    
    common_vars = sorted([v for v in all_vars if v not in ['growth_factor', 'inflation_factor']])
    
    print(f"\nComparing coefficients for {len(common_vars)} variables across regimes...")
    
    # Compute coefficient statistics
    coef_stats = []
    for var in common_vars[:10]:  # Show first 10 variables
        coefs_across_regimes = []
        for regime_id in range(4):
            coef = hmm_coefficients.get(regime_id, {}).get(var, np.nan)
            coefs_across_regimes.append(coef)
        
        if not all(np.isnan(coefs_across_regimes)):
            coefs_array = np.array(coefs_across_regimes)
            coef_stats.append({
                'variable': var,
                'mean': np.nanmean(coefs_array),
                'std': np.nanstd(coefs_array),
                'min': np.nanmin(coefs_array),
                'max': np.nanmax(coefs_array),
                'range': np.nanmax(coefs_array) - np.nanmin(coefs_array)
            })
    
    coef_df = pd.DataFrame(coef_stats)
    print("\nCoefficient statistics (first 10 variables):")
    print(coef_df.to_string(index=False))
    
    avg_std = coef_df['std'].mean()
    avg_range = coef_df['range'].mean()
    print(f"\nAverage coefficient std across regimes: {avg_std:.4f}")
    print(f"Average coefficient range across regimes: {avg_range:.4f}")
    
    # 4. Compute weighted coefficients for HMM vs 2x2
    print("\n" + "="*80)
    print("4. WEIGHTED COEFFICIENT COMPARISON")
    print("="*80)
    
    # Sample a few dates and compare weighted coefficients
    sample_dates = common_dates[::50]  # Every 50th date
    
    print(f"\nComparing weighted coefficients for {len(sample_dates)} sample dates...")
    
    differences = []
    for date in sample_dates[:10]:  # First 10 sample dates
        # HMM weighted coefficients
        hmm_probs_row = hmm_probs_aligned.loc[date]
        hmm_weighted_coefs = {}
        for var in common_vars:
            weighted_coef = 0.0
            for regime_id in range(4):
                prob = hmm_probs_row[f"prob_R{regime_id}"]
                coef = hmm_coefficients.get(regime_id, {}).get(var, 0.0)
                weighted_coef += prob * coef
            hmm_weighted_coefs[var] = weighted_coef
        
        # 2x2 coefficients (hard regime)
        two_by_two_regime = int(two_by_two_regimes_aligned.loc[date])
        two_by_two_coefs = {}
        for var in common_vars:
            coef = two_by_two_coefficients.get(two_by_two_regime, {}).get(var, 0.0)
            two_by_two_coefs[var] = coef
        
        # Compute difference
        for var in common_vars[:5]:  # First 5 variables
            diff = abs(hmm_weighted_coefs[var] - two_by_two_coefs[var])
            differences.append({
                'date': date,
                'variable': var,
                'hmm_coef': hmm_weighted_coefs[var],
                '2x2_coef': two_by_two_coefs[var],
                'difference': diff
            })
    
    diff_df = pd.DataFrame(differences)
    print("\nSample coefficient differences:")
    print(diff_df.groupby('variable')['difference'].agg(['mean', 'std', 'max']).round(4))
    
    avg_diff = diff_df['difference'].mean()
    print(f"\nAverage absolute difference in coefficients: {avg_diff:.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nIf strategies are almost identical, likely reasons:")
    print("1. HMM probabilities are highly concentrated (one regime dominates)")
    print("2. Coefficients are similar across regimes")
    print("3. 2x2 hard thresholds align with HMM probability concentrations")
    print(f"\nEvidence:")
    print(f"- Average max HMM probability: {avg_max_prob:.3f}")
    print(f"- HMM-2x2 regime match rate: {100*matches/len(comparison_df):.1f}%")
    print(f"- Average coefficient std: {avg_std:.4f}")
    print(f"- Average coefficient difference: {avg_diff:.4f}")

if __name__ == "__main__":
    main()

