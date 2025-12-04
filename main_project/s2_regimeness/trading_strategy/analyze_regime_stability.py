"""
Analyze why HMM regime probabilities stay constant for extended periods.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "regimes" / "HMM_regimes"))

from hmm_model import HMMRegimeModel
from data_loader import load_all_macro_variables
from hmm_forecasts import load_hmm_model_and_coefficients, get_regime_probabilities

def main():
    base_dir = Path(__file__).parent.parent
    
    # Load macro data
    print("Loading macro data...")
    all_macro_df = load_all_macro_variables(base_dir=base_dir)
    
    # Load the best HMM model (inflation + market volatility, K=4)
    print("\nLoading best HMM model (2vars_inflation_market_volatility, K=4)...")
    hmm_model, coefficients, macro_df = load_hmm_model_and_coefficients(
        base_dir=base_dir,
        combination="2vars_inflation_market_volatility",
        k=4
    )
    
    # Get regime probabilities over time
    print("\nComputing regime probabilities over time...")
    regime_probs = get_regime_probabilities(hmm_model, all_macro_df)
    
    # Get transition matrix
    print("\nTransition Matrix:")
    transition_matrix = hmm_model.get_transition_matrix()
    print(transition_matrix)
    print("\nTransition probabilities (row = from, col = to):")
    for i in range(4):
        print(f"From R{i}: {transition_matrix[i]}")
    
    # Analyze regime probability concentration
    print("\n" + "="*80)
    print("REGIME PROBABILITY CONCENTRATION ANALYSIS")
    print("="*80)
    
    # Find dominant regime for each period
    dominant_regimes = regime_probs.idxmax(axis=1).str.replace('prob_R', '').astype(int)
    
    # Check how often probabilities are highly concentrated
    max_probs = regime_probs.max(axis=1)
    high_concentration = (max_probs > 0.8).sum()
    very_high_concentration = (max_probs > 0.9).sum()
    extreme_concentration = (max_probs > 0.95).sum()
    
    print(f"\nTotal periods: {len(regime_probs)}")
    print(f"Periods with >80% probability in one regime: {high_concentration} ({100*high_concentration/len(regime_probs):.1f}%)")
    print(f"Periods with >90% probability in one regime: {very_high_concentration} ({100*very_high_concentration/len(regime_probs):.1f}%)")
    print(f"Periods with >95% probability in one regime: {extreme_concentration} ({100*extreme_concentration/len(regime_probs):.1f}%)")
    print(f"\nAverage maximum probability: {max_probs.mean():.3f}")
    print(f"Median maximum probability: {max_probs.median():.3f}")
    print(f"Min maximum probability: {max_probs.min():.3f}")
    print(f"Max maximum probability: {max_probs.max():.3f}")
    
    # Analyze regime persistence
    print("\n" + "="*80)
    print("REGIME PERSISTENCE ANALYSIS")
    print("="*80)
    
    # Count consecutive periods in same dominant regime
    regime_changes = (dominant_regimes != dominant_regimes.shift(1)).sum()
    print(f"\nNumber of regime changes: {regime_changes}")
    print(f"Average regime duration: {len(dominant_regimes) / (regime_changes + 1):.1f} periods")
    
    # Find longest periods in each regime
    print("\nLongest consecutive periods in each regime:")
    for regime_id in range(4):
        regime_mask = (dominant_regimes == regime_id)
        # Find consecutive periods
        regime_series = regime_mask.astype(int)
        regime_groups = (regime_series != regime_series.shift()).cumsum()
        consecutive_lengths = regime_series.groupby(regime_groups).sum()
        if len(consecutive_lengths) > 0:
            max_consecutive = consecutive_lengths.max()
            print(f"  R{regime_id}: {max_consecutive} consecutive periods")
    
    # Analyze underlying macro variables
    print("\n" + "="*80)
    print("UNDERLYING MACRO VARIABLES ANALYSIS")
    print("="*80)
    
    variables = hmm_model.variables
    print(f"\nVariables used: {variables}")
    
    # Align macro variables to regime probabilities
    macro_aligned = all_macro_df[variables].reindex(regime_probs.index).dropna()
    regime_probs_aligned = regime_probs.reindex(macro_aligned.index)
    
    print(f"\nMacro variable statistics:")
    for var in variables:
        print(f"\n{var}:")
        print(f"  Mean: {macro_aligned[var].mean():.4f}")
        print(f"  Std: {macro_aligned[var].std():.4f}")
        print(f"  Min: {macro_aligned[var].min():.4f}")
        print(f"  Max: {macro_aligned[var].max():.4f}")
        print(f"  Range: {macro_aligned[var].max() - macro_aligned[var].min():.4f}")
    
    # Check if macro variables are moving slowly
    print("\n" + "="*80)
    print("MACRO VARIABLE VOLATILITY ANALYSIS")
    print("="*80)
    
    for var in variables:
        changes = macro_aligned[var].diff().abs()
        print(f"\n{var} monthly changes:")
        print(f"  Mean absolute change: {changes.mean():.4f}")
        print(f"  Std of changes: {changes.std():.4f}")
        print(f"  Periods with |change| < 0.01: {(changes < 0.01).sum()} ({(changes < 0.01).sum() / len(changes) * 100:.1f}%)")
        print(f"  Periods with |change| < 0.05: {(changes < 0.05).sum()} ({(changes < 0.05).sum() / len(changes) * 100:.1f}%)")
    
    # Analyze how macro variable changes affect regime probabilities
    print("\n" + "="*80)
    print("MACRO VARIABLE CHANGES vs REGIME PROBABILITY CHANGES")
    print("="*80)
    
    # Align macro variables to regime probabilities
    macro_aligned = all_macro_df[variables].reindex(regime_probs.index).dropna()
    regime_probs_aligned = regime_probs.reindex(macro_aligned.index)
    
    # Calculate month-to-month changes
    macro_changes = macro_aligned.diff().abs()
    prob_changes = regime_probs_aligned.diff().abs()
    
    print("\nCorrelation between macro variable changes and regime probability changes:")
    for var in variables:
        for prob_col in regime_probs_aligned.columns:
            corr = macro_changes[var].corr(prob_changes[prob_col])
            if not np.isnan(corr):
                print(f"  {var} change vs {prob_col} change: {corr:.4f}")
    
    # Check specific periods where macro variables changed significantly
    print("\n" + "="*80)
    print("PERIODS WITH LARGE MACRO CHANGES")
    print("="*80)
    
    # Find periods with large changes in macro variables
    for var in variables:
        large_changes = macro_changes[var].nlargest(10)
        print(f"\nTop 10 largest changes in {var}:")
        for date, change in large_changes.items():
            if pd.notna(change):
                # Get regime probabilities before and after
                date_idx = macro_aligned.index.get_loc(date)
                if date_idx > 0:
                    prev_date = macro_aligned.index[date_idx - 1]
                    prev_probs = regime_probs_aligned.loc[prev_date]
                    curr_probs = regime_probs_aligned.loc[date]
                    
                    prev_dominant = prev_probs.idxmax()
                    curr_dominant = curr_probs.idxmax()
                    
                    max_prob_change = abs(curr_probs.max() - prev_probs.max())
                    
                    print(f"  {date.strftime('%Y-%m')}: change={change:.4f}, "
                          f"dominant: {prev_dominant}→{curr_dominant}, "
                          f"max_prob_change={max_prob_change:.4f}")
    
    # Check regime means to see if they're well separated
    print("\n" + "="*80)
    print("REGIME CHARACTERISTICS (from HMM model)")
    print("="*80)
    
    # Get regime means from the model
    if hasattr(hmm_model.model, 'means_'):
        print("\nRegime means (standardized space):")
        for i in range(4):
            print(f"  R{i}: {hmm_model.model.means_[i]}")
    
    # Check covariance to see how spread out regimes are
    if hasattr(hmm_model.model, 'covars_'):
        print("\nRegime covariances (diagonal, standardized space):")
        for i in range(4):
            print(f"  R{i}: {hmm_model.model.covars_[i]}")
        
        # Calculate distances between regime means
        print("\n" + "="*80)
        print("REGIME SEPARATION ANALYSIS")
        print("="*80)
        
        means = hmm_model.model.means_
        print("\nEuclidean distances between regime means:")
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(means[i] - means[j])
                print(f"  R{i} to R{j}: {dist:.4f}")
        
        # Calculate how many standard deviations apart regimes are
        print("\nRegime separation in terms of standard deviations:")
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(means[i] - means[j])
                # Use average std dev of the two regimes
                avg_std_i = np.sqrt(np.mean(np.diag(hmm_model.model.covars_[i])))
                avg_std_j = np.sqrt(np.mean(np.diag(hmm_model.model.covars_[j])))
                avg_std = (avg_std_i + avg_std_j) / 2
                std_separation = dist / avg_std if avg_std > 0 else 0
                print(f"  R{i} to R{j}: {std_separation:.2f} std devs")
        
        # Check overlap: calculate probability of each regime at other regimes' means
        print("\n" + "="*80)
        print("REGIME OVERLAP ANALYSIS")
        print("="*80)
        print("Probability of each regime at other regimes' mean locations:")
        
        for i in range(4):
            print(f"\nAt R{i} mean location ({means[i]}):")
            # Calculate probability of each regime at this location
            for j in range(4):
                # Use multivariate normal PDF (simplified for diagonal covariance)
                diff = means[i] - means[j]
                covar = hmm_model.model.covars_[j]
                # For diagonal covariance, calculate probability density
                var = np.diag(covar)
                log_prob = -0.5 * np.sum((diff**2) / var) - 0.5 * np.sum(np.log(2 * np.pi * var))
                # Normalize (rough approximation)
                prob_density = np.exp(log_prob)
                print(f"  P(R{j}|at R{i} mean): density={prob_density:.6f}")
        
        # Simulate: what happens when macro variables change significantly?
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS: How do probabilities change with macro shocks?")
        print("="*80)
        
        # Get a sample observation from the aligned data
        sample_idx = len(macro_aligned) // 2
        sample_macro = macro_aligned.iloc[sample_idx]
        sample_features = hmm_model.prepare_features(
            pd.DataFrame([sample_macro], columns=variables), 
            fit_scaler=False
        )[0]
        sample_probs = hmm_model.predict_proba(sample_features.reshape(1, -1))[0]
        print(f"\nBaseline observation (standardized): {sample_features}")
        print(f"Baseline probabilities: {dict(zip([f'R{i}' for i in range(4)], sample_probs))}")
        
        # Test large shocks
        shocks = [
            ("+2 std dev market vol", np.array([2.0, 0.0])),
            ("-2 std dev market vol", np.array([-2.0, 0.0])),
            ("+2 std dev inflation", np.array([0.0, 2.0])),
            ("-2 std dev inflation", np.array([0.0, -2.0])),
            ("+2 std dev both", np.array([2.0, 2.0])),
        ]
        
        print(f"\nTesting with actual regime means to verify model is working:")
        print(f"Regime means: {hmm_model.model.means_}")
        print(f"Covars shape: {hmm_model.model.covars_.shape}")
        print(f"Covars (first regime): {hmm_model.model.covars_[0] if hmm_model.model.covars_.ndim >= 2 else 'N/A'}")
        
        for i in range(4):
            mean_obs = hmm_model.model.means_[i]
            # Use score_samples directly to check if issue is in predict_proba
            log_probs = hmm_model.model.score_samples(mean_obs.reshape(1, -1))[1][0]
            # Convert to probabilities manually
            log_probs_stable = log_probs - np.max(log_probs)
            probs_manual = np.exp(log_probs_stable)
            probs_manual = probs_manual / probs_manual.sum()
            
            mean_probs = hmm_model.predict_proba(mean_obs.reshape(1, -1))[0]
            print(f"  At R{i} mean ({mean_obs}):")
            print(f"    Manual probs: {dict(zip([f'R{j}' for j in range(4)], probs_manual))}")
            print(f"    predict_proba: {dict(zip([f'R{j}' for j in range(4)], mean_probs))}")
            print(f"    Match: {np.allclose(probs_manual, mean_probs)}")
        
        for shock_name, shock in shocks:
            shocked_obs = sample_features + shock
            shocked_probs = hmm_model.predict_proba(shocked_obs.reshape(1, -1))[0]
            prob_change = np.abs(shocked_probs - sample_probs).max()
            print(f"\n{shock_name}:")
            print(f"  Shocked observation: {shocked_obs}")
            print(f"  New probabilities: {dict(zip([f'R{i}' for i in range(4)], shocked_probs))}")
            print(f"  Max probability change: {prob_change:.4f}")
            
            # Also check which regime this observation is closest to
            distances = [np.linalg.norm(shocked_obs - hmm_model.model.means_[i]) for i in range(4)]
            closest_regime = np.argmin(distances)
            print(f"  Closest to R{closest_regime} (distance: {distances[closest_regime]:.4f})")
    
    # Analyze dominant regime periods in detail
    print("\n" + "="*80)
    print("DOMINANT REGIME PERIODS ANALYSIS")
    print("="*80)
    
    # Find periods where same regime dominates for extended periods
    dominant_regimes = regime_probs.idxmax(axis=1).str.replace('prob_R', '').astype(int)
    
    # Group consecutive periods with same dominant regime
    regime_changes = (dominant_regimes != dominant_regimes.shift(1))
    regime_groups = regime_changes.cumsum()
    
    regime_periods = []
    for group_id in regime_groups.unique():
        group_mask = (regime_groups == group_id)
        group_dates = regime_probs.index[group_mask]
        group_regime = dominant_regimes[group_mask].iloc[0]
        group_probs = regime_probs.loc[group_mask]
        
        regime_periods.append({
            'regime': group_regime,
            'start_date': group_dates[0],
            'end_date': group_dates[-1],
            'duration': len(group_dates),
            'avg_max_prob': group_probs.max(axis=1).mean(),
            'min_max_prob': group_probs.max(axis=1).min(),
            'max_max_prob': group_probs.max(axis=1).max(),
        })
    
    regime_periods_df = pd.DataFrame(regime_periods).sort_values('duration', ascending=False)
    
    print("\nLongest periods with same dominant regime:")
    print(regime_periods_df.head(10).to_string(index=False))
    
    # Check if probabilities are actually changing during these long periods
    print("\n" + "="*80)
    print("PROBABILITY VARIATION DURING LONG PERIODS")
    print("="*80)
    
    for _, period in regime_periods_df.head(5).iterrows():
        period_mask = (regime_probs.index >= period['start_date']) & (regime_probs.index <= period['end_date'])
        period_probs = regime_probs.loc[period_mask]
        
        print(f"\nPeriod: {period['start_date']} to {period['end_date']} (R{period['regime']}, {period['duration']} periods)")
        print(f"  Average max probability: {period['avg_max_prob']:.3f}")
        print(f"  Probability range: {period['min_max_prob']:.3f} to {period['max_max_prob']:.3f}")
        print(f"  Standard deviation of max probability: {period_probs.max(axis=1).std():.3f}")
        
        # Check individual regime probabilities
        dominant_prob_col = f"prob_R{period['regime']}"
        print(f"  {dominant_prob_col} stats:")
        print(f"    Mean: {period_probs[dominant_prob_col].mean():.3f}")
        print(f"    Std: {period_probs[dominant_prob_col].std():.3f}")
        print(f"    Min: {period_probs[dominant_prob_col].min():.3f}")
        print(f"    Max: {period_probs[dominant_prob_col].max():.3f}")

if __name__ == "__main__":
    main()

