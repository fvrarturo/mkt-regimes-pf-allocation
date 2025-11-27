"""
Main script for conditional extremeness regression analysis.

Implements regression: ERP_t = α_r + β_r X_t + γ_r · Ext_t + (δ_r X_t · Ext_t) + ε_t

Focuses on:
- γ_r: extremeness main effect by regime
- δ_r: interaction effects (how extremeness changes macro-ERP relationships)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from preprocessing import (
    load_regime_data,
    load_extremeness_data,
    combine_regime_extremeness,
    create_extremeness_variables,
    create_per_variable_extremeness
)
from regression import (
    estimate_regime_extremeness_regression,
    extract_key_effects,
    compute_marginal_effects
)
from plotting import (
    plot_marginal_effects,
    plot_regime_fragility_heatmap,
    plot_beta_comparison
)


def main():
    """Main execution function."""
    print("="*80)
    print("Conditional Extremeness Regression Analysis")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load regime data (use HMM optimal as it's the best model)
    print("\n" + "="*80)
    print("Step 1: Loading regime assignments")
    print("="*80)
    regime_df = load_regime_data(regime_type='hmm_optimal')
    print(f"Loaded {len(regime_df)} regime assignments")
    print(f"Regimes: {sorted(regime_df['regime'].unique())}")
    print(f"Date range: {regime_df.index.min()} to {regime_df.index.max()}")
    
    # Step 2: Load extremeness data
    print("\n" + "="*80)
    print("Step 2: Loading extremeness scores")
    print("="*80)
    extremeness_df = load_extremeness_data()
    print(f"Loaded extremeness data: {len(extremeness_df)} observations")
    print(f"Extremeness range: [{extremeness_df['extremeness'].min():.4f}, {extremeness_df['extremeness'].max():.4f}]")
    print(f"Extreme states (90th percentile): {extremeness_df['is_extreme_p90'].sum()} ({extremeness_df['is_extreme_p90'].mean()*100:.1f}%)")
    
    # Step 3: Combine regime and extremeness data
    print("\n" + "="*80)
    print("Step 3: Combining regime and extremeness data")
    print("="*80)
    combined_df = combine_regime_extremeness(regime_df, extremeness_df)
    print(f"Combined data: {len(combined_df)} observations")
    print(f"Regimes in combined data: {sorted(combined_df['regime'].unique())}")
    
    # Step 4: Create extremeness variables
    print("\n" + "="*80)
    print("Step 4: Creating extremeness variables")
    print("="*80)
    combined_df = create_extremeness_variables(combined_df, method='binary')
    print(f"Extreme states: {combined_df['extreme'].sum()} ({combined_df['extreme'].mean()*100:.1f}%)")
    
    # Check if we have ERP data
    if 'erp' not in combined_df.columns:
        print("\nWarning: ERP not found in combined data. Loading ERP separately...")
        erp_path = base_dir / "data" / "macro_processed" / "equity_risk_pr.csv"
        erp_df = pd.read_csv(erp_path, parse_dates=["date"])
        erp_df = erp_df.set_index("date").sort_index()
        
        # Merge ERP
        if 'erp' in erp_df.columns:
            combined_df = combined_df.join(erp_df[['erp']], how='left')
        elif 'equity_risk_premium' in erp_df.columns:
            combined_df['erp'] = erp_df['equity_risk_premium']
        else:
            # Compute from components
            combined_df['erp'] = erp_df['stock_return'] - erp_df['risk_free_return']
        
        combined_df = combined_df.dropna(subset=['erp'])
        print(f"After adding ERP: {len(combined_df)} observations")
    
    # Step 5: Estimate regressions by regime
    print("\n" + "="*80)
    print("Step 5: Estimating conditional extremeness regressions")
    print("="*80)
    regression_results = estimate_regime_extremeness_regression(combined_df, regime_col='regime', horizon=1)
    
    print(f"\nEstimated regressions for {len(regression_results)} regimes:")
    for regime, res in regression_results.items():
        print(f"\nRegime {regime}:")
        print(f"  Observations: {res['n_obs']}")
        print(f"  R-squared: {res['r_squared']:.4f}")
        print(f"  RMSE: {res['rmse']:.4f}")
    
    # Step 6: Extract key effects
    print("\n" + "="*80)
    print("Step 6: Extracting extremeness effects")
    print("="*80)
    effects_df = extract_key_effects(regression_results)
    
    # Save effects table
    effects_path = output_dir / "extremeness_effects_by_regime.csv"
    effects_df.to_csv(effects_path, index=False)
    print(f"\nSaved extremeness effects table to {effects_path}")
    print("\nKey Effects Summary:")
    print(effects_df[['regime', 'variable', 'gamma_extreme', 'gamma_pvalue', 
                      'delta_interaction', 'delta_pvalue', 'beta_normal', 'beta_extreme']].to_string())
    
    # Step 7: Compute marginal effects
    print("\n" + "="*80)
    print("Step 7: Computing marginal effects")
    print("="*80)
    marginal_effects_dict = {}
    for regime in regression_results.keys():
        me_df = compute_marginal_effects(regression_results, regime)
        marginal_effects_dict[regime] = me_df
        print(f"\nRegime {regime} - Marginal Effects:")
        print(me_df.to_string(index=False))
    
    # Step 8: Create visualizations
    print("\n" + "="*80)
    print("Step 8: Creating visualizations")
    print("="*80)
    
    # Marginal effects plot
    plot_marginal_effects(marginal_effects_dict, output_dir)
    
    # Fragility heatmap
    plot_regime_fragility_heatmap(effects_df, output_dir)
    
    # Beta comparison
    plot_beta_comparison(effects_df, output_dir)
    
    # Step 9: Per-variable extremeness analysis (Task 2)
    print("\n" + "="*80)
    print("Step 9: Per-variable extremeness analysis")
    print("="*80)
    
    combined_df = create_per_variable_extremeness(combined_df)
    
    # Summary statistics for per-variable extremeness
    per_var_cols = [col for col in combined_df.columns if col.startswith('extreme_') and col.endswith('_factor')]
    
    if len(per_var_cols) > 0:
        print("\nPer-variable extremeness summary:")
        for col in per_var_cols:
            var_name = col.replace('extreme_', '').replace('_factor', '')
            extreme_count = combined_df[col].sum()
            extreme_pct = combined_df[col].mean() * 100
            
            # ERP statistics when this variable is extreme
            extreme_erp = combined_df[combined_df[col] == 1]['erp']
            normal_erp = combined_df[combined_df[col] == 0]['erp']
            
            print(f"\n{var_name}:")
            print(f"  Extreme periods: {extreme_count} ({extreme_pct:.1f}%)")
            if len(extreme_erp) > 0 and len(normal_erp) > 0:
                print(f"  ERP mean (extreme): {extreme_erp.mean():.4f}")
                print(f"  ERP mean (normal): {normal_erp.mean():.4f}")
                print(f"  ERP std (extreme): {extreme_erp.std():.4f}")
                print(f"  ERP std (normal): {normal_erp.std():.4f}")
        
        # Save per-variable summary
        per_var_summary = []
        for col in per_var_cols:
            var_name = col.replace('extreme_', '').replace('_factor', '')
            extreme_erp = combined_df[combined_df[col] == 1]['erp']
            normal_erp = combined_df[combined_df[col] == 0]['erp']
            
            if len(extreme_erp) > 0 and len(normal_erp) > 0:
                per_var_summary.append({
                    'variable': var_name,
                    'n_extreme': len(extreme_erp),
                    'erp_mean_extreme': extreme_erp.mean(),
                    'erp_std_extreme': extreme_erp.std(),
                    'erp_mean_normal': normal_erp.mean(),
                    'erp_std_normal': normal_erp.std(),
                    'mean_difference': extreme_erp.mean() - normal_erp.mean(),
                    'volatility_ratio': extreme_erp.std() / normal_erp.std() if normal_erp.std() > 0 else np.nan
                })
        
        per_var_df = pd.DataFrame(per_var_summary)
        per_var_path = output_dir / "per_variable_extremeness_summary.csv"
        per_var_df.to_csv(per_var_path, index=False)
        print(f"\nSaved per-variable extremeness summary to {per_var_path}")
    
    # Step 10: Summary
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - extremeness_effects_by_regime.csv")
    print("  - per_variable_extremeness_summary.csv")
    print("  - marginal_effects_by_regime.png")
    print("  - regime_fragility_heatmap.png")
    print("  - beta_comparison_normal_vs_extreme.png")


if __name__ == "__main__":
    main()

