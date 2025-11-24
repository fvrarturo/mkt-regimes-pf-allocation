"""
Summarize Predictive Significance of Macro Variables for VIX by Regime

This script analyzes which macro variables have statistically significant
predictive power for VIX in each regime, using LAGGED variables (t-1 predicts t).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
output_dir = Path(__file__).parent / 'results'
tables_dir = output_dir / 'tables'
tables_dir.mkdir(parents=True, exist_ok=True)
summary_output = tables_dir / 'predictive_significance_summary.csv'

print("=" * 100)
print("PREDICTIVE SIGNIFICANCE OF MACRO VARIABLES FOR VIX (CONDITIONAL ON REGIMES)")
print("=" * 100)
print()
print("Using LAGGED macro variables (t-1) to predict VIX at time t")
print("Statistical significance: *** p<0.001, ** p<0.01, * p<0.05")
print()

# Load results
corr_df = pd.read_csv(tables_dir / 'vix_correlations_by_regime.csv')
reg_df = pd.read_csv(tables_dir / 'vix_regressions_by_regime.csv')

# Merge correlation and regression results
merged = pd.merge(
    corr_df[['regime', 'regime_name', 'variable', 'correlation', 'pvalue', 'abs_correlation']],
    reg_df[['regime', 'variable', 'coefficient', 'r_squared', 'pvalue']],
    on=['regime', 'variable'],
    suffixes=('_corr', '_reg')
)

# Define significance levels
def get_significance(pvalue):
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    else:
        return ''

# Add significance flags
merged['corr_significant'] = merged['pvalue_corr'] < 0.05
merged['reg_significant'] = merged['pvalue_reg'] < 0.05
merged['corr_sig_level'] = merged['pvalue_corr'].apply(get_significance)
merged['reg_sig_level'] = merged['pvalue_reg'].apply(get_significance)

# Group variables by category
variable_categories = {
    'Inflation': ['cpi', 'PCE_price_index', 'PPI_inflation'],
    'Economic Growth': ['gdp', 'real_gdp', 'unemployment', 'industrial_production', 
                       'retail_sales', 'tot_business_inventories'],
    'Monetary Policy': ['fedfunds', 'fed_reserve_discount_rate', 
                       '10y_treasury_const_maturity_rate', 'm2_real_money_supply'],
    'Financial Conditions': ['nat_fin_condition_indx', '10y_2y_spread', 'bofa_highyield_spread'],
    'Market Indicators': ['sp500', '3m_yield', '2y_yield', '10y_yield']
}

def get_category(var):
    for cat, vars_list in variable_categories.items():
        if var in vars_list:
            return cat
    return 'Other'

merged['category'] = merged['variable'].apply(get_category)

# Create summary by regime
print("=" * 100)
print("SUMMARY: SIGNIFICANT PREDICTORS BY REGIME")
print("=" * 100)
print()

summary_results = []

for regime in sorted(merged['regime'].unique()):
    regime_data = merged[merged['regime'] == regime].copy()
    regime_name = regime_data['regime_name'].iloc[0]
    
    # Filter to significant predictors (either correlation or regression)
    significant = regime_data[
        (regime_data['corr_significant']) | (regime_data['reg_significant'])
    ].copy()
    
    # Sort by absolute correlation (predictive strength)
    significant = significant.sort_values('abs_correlation', ascending=False)
    
    print(f"\n{'='*100}")
    print(f"REGIME {int(regime)}: {regime_name}")
    print(f"{'='*100}")
    print()
    print(f"Total significant predictors: {len(significant)}")
    print()
    print(f"{'Rank':<6} {'Variable':<35} {'Category':<20} {'Corr':<10} {'Corr P':<12} {'R²':<10} {'Reg P':<12} {'Both Sig':<10}")
    print("-" * 100)
    
    for rank, (idx, row) in enumerate(significant.iterrows(), 1):
        both_sig = 'Yes' if (row['corr_significant'] and row['reg_significant']) else 'No'
        
        print(f"{rank:<6} {row['variable']:<35} {row['category']:<20} "
              f"{row['correlation']:>7.3f} {row['corr_sig_level']:<4} "
              f"{row['pvalue_corr']:>10.2e} "
              f"{row['r_squared']:>7.3f} "
              f"{row['reg_sig_level']:<4} "
              f"{row['pvalue_reg']:>10.2e} "
              f"{both_sig:<10}")
        
        summary_results.append({
            'regime': int(regime),
            'regime_name': regime_name,
            'rank': rank,
            'variable': row['variable'],
            'category': row['category'],
            'correlation': row['correlation'],
            'correlation_pvalue': row['pvalue_corr'],
            'correlation_significant': row['corr_significant'],
            'correlation_sig_level': row['corr_sig_level'],
            'r_squared': row['r_squared'],
            'regression_pvalue': row['pvalue_reg'],
            'regression_significant': row['reg_significant'],
            'regression_sig_level': row['reg_sig_level'],
            'coefficient': row['coefficient'],
            'both_significant': row['corr_significant'] and row['reg_significant'],
            'abs_correlation': row['abs_correlation']
        })
    
    # Summary statistics
    print()
    print(f"Summary Statistics:")
    print(f"  Variables with significant correlation: {regime_data['corr_significant'].sum()}")
    print(f"  Variables with significant regression: {regime_data['reg_significant'].sum()}")
    print(f"  Variables significant in both: {(regime_data['corr_significant'] & regime_data['reg_significant']).sum()}")
    print(f"  Average R² for significant predictors: {significant['r_squared'].mean():.3f}")
    print(f"  Max R²: {significant['r_squared'].max():.3f} ({significant.loc[significant['r_squared'].idxmax(), 'variable']})")

# Create summary DataFrame
summary_df = pd.DataFrame(summary_results)

# Save results
summary_df.to_csv(summary_output, index=False)
print()
print(f"\n{'='*100}")
print(f"Detailed results saved to: {summary_output}")
print(f"{'='*100}")

# Create category-level summary
print()
print("=" * 100)
print("SUMMARY BY CATEGORY AND REGIME")
print("=" * 100)
print()

category_summary = []

for regime in sorted(merged['regime'].unique()):
    regime_data = merged[merged['regime'] == regime].copy()
    regime_name = regime_data['regime_name'].iloc[0]
    
    for category in sorted(merged['category'].unique()):
        cat_data = regime_data[regime_data['category'] == category]
        significant = cat_data[
            (cat_data['corr_significant']) | (cat_data['reg_significant'])
        ]
        
        if len(significant) > 0:
            best = significant.loc[significant['abs_correlation'].idxmax()]
            category_summary.append({
                'regime': int(regime),
                'regime_name': regime_name,
                'category': category,
                'n_significant': len(significant),
                'best_variable': best['variable'],
                'best_correlation': best['correlation'],
                'best_r_squared': best['r_squared'],
                'best_corr_pvalue': best['pvalue_corr'],
                'best_reg_pvalue': best['pvalue_reg']
            })

cat_summary_df = pd.DataFrame(category_summary)

print(f"{'Regime':<8} {'Category':<25} {'N Sig':<8} {'Best Variable':<30} {'Corr':<10} {'R²':<10} {'Corr P':<12} {'Reg P':<12}")
print("-" * 100)

for _, row in cat_summary_df.iterrows():
    print(f"{int(row['regime']):<8} {row['category']:<25} {int(row['n_significant']):<8} "
          f"{row['best_variable']:<30} {row['best_correlation']:>7.3f} "
          f"{row['best_r_squared']:>7.3f} {row['best_corr_pvalue']:>10.2e} "
          f"{row['best_reg_pvalue']:>10.2e}")

# Save category summary
cat_summary_df.to_csv(tables_dir / 'category_summary_by_regime.csv', index=False)

print()
print("=" * 100)
print("KEY FINDINGS")
print("=" * 100)
print()

# Find strongest predictors overall
for regime in sorted(merged['regime'].unique()):
    regime_data = merged[merged['regime'] == regime].copy()
    regime_name = regime_data['regime_name'].iloc[0]
    
    # Top 3 by correlation
    top_corr = regime_data.nlargest(3, 'abs_correlation')
    
    # Top 3 by R²
    top_r2 = regime_data.nlargest(3, 'r_squared')
    
    print(f"\nRegime {int(regime)}: {regime_name}")
    print(f"  Top 3 by predictive correlation (lagged):")
    for idx, row in top_corr.iterrows():
        sig = row['corr_sig_level']
        print(f"    {row['variable']:<30} corr={row['correlation']:>7.3f} {sig:<4} (p={row['pvalue_corr']:.2e})")
    
    print(f"  Top 3 by R² (predictive power):")
    for idx, row in top_r2.iterrows():
        sig = row['reg_sig_level']
        print(f"    {row['variable']:<30} R²={row['r_squared']:>7.3f} {sig:<4} (p={row['pvalue_reg']:.2e})")

print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print()
print("NOTE: This analysis uses LAGGED variables (t-1) to predict VIX at time t.")
print("Significant predictors indicate variables that have predictive power")
print("conditional on being in each regime.")
print()
print("⚠️  IMPORTANT LIMITATIONS:")
print("  1. Regime assignments use full sample (look-ahead bias in regime classification)")
print("  2. Results are in-sample (no out-of-sample validation)")
print("  3. Publication lags not accounted for")
print()
print("  See BIAS_ACCOUNTING_STATUS.md for details.")
print()
print("Results saved to:")
print(f"  - {summary_output}")
print(f"  - {output_dir / 'category_summary_by_regime.csv'}")

