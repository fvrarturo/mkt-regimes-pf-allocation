"""
Create Comprehensive Summary Table for Each Regime

This script creates a detailed summary table showing p-values, R², correlations,
coefficients, and other key metrics for each variable in each regime.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
output_dir = Path(__file__).parent / 'results'
tables_dir = output_dir / 'tables'
tables_dir.mkdir(parents=True, exist_ok=True)

print("=" * 120)
print("CREATING COMPREHENSIVE SUMMARY TABLE BY REGIME")
print("=" * 120)
print()

# Load results
print("Loading results...")
corr_df = pd.read_csv(tables_dir / 'vix_correlations_by_regime.csv')
reg_df = pd.read_csv(tables_dir / 'vix_regressions_by_regime.csv')
summary_df = pd.read_csv(tables_dir / 'vix_macro_relevance_summary.csv')

# Merge all data
merged = pd.merge(
    corr_df[['regime', 'regime_name', 'variable', 'correlation', 'pvalue', 'abs_correlation', 'n_observations']],
    reg_df[['regime', 'variable', 'coefficient', 'r_squared', 'pvalue', 't_statistic', 'n_observations']],
    on=['regime', 'variable'],
    suffixes=('_corr', '_reg'),
    how='outer'
)

# Add summary metrics
merged = pd.merge(
    merged,
    summary_df[['regime', 'variable', 'relevance_score', 'rf_importance']],
    on=['regime', 'variable'],
    how='left'
)

# Define significance levels
def get_significance(pvalue):
    if pd.isna(pvalue):
        return ''
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    else:
        return ''

# Add significance flags
merged['corr_sig'] = merged['pvalue_corr'].apply(get_significance)
merged['reg_sig'] = merged['pvalue_reg'].apply(get_significance)
merged['both_sig'] = (merged['corr_sig'] != '') & (merged['reg_sig'] != '')

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

# Create comprehensive summary table
summary_table = []

for regime in sorted(merged['regime'].unique()):
    regime_data = merged[merged['regime'] == regime].copy()
    regime_name = regime_data['regime_name'].iloc[0]
    
    # Sort by relevance score (or R² if missing)
    regime_data = regime_data.sort_values('r_squared', ascending=False, na_position='last')
    
    for idx, row in regime_data.iterrows():
        summary_table.append({
            'Regime': int(regime),
            'Regime_Name': regime_name,
            'Variable': row['variable'],
            'Category': row['category'],
            
            # Correlation metrics
            'Correlation': row['correlation'] if not pd.isna(row['correlation']) else np.nan,
            'Corr_Pvalue': row['pvalue_corr'] if not pd.isna(row['pvalue_corr']) else np.nan,
            'Corr_Significant': row['corr_sig'],
            'Abs_Correlation': row['abs_correlation'] if not pd.isna(row['abs_correlation']) else np.nan,
            
            # Regression metrics
            'R_Squared': row['r_squared'] if not pd.isna(row['r_squared']) else np.nan,
            'Reg_Pvalue': row['pvalue_reg'] if not pd.isna(row['pvalue_reg']) else np.nan,
            'Reg_Significant': row['reg_sig'],
            'Coefficient': row['coefficient'] if not pd.isna(row['coefficient']) else np.nan,
            'T_Statistic': row['t_statistic'] if not pd.isna(row['t_statistic']) else np.nan,
            
            # Combined metrics
            'Relevance_Score': row['relevance_score'] if not pd.isna(row['relevance_score']) else np.nan,
            'RF_Importance': row['rf_importance'] if not pd.isna(row['rf_importance']) else np.nan,
            'Both_Significant': 'Yes' if row['both_sig'] else 'No',
            
            # Sample size
            'N_Observations': int(row['n_observations_corr']) if not pd.isna(row['n_observations_corr']) else 
                            (int(row['n_observations_reg']) if not pd.isna(row['n_observations_reg']) else np.nan)
        })

df_summary = pd.DataFrame(summary_table)

# Save full table
output_path = tables_dir / 'comprehensive_summary_by_regime.csv'
df_summary.to_csv(output_path, index=False)
print(f"Saved comprehensive summary: {output_path}")
print()

# Create formatted summary for each regime
print("=" * 120)
print("COMPREHENSIVE SUMMARY BY REGIME")
print("=" * 120)
print()

for regime in sorted(df_summary['Regime'].unique()):
    regime_data = df_summary[df_summary['Regime'] == regime].copy()
    regime_name = regime_data['Regime_Name'].iloc[0]
    
    # Sort by R² (predictive power)
    regime_data = regime_data.sort_values('R_Squared', ascending=False, na_position='last')
    
    print(f"\n{'='*120}")
    print(f"REGIME {int(regime)}: {regime_name}")
    print(f"{'='*120}")
    print()
    print(f"{'Rank':<6} {'Variable':<30} {'Category':<20} {'Corr':<8} {'Corr_P':<12} {'R²':<8} {'Reg_P':<12} {'Coef':<10} {'T_Stat':<10} {'Both_Sig':<10}")
    print("-" * 120)
    
    for rank, (idx, row) in enumerate(regime_data.iterrows(), 1):
        corr_str = f"{row['Correlation']:.3f}" if not pd.isna(row['Correlation']) else "N/A"
        corr_p_str = f"{row['Corr_Pvalue']:.2e}" if not pd.isna(row['Corr_Pvalue']) else "N/A"
        r2_str = f"{row['R_Squared']:.3f}" if not pd.isna(row['R_Squared']) else "N/A"
        reg_p_str = f"{row['Reg_Pvalue']:.2e}" if not pd.isna(row['Reg_Pvalue']) else "N/A"
        coef_str = f"{row['Coefficient']:.3f}" if not pd.isna(row['Coefficient']) else "N/A"
        t_stat_str = f"{row['T_Statistic']:.2f}" if not pd.isna(row['T_Statistic']) else "N/A"
        
        print(f"{rank:<6} {row['Variable']:<30} {row['Category']:<20} "
              f"{corr_str:<8} {row['Corr_Significant']:<4} {corr_p_str:<12} "
              f"{r2_str:<8} {row['Reg_Significant']:<4} {reg_p_str:<12} "
              f"{coef_str:<10} {t_stat_str:<10} {row['Both_Significant']:<10}")
    
    # Summary statistics
    print()
    print(f"Summary Statistics:")
    print(f"  Total variables: {len(regime_data)}")
    print(f"  Significant correlation (p<0.05): {(regime_data['Corr_Significant'] != '').sum()}")
    print(f"  Significant regression (p<0.05): {(regime_data['Reg_Significant'] != '').sum()}")
    print(f"  Significant in both: {(regime_data['Both_Significant'] == 'Yes').sum()}")
    print(f"  Average R²: {regime_data['R_Squared'].mean():.3f}")
    print(f"  Max R²: {regime_data['R_Squared'].max():.3f} ({regime_data.loc[regime_data['R_Squared'].idxmax(), 'Variable']})")
    print(f"  Average |Correlation|: {regime_data['Abs_Correlation'].mean():.3f}")
    print(f"  Max |Correlation|: {regime_data['Abs_Correlation'].max():.3f} ({regime_data.loc[regime_data['Abs_Correlation'].idxmax(), 'Variable']})")

# Create top predictors table (top 5 per regime)
print()
print("=" * 120)
print("TOP 5 PREDICTORS BY REGIME (BY R²)")
print("=" * 120)
print()

top_predictors = []

for regime in sorted(df_summary['Regime'].unique()):
    regime_data = df_summary[df_summary['Regime'] == regime].copy()
    regime_name = regime_data['Regime_Name'].iloc[0]
    
    # Get top 5 by R²
    top5 = regime_data.nlargest(5, 'R_Squared')
    
    for rank, (idx, row) in enumerate(top5.iterrows(), 1):
        top_predictors.append({
            'Regime': int(regime),
            'Regime_Name': regime_name,
            'Rank': rank,
            'Variable': row['Variable'],
            'Category': row['Category'],
            'Correlation': row['Correlation'],
            'Corr_Pvalue': row['Corr_Pvalue'],
            'Corr_Significant': row['Corr_Significant'],
            'R_Squared': row['R_Squared'],
            'Reg_Pvalue': row['Reg_Pvalue'],
            'Reg_Significant': row['Reg_Significant'],
            'Coefficient': row['Coefficient'],
            'T_Statistic': row['T_Statistic'],
            'Relevance_Score': row['Relevance_Score']
        })
        
        print(f"Regime {int(regime)} ({regime_name}) - Rank {rank}:")
        print(f"  Variable: {row['Variable']} ({row['Category']})")
        print(f"  Correlation: {row['Correlation']:.3f} {row['Corr_Significant']} (p={row['Corr_Pvalue']:.2e})")
        print(f"  R²: {row['R_Squared']:.3f} {row['Reg_Significant']} (p={row['Reg_Pvalue']:.2e})")
        print(f"  Coefficient: {row['Coefficient']:.3f}, T-stat: {row['T_Statistic']:.2f}")
        print(f"  Relevance Score: {row['Relevance_Score']:.3f}")
        print()

# Save top predictors table
top_predictors_df = pd.DataFrame(top_predictors)
top_predictors_path = tables_dir / 'top_5_predictors_by_regime.csv'
top_predictors_df.to_csv(top_predictors_path, index=False)
print(f"Saved top 5 predictors: {top_predictors_path}")
print()

# Create pivot table: Variables x Regimes with R²
print("=" * 120)
print("R² BY VARIABLE AND REGIME (PIVOT TABLE)")
print("=" * 120)
print()

pivot_r2 = df_summary.pivot_table(
    index='Variable',
    columns='Regime',
    values='R_Squared',
    aggfunc='first'
)

# Add category
categories = df_summary[['Variable', 'Category']].drop_duplicates().set_index('Variable')
pivot_r2['Category'] = categories['Category']
pivot_r2 = pivot_r2[['Category'] + [col for col in pivot_r2.columns if col != 'Category']]

# Sort by category then by average R²
pivot_r2['Avg_R2'] = pivot_r2[[0, 1, 2, 3]].mean(axis=1)
pivot_r2 = pivot_r2.sort_values(['Category', 'Avg_R2'], ascending=[True, False])

print(pivot_r2[[0, 1, 2, 3, 'Avg_R2']].to_string())
print()

# Save pivot table
pivot_r2_path = tables_dir / 'r2_pivot_by_regime.csv'
pivot_r2.to_csv(pivot_r2_path)
print(f"Saved R² pivot table: {pivot_r2_path}")
print()

# Create pivot table for p-values
print("=" * 120)
print("REGRESSION P-VALUES BY VARIABLE AND REGIME (PIVOT TABLE)")
print("=" * 120)
print()

pivot_pval = df_summary.pivot_table(
    index='Variable',
    columns='Regime',
    values='Reg_Pvalue',
    aggfunc='first'
)

pivot_pval['Category'] = categories['Category']
pivot_pval = pivot_pval[['Category'] + [col for col in pivot_pval.columns if col != 'Category']]
pivot_pval = pivot_pval.sort_values(['Category', 'Variable'])

print(pivot_pval[[0, 1, 2, 3]].to_string())
print()

# Save p-value pivot table
pivot_pval_path = tables_dir / 'pvalue_pivot_by_regime.csv'
pivot_pval.to_csv(pivot_pval_path)
print(f"Saved p-value pivot table: {pivot_pval_path}")
print()

print("=" * 120)
print("ANALYSIS COMPLETE")
print("=" * 120)
print()
print("Summary tables saved to:")
print(f"  - {output_path}")
print(f"  - {top_predictors_path}")
print(f"  - {pivot_r2_path}")
print(f"  - {pivot_pval_path}")
print()

