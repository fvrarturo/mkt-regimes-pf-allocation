"""
Enhanced ERP Analysis with Lagged Variables, Overall Analysis, and Multiple Time Horizons

This addresses the issue that volatility appears too relevant due to contemporaneous correlations.
We analyze:
1. Lagged relationships (macro at t-1 predicts ERP at t)
2. Overall analysis (not conditional on regimes)
3. Multiple time horizons (monthly, quarterly, annual)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from erp_macro_relevance import ERPMacroRelevanceAnalyzer

# Set up paths
project_root = Path(__file__).parent.parent
regime_assignments_path = project_root / 'test_regimes_matrix' / 'results' / 'regime_assignments.csv'
macro_data_dir = project_root / 'data' / 'macro_processed'
sp500_path = project_root / 'data' / 'macro_processed' / 'other' / 'sp500_processed.csv'
yield_3m_path = project_root / 'data' / 'macro_processed' / 'other' / '3m_yield_processed.csv'
output_dir = Path(__file__).parent / 'results' / 'enhanced_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("ENHANCED ERP ANALYSIS")
print("=" * 90)
print()
print("Addressing issues:")
print("1. Contemporaneous vs Lagged relationships")
print("2. Overall analysis (not just regime-conditional)")
print("3. Multiple time horizons")
print()

# Initialize analyzer
analyzer = ERPMacroRelevanceAnalyzer(
    regime_assignments_path=regime_assignments_path,
    macro_data_dir=macro_data_dir,
    sp500_path=sp500_path,
    yield_3m_path=yield_3m_path,
    output_dir=output_dir
)

# Load data
analyzer.load_regime_assignments()
analyzer.calculate_erp()
analyzer.load_macro_variables()
analyzer.combine_data()

data = analyzer.combined_data.copy()
macro_cols = list(analyzer.macro_data.keys())

print("=" * 90)
print("1. CONTEMPORANEOUS vs LAGGED ANALYSIS")
print("=" * 90)
print()

# Create lagged versions of macro variables
for var in macro_cols:
    if var in data.columns:
        data[f'{var}_lag1'] = data[var].shift(1)  # Previous period

results_lagged = []
results_contemp = []

print("Comparing contemporaneous vs lagged correlations...")
print()

for var in macro_cols:
    if var not in data.columns:
        continue
    
    # Contemporaneous correlation
    contemp_data = data[['erp', var]].dropna()
    if len(contemp_data) >= 20:
        corr_contemp, pval_contemp = stats.pearsonr(contemp_data['erp'], contemp_data[var])
        results_contemp.append({
            'variable': var,
            'correlation': corr_contemp,
            'pvalue': pval_contemp,
            'n_obs': len(contemp_data),
            'type': 'contemporaneous'
        })
    
    # Lagged correlation (macro at t-1, ERP at t)
    lagged_data = data[['erp', f'{var}_lag1']].dropna()
    if len(lagged_data) >= 20:
        corr_lagged, pval_lagged = stats.pearsonr(lagged_data['erp'], lagged_data[f'{var}_lag1'])
        results_lagged.append({
            'variable': var,
            'correlation': corr_lagged,
            'pvalue': pval_lagged,
            'n_obs': len(lagged_data),
            'type': 'lagged'
        })

df_contemp = pd.DataFrame(results_contemp).sort_values('correlation', key=abs, ascending=False)
df_lagged = pd.DataFrame(results_lagged).sort_values('correlation', key=abs, ascending=False)

print("TOP 10 VARIABLES - CONTEMPORANEOUS:")
print("-" * 90)
print(f"{'Variable':<40} {'Correlation':<15} {'P-value':<15} {'Abs Corr':<12}")
print("-" * 90)
for _, row in df_contemp.head(10).iterrows():
    sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
    print(f"{row['variable']:<40} {row['correlation']:>8.3f} {sig:<4} {row['pvalue']:>12.2e} {abs(row['correlation']):>10.3f}")

print()
print("TOP 10 VARIABLES - LAGGED (Predictive):")
print("-" * 90)
print(f"{'Variable':<40} {'Correlation':<15} {'P-value':<15} {'Abs Corr':<12}")
print("-" * 90)
for _, row in df_lagged.head(10).iterrows():
    sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
    print(f"{row['variable']:<40} {row['correlation']:>8.3f} {sig:<4} {row['pvalue']:>12.2e} {abs(row['correlation']):>10.3f}")

print()
print("KEY INSIGHT: If volatility is only relevant contemporaneously but not lagged,")
print("it means it's describing the same phenomenon, not predicting it.")
print()

# Compare volatility variables
vol_vars = ['vix', 'nasdaq_vol_indx', '3month_vol_index_sp500']
print("VOLATILITY VARIABLES - CONTEMPORANEOUS vs LAGGED:")
print("-" * 90)
for var in vol_vars:
    if var in df_contemp['variable'].values:
        contemp = df_contemp[df_contemp['variable'] == var].iloc[0]
        lagged = df_lagged[df_lagged['variable'] == var].iloc[0] if var in df_lagged['variable'].values else None
        
        print(f"{var}:")
        print(f"  Contemporaneous: {contemp['correlation']:.3f} (p={contemp['pvalue']:.2e})")
        if lagged is not None:
            print(f"  Lagged:          {lagged['correlation']:.3f} (p={lagged['pvalue']:.2e})")
            diff = abs(contemp['correlation']) - abs(lagged['correlation'])
            print(f"  Difference:      {diff:.3f} (contemp stronger by {diff:.3f})")
        print()

# Save results
df_contemp.to_csv(output_dir / 'correlations_contemporaneous.csv', index=False)
df_lagged.to_csv(output_dir / 'correlations_lagged.csv', index=False)

print("=" * 90)
print("2. OVERALL ANALYSIS (NOT CONDITIONAL ON REGIMES)")
print("=" * 90)
print()

# Overall correlations (all data, not split by regime)
results_overall = []
for var in macro_cols:
    if var not in data.columns:
        continue
    
    var_data = data[['erp', var]].dropna()
    if len(var_data) >= 20:
        corr, pval = stats.pearsonr(var_data['erp'], var_data[var])
        results_overall.append({
            'variable': var,
            'correlation': corr,
            'pvalue': pval,
            'abs_correlation': abs(corr),
            'n_obs': len(var_data)
        })

df_overall = pd.DataFrame(results_overall).sort_values('abs_correlation', ascending=False)

print("TOP 15 VARIABLES - OVERALL (ALL REGIMES COMBINED):")
print("-" * 90)
print(f"{'Rank':<6} {'Variable':<40} {'Correlation':<15} {'P-value':<15} {'N':<8}")
print("-" * 90)
for rank, (idx, row) in enumerate(df_overall.head(15).iterrows(), 1):
    sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
    print(f"{rank:<6} {row['variable']:<40} {row['correlation']:>8.3f} {sig:<4} {row['pvalue']:>12.2e} {row['n_obs']:>6}")

df_overall.to_csv(output_dir / 'correlations_overall.csv', index=False)

print()
print("=" * 90)
print("3. MULTIPLE TIME HORIZONS")
print("=" * 90)
print()

# Calculate quarterly and annual returns
data['erp_quarterly'] = data['erp'].resample('Q').sum()  # Sum monthly returns for quarterly
data['erp_annual'] = data['erp'].resample('Y').sum()  # Sum monthly returns for annual

# For macro variables, use end-of-period values
for var in macro_cols:
    if var in data.columns:
        data[f'{var}_quarterly'] = data[var].resample('Q').last()
        data[f'{var}_annual'] = data[var].resample('Y').last()

results_horizons = []

for horizon in ['monthly', 'quarterly', 'annual']:
    if horizon == 'monthly':
        erp_col = 'erp'
    elif horizon == 'quarterly':
        erp_col = 'erp_quarterly'
    else:
        erp_col = 'erp_annual'
    
    for var in macro_cols:
        if horizon == 'monthly':
            var_col = var
        elif horizon == 'quarterly':
            var_col = f'{var}_quarterly'
        else:
            var_col = f'{var}_annual'
        
        if var_col not in data.columns or erp_col not in data.columns:
            continue
        
        var_data = data[[erp_col, var_col]].dropna()
        if len(var_data) >= 10:
            corr, pval = stats.pearsonr(var_data[erp_col], var_data[var_col])
            results_horizons.append({
                'horizon': horizon,
                'variable': var,
                'correlation': corr,
                'pvalue': pval,
                'abs_correlation': abs(corr),
                'n_obs': len(var_data)
            })

df_horizons = pd.DataFrame(results_horizons)

print("TOP 10 VARIABLES BY TIME HORIZON:")
print()

for horizon in ['monthly', 'quarterly', 'annual']:
    horizon_data = df_horizons[df_horizons['horizon'] == horizon].sort_values('abs_correlation', ascending=False)
    
    print(f"{horizon.upper()} HORIZON:")
    print("-" * 90)
    print(f"{'Variable':<40} {'Correlation':<15} {'P-value':<15} {'N':<8}")
    print("-" * 90)
    for _, row in horizon_data.head(10).iterrows():
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        print(f"{row['variable']:<40} {row['correlation']:>8.3f} {sig:<4} {row['pvalue']:>12.2e} {row['n_obs']:>6}")
    print()

df_horizons.to_csv(output_dir / 'correlations_by_horizon.csv', index=False)

print("=" * 90)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 90)
print()
print("1. VOLATILITY ISSUE:")
print("   - If volatility shows strong contemporaneous but weak lagged correlation,")
print("     it's describing the same phenomenon (high vol → low returns at same time)")
print("     not predicting future returns")
print("   - For prediction, focus on lagged relationships")
print()
print("2. REGIME CONDITIONALITY:")
print("   - Overall analysis shows relationships across all regimes")
print("   - Regime-conditional analysis shows regime-specific effects")
print("   - Both are useful for different purposes")
print()
print("3. TIME HORIZONS:")
print("   - Monthly: Short-term relationships")
print("   - Quarterly: Medium-term relationships")
print("   - Annual: Long-term relationships")
print("   - Different variables may matter at different horizons")
print()
print(f"Results saved to: {output_dir}")
print()

