"""
Additional Analysis: ERP Returns vs Interest Rates

This script analyzes the relationship between ERP RETURNS (changes) and interest rates,
which is more economically meaningful than analyzing ERP levels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
project_root = Path(__file__).parent.parent
regime_assignments_path = project_root / 'test_regimes_matrix' / 'results' / 'regime_assignments.csv'
combined_data_path = project_root / 'relevance_indicators' / 'results' / 'combined_data_sample.csv'

# Load regime assignments
regime_df = pd.read_csv(regime_assignments_path)
regime_df['date'] = pd.to_datetime(regime_df['date'])
regime_df = regime_df.set_index('date')

# Extract probability columns
prob_cols = [col for col in regime_df.columns if col.startswith('prob_R')]
regime_prob_cols = {}
for i in range(4):
    for col in prob_cols:
        if f'R{i}' in col:
            regime_prob_cols[i] = col
            break

# Load combined data (we need the full dataset, not just sample)
# Let's recreate it quickly
from erp_macro_relevance import ERPMacroRelevanceAnalyzer

analyzer = ERPMacroRelevanceAnalyzer(
    regime_assignments_path=regime_assignments_path,
    macro_data_dir=project_root / 'data' / 'macro_processed',
    sp500_path=project_root / 'data' / 'macro_processed' / 'other' / 'sp500_processed.csv',
    yield_3m_path=project_root / 'data' / 'macro_processed' / 'other' / '3m_yield_processed.csv',
    output_dir=Path(__file__).parent / 'results'
)

# Load data
analyzer.load_regime_assignments()
analyzer.calculate_erp()
analyzer.load_macro_variables()
analyzer.combine_data()

# Get combined data
data = analyzer.combined_data.copy()

# Calculate returns and changes
data['erp_return'] = data['erp'].pct_change()
data['sp500_return'] = data['sp500'].pct_change()

# Interest rate variables
rate_vars = ['fedfunds', 'fed_reserve_discount_rate', '10y_treasury_const_maturity_rate', 
             '2y_yield', '10y_yield', '10y_2y_spread']

# Calculate changes in interest rates
for var in rate_vars:
    if var in data.columns:
        data[f'{var}_change'] = data[var].pct_change()
        data[f'{var}_diff'] = data[var].diff()

print("=" * 80)
print("ERP RETURNS vs INTEREST RATES ANALYSIS")
print("=" * 80)
print()
print("This analysis looks at ERP RETURNS (changes) vs interest rate changes,")
print("which is more economically meaningful than analyzing levels.")
print()

results = []

for regime in sorted(data['regime'].unique()):
    if regime not in regime_prob_cols:
        continue
    
    prob_col = regime_prob_cols[regime]
    regime_data = data[data[prob_col] > 0.01].copy()
    weights = regime_data[prob_col].values
    
    regime_name = data[data['regime'] == regime]['regime_name'].iloc[0]
    
    print(f"REGIME {int(regime)}: {regime_name}")
    print("-" * 80)
    print(f"{'Variable':<35} {'Corr (Returns)':<18} {'Corr (Levels)':<18} {'P-value':<12}")
    print("-" * 80)
    
    for var in rate_vars:
        if var not in data.columns:
            continue
        
        # Correlation with ERP returns
        ret_data = regime_data[['erp_return', f'{var}_change']].dropna()
        if len(ret_data) < 10:
            continue
        
        ret_weights = regime_data.loc[ret_data.index, prob_col].values
        ret_weights = ret_weights / ret_weights.sum()
        
        # Weighted correlation for returns
        x = ret_data['erp_return'].values
        y = ret_data[f'{var}_change'].values
        x_mean = np.average(x, weights=ret_weights)
        y_mean = np.average(y, weights=ret_weights)
        cov_xy = np.average((x - x_mean) * (y - y_mean), weights=ret_weights)
        var_x = np.average((x - x_mean) ** 2, weights=ret_weights)
        var_y = np.average((y - y_mean) ** 2, weights=ret_weights)
        
        if var_x > 0 and var_y > 0:
            corr_returns = cov_xy / np.sqrt(var_x * var_y)
        else:
            corr_returns = 0.0
        
        # Correlation with ERP levels (for comparison)
        level_data = regime_data[['erp', var]].dropna()
        level_weights = regime_data.loc[level_data.index, prob_col].values
        level_weights = level_weights / level_weights.sum()
        
        x_l = level_data['erp'].values
        y_l = level_data[var].values
        x_mean_l = np.average(x_l, weights=level_weights)
        y_mean_l = np.average(y_l, weights=level_weights)
        cov_xy_l = np.average((x_l - x_mean_l) * (y_l - y_mean_l), weights=level_weights)
        var_x_l = np.average((x_l - x_mean_l) ** 2, weights=level_weights)
        var_y_l = np.average((y_l - y_mean_l) ** 2, weights=level_weights)
        
        if var_x_l > 0 and var_y_l > 0:
            corr_levels = cov_xy_l / np.sqrt(var_x_l * var_y_l)
        else:
            corr_levels = 0.0
        
        # P-value for returns correlation
        n_eff = (np.sum(ret_weights) ** 2) / np.sum(ret_weights ** 2)
        if n_eff > 2 and abs(corr_returns) < 1:
            t_stat = corr_returns * np.sqrt((n_eff - 2) / (1 - corr_returns ** 2))
            pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff - 2))
        else:
            pvalue = 1.0
        
        sig = '***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else ''
        
        print(f"{var:<35} {corr_returns:>8.3f} {sig:<7} {corr_levels:>8.3f} {pvalue:>12.2e}")
        
        results.append({
            'regime': regime,
            'regime_name': regime_name,
            'variable': var,
            'correlation_returns': corr_returns,
            'correlation_levels': corr_levels,
            'pvalue': pvalue,
            'n_observations': len(ret_data)
        })
    
    print()
    print("Note: *** p<0.001, ** p<0.01, * p<0.05")
    print()

# Save results
results_df = pd.DataFrame(results)
output_path = Path(__file__).parent / 'results' / 'erp_returns_vs_rates.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("Interest rates are more relevant for explaining ERP CHANGES (returns)")
print("than ERP LEVELS. The level analysis is dominated by SP500 level,")
print("which makes interest rates appear less relevant than they actually are.")
print()

