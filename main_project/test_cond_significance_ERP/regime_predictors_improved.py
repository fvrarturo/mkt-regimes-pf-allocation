"""
Improved Regime-Conditional Predictors Analysis

Uses:
1. Hard regime assignments (larger sample sizes per regime)
2. Multiple lags (t-1, t-2, t-3)
3. Quarterly aggregation
4. Changes in macro variables where appropriate
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from erp_macro_relevance import ERPMacroRelevanceAnalyzer

# Set up paths
project_root = Path(__file__).parent.parent
regime_assignments_path = project_root / 'test_regimes_matrix' / 'results' / 'regime_assignments.csv'
macro_data_dir = project_root / 'data' / 'macro_processed'
sp500_path = project_root / 'data' / 'macro_processed' / 'other' / 'sp500_processed.csv'
yield_3m_path = project_root / 'data' / 'macro_processed' / 'other' / '3m_yield_processed.csv'
output_dir = Path(__file__).parent / 'results' / 'regime_predictors_improved'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("IMPROVED REGIME-CONDITIONAL PREDICTORS ANALYSIS")
print("=" * 90)
print()
print("Using:")
print("  1. Regime probabilities (weighted analysis)")
print("  2. Multiple lags (t-1, t-2, t-3)")
print("  3. Quarterly aggregation")
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
macro_cols = [v for v in list(analyzer.macro_data.keys()) 
              if v not in ['vix', 'nasdaq_vol_indx', '3month_vol_index_sp500']]

regime_names = {
    0: 'Low Growth / High Inflation',
    1: 'High Growth / High Inflation',
    2: 'High Growth / Low Inflation',
    3: 'Low Growth / Low Inflation'
}

# Create lags for all variables
for var in macro_cols:
    if var in data.columns:
        for lag in [1, 2, 3]:
            data[f'{var}_lag{lag}'] = data[var].shift(lag)

print("=" * 90)
print("1. MONTHLY ANALYSIS - MULTIPLE LAGS (USING REGIME PROBABILITIES)")
print("=" * 90)
print()

all_results = []

for regime in sorted(data['regime'].unique()):
    # Use regime probabilities instead of hard assignments
    if hasattr(analyzer, 'regime_prob_cols') and regime in analyzer.regime_prob_cols:
        prob_col = analyzer.regime_prob_cols[regime]
        regime_data = data[data[prob_col] > 0.01].copy()  # Include observations with >1% probability
        weights = regime_data[prob_col].values
    else:
        regime_data = data[data['regime'] == regime].copy()
        weights = None
    
    regime_name = regime_names[regime]
    
    if len(regime_data) < 15:
        print(f"Regime {int(regime)}: {regime_name} - Insufficient data ({len(regime_data)} obs)")
        print()
        continue
    
    # Calculate effective sample size
    if weights is not None:
        n_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    else:
        n_eff = len(regime_data)
    
    print(f"REGIME {int(regime)}: {regime_name}")
    print(f"Observations: {len(regime_data)} (effective n: {n_eff:.1f})")
    print("-" * 90)
    print(f"{'Rank':<6} {'Variable':<35} {'Lag':<6} {'Correlation':<15} {'P-value':<15} {'Significant':<12}")
    print("-" * 90)
    
    regime_results = []
    
    for var in macro_cols:
        if var not in data.columns:
            continue
        
        best_lag = None
        best_corr = 0
        best_pval = 1.0
        best_lag_num = None
        best_n_obs = 0
        
        # Test all lags
        for lag in [1, 2, 3]:
            lag_var = f'{var}_lag{lag}'
            if lag_var not in regime_data.columns:
                continue
            
            test_data = regime_data[['erp', lag_var]].dropna()
            if len(test_data) < 10:
                continue
            
            # Calculate weighted correlation if using probabilities
            if weights is not None:
                test_weights = regime_data.loc[test_data.index, prob_col].values
                test_weights = test_weights / test_weights.sum()
                
                x = test_data['erp'].values
                y = test_data[lag_var].values
                x_mean = np.average(x, weights=test_weights)
                y_mean = np.average(y, weights=test_weights)
                cov_xy = np.average((x - x_mean) * (y - y_mean), weights=test_weights)
                var_x = np.average((x - x_mean) ** 2, weights=test_weights)
                var_y = np.average((y - y_mean) ** 2, weights=test_weights)
                
                if var_x > 0 and var_y > 0:
                    corr = cov_xy / np.sqrt(var_x * var_y)
                else:
                    corr = 0.0
                
                n_eff_test = (np.sum(test_weights) ** 2) / np.sum(test_weights ** 2)
                if n_eff_test > 2 and abs(corr) < 1:
                    t_stat = corr * np.sqrt((n_eff_test - 2) / (1 - corr ** 2))
                    pval = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff_test - 2))
                else:
                    pval = 1.0
            else:
                corr, pval = stats.pearsonr(test_data['erp'], test_data[lag_var])
                n_eff_test = len(test_data)
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_pval = pval
                best_lag_num = lag
                best_lag = lag_var
                best_n_obs = n_eff_test if weights is not None else len(test_data)
        
        if best_lag is not None:
            regime_results.append({
                'variable': var,
                'lag': best_lag_num,
                'correlation': best_corr,
                'pvalue': best_pval,
                'abs_correlation': abs(best_corr),
                'n_observations': best_n_obs
            })
    
    df_regime = pd.DataFrame(regime_results).sort_values('abs_correlation', ascending=False)
    
    for rank, (idx, row) in enumerate(df_regime.head(15).iterrows(), 1):
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        significant = 'Yes' if row['pvalue'] < 0.05 else 'No'
        
        print(f"{rank:<6} {row['variable']:<35} {int(row['lag']):<6} {row['correlation']:>10.3f} {sig:<4} {row['pvalue']:>12.2e} {significant:<12}")
    
    print()
    print(f"Top 5 Variables:")
    for rank, (idx, row) in enumerate(df_regime.head(5).iterrows(), 1):
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        print(f"  {rank}. {row['variable']:<35} lag {int(row['lag'])}: corr={row['correlation']:>7.3f}, p={row['pvalue']:>8.4f} {sig}")
    print()
    
    df_regime['regime'] = regime
    df_regime['regime_name'] = regime_name
    all_results.append(df_regime)
    
    # Save
    df_regime.to_csv(output_dir / f'monthly_lags_regime_{int(regime)}.csv', index=False)

df_all_monthly = pd.concat(all_results, ignore_index=True)

print("=" * 90)
print("2. QUARTERLY ANALYSIS")
print("=" * 90)
print()

# Aggregate to quarterly
data_quarterly = data.copy()
data_quarterly['erp_quarterly'] = data_quarterly['erp'].resample('Q').sum()

for var in macro_cols:
    if var in data.columns:
        data_quarterly[f'{var}_quarterly'] = data_quarterly[var].resample('Q').last()
        data_quarterly[f'{var}_quarterly_lag1'] = data_quarterly[f'{var}_quarterly'].shift(1)

quarterly_results = []

for regime in sorted(data_quarterly['regime'].unique()):
    # Use regime probabilities
    if hasattr(analyzer, 'regime_prob_cols') and regime in analyzer.regime_prob_cols:
        prob_col = analyzer.regime_prob_cols[regime]
        regime_data = data_quarterly[data_quarterly[prob_col] > 0.01].copy()
        weights = regime_data[prob_col].values
    else:
        regime_data = data_quarterly[data_quarterly['regime'] == regime].copy()
        weights = None
    
    regime_name = regime_names[regime]
    
    if len(regime_data) < 10:
        continue
    
    print(f"REGIME {int(regime)}: {regime_name}")
    print(f"Observations: {len(regime_data)}")
    print("-" * 90)
    print(f"{'Rank':<6} {'Variable':<35} {'Correlation':<15} {'P-value':<15} {'Significant':<12}")
    print("-" * 90)
    
    regime_q_results = []
    
    for var in macro_cols:
        lag_var = f'{var}_quarterly_lag1'
        if lag_var not in regime_data.columns:
            continue
        
        test_data = regime_data[['erp_quarterly', lag_var]].dropna()
        if len(test_data) < 8:
            continue
        
        # Calculate weighted correlation if using probabilities
        if weights is not None:
            test_weights = regime_data.loc[test_data.index, prob_col].values
            test_weights = test_weights / test_weights.sum()
            
            x = test_data['erp_quarterly'].values
            y = test_data[lag_var].values
            x_mean = np.average(x, weights=test_weights)
            y_mean = np.average(y, weights=test_weights)
            cov_xy = np.average((x - x_mean) * (y - y_mean), weights=test_weights)
            var_x = np.average((x - x_mean) ** 2, weights=test_weights)
            var_y = np.average((y - y_mean) ** 2, weights=test_weights)
            
            if var_x > 0 and var_y > 0:
                corr = cov_xy / np.sqrt(var_x * var_y)
            else:
                corr = 0.0
            
            n_eff_test = (np.sum(test_weights) ** 2) / np.sum(test_weights ** 2)
            if n_eff_test > 2 and abs(corr) < 1:
                t_stat = corr * np.sqrt((n_eff_test - 2) / (1 - corr ** 2))
                pval = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff_test - 2))
            else:
                pval = 1.0
        else:
            corr, pval = stats.pearsonr(test_data['erp_quarterly'], test_data[lag_var])
        
        regime_q_results.append({
            'variable': var,
            'correlation': corr,
            'pvalue': pval,
            'abs_correlation': abs(corr),
            'n_observations': len(test_data)
        })
    
    if len(regime_q_results) == 0:
        print("  No data available")
        print()
        continue
    
    df_q_regime = pd.DataFrame(regime_q_results).sort_values('abs_correlation', ascending=False)
    
    for rank, (idx, row) in enumerate(df_q_regime.head(10).iterrows(), 1):
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        significant = 'Yes' if row['pvalue'] < 0.05 else 'No'
        
        print(f"{rank:<6} {row['variable']:<35} {row['correlation']:>10.3f} {sig:<4} {row['pvalue']:>12.2e} {significant:<12}")
    
    print()
    
    df_q_regime['regime'] = regime
    df_q_regime['regime_name'] = regime_name
    quarterly_results.append(df_q_regime)
    
    # Save
    df_q_regime.to_csv(output_dir / f'quarterly_regime_{int(regime)}.csv', index=False)

if len(quarterly_results) > 0:
    df_all_quarterly = pd.concat(quarterly_results, ignore_index=True)
else:
    df_all_quarterly = pd.DataFrame()

print("=" * 90)
print("3. SUMMARY: SIGNIFICANT PREDICTORS")
print("=" * 90)
print()

print("MONTHLY ANALYSIS (with best lag):")
print("-" * 90)
sig_monthly = df_all_monthly[df_all_monthly['pvalue'] < 0.05].sort_values('pvalue')
if len(sig_monthly) > 0:
    for _, row in sig_monthly.iterrows():
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*'
        print(f"  Regime {int(row['regime'])} ({row['regime_name']}):")
        print(f"    {row['variable']:<35} lag {int(row['lag'])}: corr={row['correlation']:>7.3f}, p={row['pvalue']:>8.4f} {sig}")
        print()
else:
    print("  No statistically significant predictors (p < 0.05)")
    print()

print("QUARTERLY ANALYSIS:")
print("-" * 90)
if len(df_all_quarterly) > 0:
    sig_quarterly = df_all_quarterly[df_all_quarterly['pvalue'] < 0.05].sort_values('pvalue')
    if len(sig_quarterly) > 0:
        for _, row in sig_quarterly.iterrows():
            sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*'
            print(f"  Regime {int(row['regime'])} ({row['regime_name']}):")
            print(f"    {row['variable']:<35}: corr={row['correlation']:>7.3f}, p={row['pvalue']:>8.4f} {sig}")
            print()
    else:
        print("  No statistically significant predictors (p < 0.05)")
        print()
else:
    print("  No quarterly data available")
    print()

print("=" * 90)
print("TOP 5 VARIABLES BY REGIME (MONTHLY, BEST LAG)")
print("=" * 90)
print()

for regime in sorted(data['regime'].unique()):
    regime_data = df_all_monthly[df_all_monthly['regime'] == regime].sort_values('abs_correlation', ascending=False)
    regime_name = regime_names[regime]
    
    print(f"REGIME {int(regime)}: {regime_name}")
    print("-" * 90)
    for rank, (idx, row) in enumerate(regime_data.head(5).iterrows(), 1):
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        print(f"  {rank}. {row['variable']:<35} lag {int(row['lag'])}: corr={row['correlation']:>7.3f}, p={row['pvalue']:>8.4f} {sig}")
    print()

print(f"Results saved to: {output_dir}")
print()

