"""
Regime-Conditional Analysis: Which Macro Variables Predict VIX?

This analysis uses LAGGED macro variables to understand what PREDICTS VIX
in each regime, addressing the hypothesis that different variables matter
in different economic regimes (e.g., inflation in some periods, growth in others).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from vix_macro_relevance import VIXMacroRelevanceAnalyzer

# Set up paths
project_root = Path(__file__).parent.parent
regime_assignments_path = project_root / 'test_4regimes_HMM' / 'results' / 'regime_assignments.csv'
macro_data_dir = project_root / 'data' / 'macro_processed'
vix_path = project_root / 'data' / 'macro_processed' / 'selection' / 'vix_processed.csv'
output_dir = Path(__file__).parent / 'results' / 'detailed_by_regime'
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("REGIME-CONDITIONAL PREDICTORS OF VIX")
print("=" * 90)
print()
print("Using LAGGED macro variables to predict VIX")
print("(Macro at t-1 predicts VIX at t)")
print()
print("Hypothesis: Different macro variables matter in different regimes")
print("  - Inflation might drive volatility in high-inflation regimes")
print("  - Growth might drive volatility in growth regimes")
print("  - Monetary policy might matter more in certain regimes")
print()

# Initialize analyzer
analyzer = VIXMacroRelevanceAnalyzer(
    regime_assignments_path=regime_assignments_path,
    macro_data_dir=macro_data_dir,
    vix_path=vix_path,
    output_dir=output_dir
)

# Load data
analyzer.load_regime_assignments()
analyzer.load_vix_data()
analyzer.load_macro_variables()
analyzer.combine_data()

data = analyzer.combined_data.copy()
macro_cols = list(analyzer.macro_data.keys())

# Exclude volatility variables - they're circular predictors (using volatility to predict volatility)
# These should already be excluded in load_macro_variables, but double-check here
volatility_vars = ['nasdaq_vol_indx', '3month_vol_index_sp500', 'vix']
macro_cols = [v for v in macro_cols if v not in volatility_vars]

print(f"Analyzing {len(macro_cols)} macro variables (excluding other volatility indices)")
print()

# Create lagged versions of macro variables (t-1 predicts t)
for var in macro_cols:
    if var in data.columns:
        data[f'{var}_lag1'] = data[var].shift(1)

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

print("=" * 90)
print("1. LAGGED CORRELATIONS BY REGIME")
print("=" * 90)
print()

results_by_regime = {}

for regime in sorted(data['regime'].unique()):
    regime_data = data[data['regime'] == regime].copy()
    
    if len(regime_data) < 20:
        continue
    
    regime_name = regime_data['regime_name'].iloc[0]
    
    # Use regime probabilities if available
    if hasattr(analyzer, 'regime_prob_cols') and regime in analyzer.regime_prob_cols:
        prob_col = analyzer.regime_prob_cols[regime]
        regime_data = data[data[prob_col] > 0.01].copy()
        weights = regime_data[prob_col].values
    else:
        weights = None
    
    regime_results = []
    
    for var in macro_cols:
        lag_var = f'{var}_lag1'
        if lag_var not in regime_data.columns:
            continue
        
        # Prepare data
        corr_data = regime_data[['vix', lag_var]].dropna()
        
        if len(corr_data) < 10:
            continue
        
        # Calculate correlation
        if weights is not None:
            # Weighted correlation
            corr_weights = regime_data.loc[corr_data.index, prob_col].values
            corr_weights = corr_weights / corr_weights.sum()
            
            x = corr_data['vix'].values
            y = corr_data[lag_var].values
            x_mean = np.average(x, weights=corr_weights)
            y_mean = np.average(y, weights=corr_weights)
            cov_xy = np.average((x - x_mean) * (y - y_mean), weights=corr_weights)
            var_x = np.average((x - x_mean) ** 2, weights=corr_weights)
            var_y = np.average((y - y_mean) ** 2, weights=corr_weights)
            
            if var_x > 0 and var_y > 0:
                corr = cov_xy / np.sqrt(var_x * var_y)
            else:
                corr = 0.0
            
            n_eff = (np.sum(corr_weights) ** 2) / np.sum(corr_weights ** 2)
            if n_eff > 2 and abs(corr) < 1:
                t_stat = corr * np.sqrt((n_eff - 2) / (1 - corr ** 2))
                pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff - 2))
            else:
                pvalue = 1.0
        else:
            corr, pvalue = stats.pearsonr(corr_data['vix'], corr_data[lag_var])
            n_eff = len(corr_data)
        
        regime_results.append({
            'variable': var,
            'correlation': corr,
            'pvalue': pvalue,
            'abs_correlation': abs(corr),
            'n_observations': len(corr_data),
            'effective_n': n_eff
        })
    
    df_regime = pd.DataFrame(regime_results).sort_values('abs_correlation', ascending=False)
    results_by_regime[regime] = {
        'name': regime_name,
        'results': df_regime
    }
    
    print(f"REGIME {int(regime)}: {regime_name}")
    print("-" * 90)
    print(f"{'Rank':<6} {'Variable':<35} {'Category':<20} {'Correlation':<15} {'P-value':<15} {'N':<8}")
    print("-" * 90)
    
    for rank, (idx, row) in enumerate(df_regime.head(15).iterrows(), 1):
        # Find category
        category = 'Other'
        for cat, vars_list in variable_categories.items():
            if row['variable'] in vars_list:
                category = cat
                break
        
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        
        print(f"{rank:<6} {row['variable']:<35} {category:<20} {row['correlation']:>8.3f} {sig:<4} {row['pvalue']:>12.2e} {int(row['effective_n']):>6}")
    
    print()
    print(f"Note: *** p<0.001, ** p<0.01, * p<0.05")
    print()

# Save results
for regime, regime_data in results_by_regime.items():
    regime_data['results'].to_csv(
        output_dir / f'lagged_correlations_regime_{int(regime)}.csv',
        index=False
    )

print("=" * 90)
print("2. REGRESSION ANALYSIS BY REGIME")
print("=" * 90)
print()

regression_results = {}

for regime, regime_data_dict in results_by_regime.items():
    regime_name = regime_data_dict['name']
    regime_data = data[data['regime'] == regime].copy()
    
    if hasattr(analyzer, 'regime_prob_cols') and regime in analyzer.regime_prob_cols:
        prob_col = analyzer.regime_prob_cols[regime]
        regime_data = data[data[prob_col] > 0.01].copy()
        weights = regime_data[prob_col].values
    else:
        weights = None
    
    if len(regime_data) < 20:
        continue
    
    reg_results = []
    
    for var in macro_cols:
        lag_var = f'{var}_lag1'
        if lag_var not in regime_data.columns:
            continue
        
        # Prepare data
        reg_data = regime_data[['vix', lag_var]].dropna()
        
        if len(reg_data) < 10:
            continue
        
        X = reg_data[[lag_var]].values
        y = reg_data['vix'].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit regression
        if weights is not None:
            reg_weights = regime_data.loc[reg_data.index, prob_col].values
            reg_weights = reg_weights / reg_weights.sum() * len(reg_weights)
        else:
            reg_weights = None
        
        model = LinearRegression()
        if reg_weights is not None:
            model.fit(X_scaled, y, sample_weight=reg_weights)
        else:
            model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        r2 = model.score(X_scaled, y)
        
        # Calculate p-value
        n = len(X_scaled)
        k = 1
        mse = np.mean((y - y_pred) ** 2)
        var_coef = mse / np.sum((X_scaled.flatten() - X_scaled.mean()) ** 2)
        se_coef = np.sqrt(var_coef) if var_coef > 0 else 0
        t_stat = model.coef_[0] / se_coef if se_coef > 0 else 0
        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1)) if n > k + 1 else 1.0
        
        reg_results.append({
            'variable': var,
            'coefficient': model.coef_[0],
            'r_squared': r2,
            'pvalue': pvalue,
            'n_observations': n
        })
    
    df_reg = pd.DataFrame(reg_results).sort_values('r_squared', ascending=False)
    regression_results[regime] = {
        'name': regime_name,
        'results': df_reg
    }
    
    print(f"REGIME {int(regime)}: {regime_name}")
    print("-" * 90)
    print(f"{'Rank':<6} {'Variable':<35} {'Category':<20} {'R²':<12} {'Coefficient':<15} {'P-value':<15}")
    print("-" * 90)
    
    for rank, (idx, row) in enumerate(df_reg.head(15).iterrows(), 1):
        category = 'Other'
        for cat, vars_list in variable_categories.items():
            if row['variable'] in vars_list:
                category = cat
                break
        
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        
        print(f"{rank:<6} {row['variable']:<35} {category:<20} {row['r_squared']:>8.3f} {row['coefficient']:>10.3f} {sig:<4} {row['pvalue']:>12.2e}")
    
    print()

# Save regression results
for regime, reg_data in regression_results.items():
    reg_data['results'].to_csv(
        output_dir / f'regressions_regime_{int(regime)}.csv',
        index=False
    )

print("=" * 90)
print("3. SUMMARY: TOP PREDICTORS BY REGIME AND CATEGORY")
print("=" * 90)
print()

# Create summary by category
summary_data = []

for regime, regime_data_dict in results_by_regime.items():
    regime_name = regime_data_dict['name']
    df_regime = regime_data_dict['results']
    
    for category, vars_list in variable_categories.items():
        category_vars = df_regime[df_regime['variable'].isin(vars_list)]
        
        if len(category_vars) > 0:
            best_var = category_vars.iloc[0]
            summary_data.append({
                'regime': int(regime),
                'regime_name': regime_name,
                'category': category,
                'top_variable': best_var['variable'],
                'correlation': best_var['correlation'],
                'abs_correlation': best_var['abs_correlation'],
                'pvalue': best_var['pvalue'],
                'rank': (df_regime['abs_correlation'] > best_var['abs_correlation']).sum() + 1
            })

df_summary = pd.DataFrame(summary_data)

print("TOP VARIABLE BY CATEGORY IN EACH REGIME:")
print("-" * 90)
print(f"{'Regime':<8} {'Category':<20} {'Top Variable':<35} {'Correlation':<15} {'Rank':<8}")
print("-" * 90)

for regime in sorted(df_summary['regime'].unique()):
    regime_summary = df_summary[df_summary['regime'] == regime].sort_values('abs_correlation', ascending=False)
    regime_name = regime_summary['regime_name'].iloc[0]
    
    print(f"\nRegime {int(regime)}: {regime_name}")
    for _, row in regime_summary.iterrows():
        sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
        print(f"  {row['category']:<20} {row['top_variable']:<35} {row['correlation']:>8.3f} {sig:<4} (rank {int(row['rank'])})")

df_summary.to_csv(output_dir / 'summary_by_category.csv', index=False)

print()
print("=" * 90)
print("4. KEY INSIGHTS")
print("=" * 90)
print()

# Compare inflation variables across regimes
print("INFLATION VARIABLES ACROSS REGIMES:")
print("-" * 90)
inflation_vars = variable_categories['Inflation']

for var in inflation_vars:
    print(f"\n{var}:")
    for regime in sorted(results_by_regime.keys()):
        regime_results = results_by_regime[regime]['results']
        var_result = regime_results[regime_results['variable'] == var]
        if len(var_result) > 0:
            row = var_result.iloc[0]
            rank = (regime_results['abs_correlation'] > row['abs_correlation']).sum() + 1
            total = len(regime_results)
            sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
            print(f"  Regime {int(regime)} ({results_by_regime[regime]['name']}): "
                  f"corr={row['correlation']:.3f} {sig}, rank {rank}/{total}")

print()
print("ECONOMIC GROWTH VARIABLES ACROSS REGIMES:")
print("-" * 90)
growth_vars = variable_categories['Economic Growth']

for var in growth_vars[:5]:  # Top 5 growth variables
    print(f"\n{var}:")
    for regime in sorted(results_by_regime.keys()):
        regime_results = results_by_regime[regime]['results']
        var_result = regime_results[regime_results['variable'] == var]
        if len(var_result) > 0:
            row = var_result.iloc[0]
            rank = (regime_results['abs_correlation'] > row['abs_correlation']).sum() + 1
            total = len(regime_results)
            sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
            print(f"  Regime {int(regime)} ({results_by_regime[regime]['name']}): "
                  f"corr={row['correlation']:.3f} {sig}, rank {rank}/{total}")

print()
print("MONETARY POLICY VARIABLES ACROSS REGIMES:")
print("-" * 90)
mon_policy_vars = variable_categories['Monetary Policy']

for var in mon_policy_vars:
    print(f"\n{var}:")
    for regime in sorted(results_by_regime.keys()):
        regime_results = results_by_regime[regime]['results']
        var_result = regime_results[regime_results['variable'] == var]
        if len(var_result) > 0:
            row = var_result.iloc[0]
            rank = (regime_results['abs_correlation'] > row['abs_correlation']).sum() + 1
            total = len(regime_results)
            sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
            print(f"  Regime {int(regime)} ({results_by_regime[regime]['name']}): "
                  f"corr={row['correlation']:.3f} {sig}, rank {rank}/{total}")

print()
print("=" * 90)
print("CONCLUSIONS")
print("=" * 90)
print()
print("This analysis uses LAGGED variables to identify what PREDICTS VIX")
print("in each regime, addressing your hypothesis that different variables")
print("matter in different economic environments.")
print()
print(f"Results saved to: {output_dir}")
print()

