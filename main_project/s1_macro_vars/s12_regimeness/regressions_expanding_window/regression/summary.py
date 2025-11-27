"""
Summary Generation for Regime-Conditional Regression Results

This module contains functions to generate comprehensive summary reports
of regression analysis results.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def create_summary(regressor, output_dir: Optional[Path] = None):
    """
    Create a summary markdown file with key findings.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for summary
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    summary_path = output_dir / 'SUMMARY.md'
    
    print("  Creating summary report...")
    
    with open(summary_path, 'w') as f:
        f.write(f"# Regime-Conditional Regression Analysis Summary\n\n")
        f.write(f"**Model**: {regressor.regime_model.upper()}\n\n")
        f.write(f"## Overview\n\n")
        f.write(f"This analysis identifies which macro variables are most significant ")
        f.write(f"for predicting ERP in different economic regimes.\n\n")
        
        if regressor.regression_results is not None and len(regressor.regression_results) > 0:
            f.write(f"## Key Findings\n\n")
            
            # Overall statistics
            total_regressions = len(regressor.regression_results)
            significant = (regressor.regression_results['p_value'] < 0.05).sum()
            highly_significant = (regressor.regression_results['p_value'] < 0.01).sum()
            
            f.write(f"### Overall Statistics\n\n")
            f.write(f"- Total regressions run: {total_regressions}\n")
            f.write(f"- Significant results (p < 0.05): {significant} ({100*significant/total_regressions:.1f}%)\n")
            f.write(f"- Highly significant (p < 0.01): {highly_significant} ({100*highly_significant/total_regressions:.1f}%)\n\n")
            
            # Find most significant variables by regime
            for horizon in sorted(regressor.regression_results['horizon'].unique()):
                f.write(f"### Forecast Horizon: {horizon} month(s)\n\n")
                
                horizon_data = regressor.regression_results[
                    regressor.regression_results['horizon'] == horizon
                ].copy()
                
                # Rank by absolute t-statistic
                horizon_data['abs_tstat'] = horizon_data['t_statistic'].abs()
                horizon_data = horizon_data.sort_values('abs_tstat', ascending=False)
                
                for regime in sorted(horizon_data['regime'].unique()):
                    regime_data = horizon_data[horizon_data['regime'] == regime].copy()
                    regime_name = regime_data['regime_name'].iloc[0] if 'regime_name' in regime_data.columns else f"Regime {regime}"
                    
                    f.write(f"#### {regime_name}\n\n")
                    f.write(f"Top 5 most significant predictors:\n\n")
                    f.write(f"| Variable | Coefficient | t-stat | p-value | R² |\n")
                    f.write(f"|----------|-------------|--------|---------|-----|\n")
                    
                    top5 = regime_data.head(5)
                    for _, row in top5.iterrows():
                        sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.10 else ''
                        f.write(f"| {row['variable']} | {row['coefficient']:.4f}{sig} | {row['t_statistic']:.2f} | {row['p_value']:.4f} | {row['r_squared']:.4f} |\n")
                    
                    f.write(f"\n")
        
        if regressor.statistical_tests is not None and len(regressor.statistical_tests) > 0:
            f.write(f"## Coefficient Differences Across Regimes\n\n")
            sig_tests = regressor.statistical_tests[regressor.statistical_tests['significant'] == True].copy()
            
            if len(sig_tests) > 0:
                f.write(f"Found {len(sig_tests)} significant coefficient differences:\n\n")
                f.write(f"| Variable | Horizon | Regime1 | Regime2 | Difference | p-value |\n")
                f.write(f"|----------|---------|---------|---------|------------|----------|\n")
                
                for _, row in sig_tests.head(20).iterrows():
                    f.write(f"| {row['variable']} | {row['horizon']} | {row['regime1']} | {row['regime2']} | {row['difference']:.4f} | {row['p_value']:.4f} |\n")
            else:
                f.write(f"No significant coefficient differences found.\n")
    
    print(f"    Saved: SUMMARY.md")


def create_executive_summary(regressor, output_dir: Optional[Path] = None):
    """
    Create an executive summary with key takeaways.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for summary
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating executive summary...")
    
    results = regressor.regression_results.copy()
    results['significant'] = results['p_value'] < 0.05
    results['highly_sig'] = results['p_value'] < 0.01
    results['abs_tstat'] = results['t_statistic'].abs()
    
    summary_path = output_dir / 'EXECUTIVE_SUMMARY.md'
    
    with open(summary_path, 'w') as f:
        f.write("# Executive Summary: Regime-Conditional Regression Analysis\n\n")
        f.write(f"**Model**: {regressor.regime_model.upper()}\n\n")
        f.write("## Key Takeaways\n\n")
        
        # Overall performance
        total = len(results)
        n_sig = results['significant'].sum()
        n_highly_sig = results['highly_sig'].sum()
        
        f.write(f"### Overall Performance\n\n")
        f.write(f"- **Total regressions**: {total}\n")
        f.write(f"- **Significant results (p<0.05)**: {n_sig} ({100*n_sig/total:.1f}%)\n")
        f.write(f"- **Highly significant (p<0.01)**: {n_highly_sig} ({100*n_highly_sig/total:.1f}%)\n\n")
        
        # Best predictors overall
        var_performance = results.groupby('variable').agg({
            'abs_tstat': 'mean',
            'significant': 'sum',
            'r_squared': 'mean'
        }).sort_values('abs_tstat', ascending=False)
        
        f.write(f"### Top 5 Overall Predictors\n\n")
        f.write(f"| Variable | Avg |t-stat| | Significant Results | Avg R² |\n")
        f.write(f"|----------|----------------|---------------------|--------|\n")
        for var in var_performance.head(5).index:
            row = var_performance.loc[var]
            f.write(f"| {var} | {row['abs_tstat']:.2f} | {int(row['significant'])} | {row['r_squared']:.4f} |\n")
        f.write(f"\n")
        
        # Best predictors by regime
        f.write(f"### Best Predictors by Regime\n\n")
        for regime in sorted(results['regime'].unique()):
            regime_data = results[results['regime'] == regime]
            regime_name = regime_data['regime_name'].iloc[0] if 'regime_name' in regime_data.columns else f"Regime {regime}"
            
            var_avg = regime_data.groupby('variable')['abs_tstat'].mean().sort_values(ascending=False)
            top_var = var_avg.index[0]
            top_tstat = var_avg.iloc[0]
            
            f.write(f"**{regime_name}**: {top_var} (avg |t-stat| = {top_tstat:.2f})\n\n")
        
        # Regime differences
        if regressor.statistical_tests is not None and len(regressor.statistical_tests) > 0:
            sig_tests = regressor.statistical_tests[regressor.statistical_tests['significant'] == True]
            if len(sig_tests) > 0:
                f.write(f"### Regime-Dependent Relationships\n\n")
                f.write(f"- **{len(sig_tests)} significant coefficient differences** found across regimes\n")
                f.write(f"- Indicates that macro variables have **different predictive power** in different regimes\n\n")
        
        # Horizon effects
        horizon_perf = results.groupby('horizon').agg({
            'significant': 'sum',
            'r_squared': 'mean'
        })
        
        f.write(f"### Forecast Horizon Performance\n\n")
        f.write(f"| Horizon | Significant Results | Avg R² |\n")
        f.write(f"|---------|---------------------|--------|\n")
        for horizon in sorted(horizon_perf.index):
            row = horizon_perf.loc[horizon]
            f.write(f"| {horizon} month(s) | {int(row['significant'])} | {row['r_squared']:.4f} |\n")
        f.write(f"\n")
        
        # Recommendations
        f.write(f"## Recommendations\n\n")
        f.write(f"1. **Focus on top predictors** identified above for each regime\n")
        f.write(f"2. **Use regime-specific models** - coefficients differ significantly across regimes\n")
        f.write(f"3. **Consider forecast horizon** - predictive power varies by horizon\n")
        f.write(f"4. **Weight by probabilities** - Use Mahalanobis distance probabilities for soft regime assignments\n")
    
    print(f"    Saved: EXECUTIVE_SUMMARY.md")


def create_detailed_variable_analysis(regressor, output_dir: Optional[Path] = None):
    """
    Create detailed analysis for each variable.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for summary
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating detailed variable analysis...")
    
    results = regressor.regression_results.copy()
    results['abs_tstat'] = results['t_statistic'].abs()
    results['significant'] = results['p_value'] < 0.05
    
    var_analysis = []
    
    for var in sorted(results['variable'].unique()):
        var_data = results[results['variable'] == var].copy()
        
        # Overall stats
        n_sig = var_data['significant'].sum()
        avg_tstat = var_data['abs_tstat'].mean()
        avg_r2 = var_data['r_squared'].mean()
        avg_coef = var_data['coefficient'].mean()
        
        # Best regime
        regime_perf = var_data.groupby('regime')['abs_tstat'].mean()
        best_regime = regime_perf.idxmax()
        best_regime_name = var_data[var_data['regime'] == best_regime]['regime_name'].iloc[0] if len(var_data[var_data['regime'] == best_regime]) > 0 else f"R{best_regime}"
        best_tstat = regime_perf.max()
        
        # Best horizon
        horizon_perf = var_data.groupby('horizon')['abs_tstat'].mean()
        best_horizon = horizon_perf.idxmax()
        best_horizon_tstat = horizon_perf.max()
        
        var_analysis.append({
            'variable': var,
            'n_significant': n_sig,
            'avg_abs_tstat': avg_tstat,
            'avg_r_squared': avg_r2,
            'avg_coefficient': avg_coef,
            'best_regime': best_regime_name,
            'best_regime_tstat': best_tstat,
            'best_horizon': best_horizon,
            'best_horizon_tstat': best_horizon_tstat
        })
    
    var_df = pd.DataFrame(var_analysis).sort_values('avg_abs_tstat', ascending=False)
    var_df.to_csv(output_dir / 'detailed_variable_analysis.csv', index=False)
    print(f"    Saved: detailed_variable_analysis.csv")


def create_regime_comparison_table(regressor, output_dir: Optional[Path] = None):
    """
    Create comparison table showing top predictors for each regime.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for summary
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating regime comparison table...")
    
    results = regressor.regression_results.copy()
    results['abs_tstat'] = results['t_statistic'].abs()
    
    # Get top 3 predictors for each regime (across all horizons)
    regimes = sorted(results['regime'].unique())
    comparison_data = []
    
    for regime in regimes:
        regime_data = results[results['regime'] == regime].copy()
        regime_name = regime_data['regime_name'].iloc[0] if 'regime_name' in regime_data.columns else f"Regime {regime}"
        
        var_avg = regime_data.groupby('variable')['abs_tstat'].mean().sort_values(ascending=False)
        top_vars = var_avg.head(3)
        
        for rank, (var, tstat) in enumerate(top_vars.items(), 1):
            # Get average coefficient and p-value for this variable in this regime
            var_regime_data = regime_data[regime_data['variable'] == var]
            avg_coef = var_regime_data['coefficient'].mean()
            avg_pval = var_regime_data['p_value'].mean()
            
            comparison_data.append({
                'regime': regime_name,
                'rank': rank,
                'variable': var,
                'avg_abs_tstat': tstat,
                'avg_coefficient': avg_coef,
                'avg_p_value': avg_pval
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'regime_comparison_table.csv', index=False)
    print(f"    Saved: regime_comparison_table.csv")


def create_statistics_summary(regressor, output_dir: Optional[Path] = None):
    """
    Create a detailed statistics summary CSV file.
    
    Parameters:
    -----------
    regressor : RegimeConditionalRegressor
        Regressor object with results
    output_dir : Path, optional
        Output directory for summary
    """
    if output_dir is None:
        output_dir = regressor.output_dir
    
    print("  Creating statistics summary...")
    
    results = regressor.regression_results.copy()
    
    # Summary statistics by variable
    var_summary = results.groupby('variable').agg({
        'coefficient': ['mean', 'std', 'min', 'max'],
        't_statistic': ['mean', 'std', 'min', 'max'],
        'p_value': ['mean', 'min'],
        'r_squared': ['mean', 'std', 'min', 'max'],
        'n_observations': 'mean'
    }).round(4)
    
    var_summary.columns = ['_'.join(col).strip() for col in var_summary.columns.values]
    var_summary = var_summary.sort_values('p_value_mean', ascending=True)
    
    # Count significant results
    results['significant'] = results['p_value'] < 0.05
    sig_counts = results.groupby('variable')['significant'].sum()
    var_summary['n_significant'] = sig_counts
    
    var_summary.to_csv(output_dir / 'variable_statistics_summary.csv')
    print(f"    Saved: variable_statistics_summary.csv")
    
    # Summary by regime
    regime_summary = results.groupby(['regime', 'regime_name']).agg({
        'coefficient': ['mean', 'std'],
        't_statistic': ['mean', 'std'],
        'p_value': 'mean',
        'r_squared': ['mean', 'std'],
        'n_observations': 'mean'
    }).round(4)
    
    regime_summary.columns = ['_'.join(col).strip() for col in regime_summary.columns.values]
    regime_summary.to_csv(output_dir / 'regime_statistics_summary.csv')
    print(f"    Saved: regime_statistics_summary.csv")
    
    # Summary by horizon
    horizon_summary = results.groupby('horizon').agg({
        'coefficient': ['mean', 'std'],
        't_statistic': ['mean', 'std'],
        'p_value': 'mean',
        'r_squared': ['mean', 'std'],
        'n_observations': 'mean'
    }).round(4)
    
    horizon_summary.columns = ['_'.join(col).strip() for col in horizon_summary.columns.values]
    horizon_summary.to_csv(output_dir / 'horizon_statistics_summary.csv')
    print(f"    Saved: horizon_statistics_summary.csv")
    
    # Create additional summaries
    create_executive_summary(regressor, output_dir)
    create_detailed_variable_analysis(regressor, output_dir)
    create_regime_comparison_table(regressor, output_dir)

