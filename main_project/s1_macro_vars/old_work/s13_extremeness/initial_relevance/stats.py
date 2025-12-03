"""
Statistical analysis for extremeness models.

Computes statistics, tests, and comparisons between normal and extreme states.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def compute_erp_statistics(results_df, group_col='is_extreme'):
    """
    Compute ERP statistics for normal vs extreme states.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with ERP and group labels
    group_col : str
        Column name for grouping (default 'is_extreme')
    
    Returns:
    --------
    pd.DataFrame
        Statistics table
    """
    stats_list = []
    
    for group_name in [False, True]:
        group_data = results_df[results_df[group_col] == group_name]['ERP']
        group_label = 'extreme' if group_name else 'normal'
        
        stats_list.append({
            'state': group_label,
            'n_obs': len(group_data),
            'mean': group_data.mean(),
            'std': group_data.std(),
            'min': group_data.min(),
            'max': group_data.max(),
            'median': group_data.median(),
            'skewness': group_data.skew(),
            'kurtosis': group_data.kurtosis(),
            'p5': group_data.quantile(0.05),
            'p25': group_data.quantile(0.25),
            'p75': group_data.quantile(0.75),
            'p95': group_data.quantile(0.95),
            'p1': group_data.quantile(0.01),
            'p99': group_data.quantile(0.99)
        })
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def test_erp_differences(results_df, group_col='is_extreme'):
    """
    Perform statistical tests comparing ERP distributions.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with ERP and group labels
    group_col : str
        Column name for grouping
    
    Returns:
    --------
    dict
        Dictionary with test results
    """
    normal_erp = results_df[results_df[group_col] == False]['ERP'].values
    extreme_erp = results_df[results_df[group_col] == True]['ERP'].values
    
    # T-test for means
    t_stat, t_pvalue = scipy_stats.ttest_ind(normal_erp, extreme_erp)
    
    # Kolmogorov-Smirnov test for distributions
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(normal_erp, extreme_erp)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = scipy_stats.mannwhitneyu(normal_erp, extreme_erp, alternative='two-sided')
    
    # Tail quantile differences
    tail_diff_p5 = np.percentile(extreme_erp, 5) - np.percentile(normal_erp, 5)
    tail_diff_p1 = np.percentile(extreme_erp, 1) - np.percentile(normal_erp, 1)
    
    return {
        't_test': {
            'statistic': t_stat,
            'pvalue': t_pvalue,
            'normal_mean': normal_erp.mean(),
            'extreme_mean': extreme_erp.mean(),
            'mean_diff': extreme_erp.mean() - normal_erp.mean()
        },
        'ks_test': {
            'statistic': ks_stat,
            'pvalue': ks_pvalue
        },
        'mannwhitney_test': {
            'statistic': u_stat,
            'pvalue': u_pvalue
        },
        'tail_differences': {
            'p5_diff': tail_diff_p5,
            'p1_diff': tail_diff_p1
        }
    }


def compare_extremeness_measures(results_dict_1, results_dict_2, name_1, name_2):
    """
    Compare two extremeness measures (e.g., Isolation Forest vs PCA).
    
    Parameters:
    -----------
    results_dict_1 : dict
        Results from first model
    results_dict_2 : dict
        Results from second model
    name_1 : str
        Name of first model
    name_2 : str
        Name of second model
    
    Returns:
    --------
    dict
        Comparison statistics
    """
    extremeness_1 = results_dict_1['extremeness']
    extremeness_2 = results_dict_2['extremeness']
    is_extreme_1 = results_dict_1['is_extreme']
    is_extreme_2 = results_dict_2['is_extreme']
    
    # Correlation between extremeness scores
    correlation = np.corrcoef(extremeness_1, extremeness_2)[0, 1]
    
    # Overlap in extreme states
    overlap = np.sum(is_extreme_1 & is_extreme_2)
    total_extreme_1 = np.sum(is_extreme_1)
    total_extreme_2 = np.sum(is_extreme_2)
    overlap_rate = overlap / max(total_extreme_1, total_extreme_2) if max(total_extreme_1, total_extreme_2) > 0 else 0
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        f'{name_1}_extremeness': extremeness_1,
        f'{name_2}_extremeness': extremeness_2,
        f'{name_1}_is_extreme': is_extreme_1,
        f'{name_2}_is_extreme': is_extreme_2
    })
    
    return {
        'correlation': correlation,
        'overlap_count': overlap,
        'overlap_rate': overlap_rate,
        'total_extreme_1': total_extreme_1,
        'total_extreme_2': total_extreme_2,
        'comparison_df': comparison_df
    }


def compute_erp_statistics_by_percentiles(results_df, percentile_cols=['is_extreme_p99', 'is_extreme_p95', 'is_extreme_p90', 'is_extreme_p80']):
    """
    Compute ERP statistics for multiple percentile thresholds.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with ERP and percentile flags
    percentile_cols : list
        List of percentile flag column names
    
    Returns:
    --------
    pd.DataFrame
        Statistics table with one row per percentile threshold
    """
    stats_list = []
    
    for col in percentile_cols:
        if col in results_df.columns:
            # Extract percentile number from column name
            p = int(col.split('_')[-1].replace('p', ''))
            
            # Normal state (not extreme at this percentile)
            normal_data = results_df[results_df[col] == False]['ERP'].values
            # Extreme state (extreme at this percentile, but not higher percentiles)
            if p == 99:
                extreme_mask = results_df[col] == True
            elif p == 95:
                extreme_mask = (results_df[col] == True) & (~results_df.get('is_extreme_p99', pd.Series([False]*len(results_df))))
            elif p == 90:
                extreme_mask = (results_df[col] == True) & (~results_df.get('is_extreme_p95', pd.Series([False]*len(results_df))))
            elif p == 80:
                extreme_mask = (results_df[col] == True) & (~results_df.get('is_extreme_p90', pd.Series([False]*len(results_df))))
            else:
                extreme_mask = results_df[col] == True
            
            extreme_data = results_df[extreme_mask]['ERP'].values
            
            if len(normal_data) > 0 and len(extreme_data) > 0:
                stats_list.append({
                    'percentile': p,
                    'normal_n': len(normal_data),
                    'extreme_n': len(extreme_data),
                    'normal_mean': normal_data.mean(),
                    'extreme_mean': extreme_data.mean(),
                    'mean_diff': extreme_data.mean() - normal_data.mean(),
                    'normal_std': normal_data.std(),
                    'extreme_std': extreme_data.std(),
                    'normal_p5': np.percentile(normal_data, 5),
                    'extreme_p5': np.percentile(extreme_data, 5),
                    'normal_p95': np.percentile(normal_data, 95),
                    'extreme_p95': np.percentile(extreme_data, 95),
                })
    
    return pd.DataFrame(stats_list)


def create_summary_statistics(all_results):
    """
    Create comprehensive summary statistics for all models.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with results from all models
        Keys: model names, Values: result dictionaries
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics table
    """
    summary_rows = []
    
    for model_name, results in all_results.items():
        results_df = results['results_df']
        
        # Basic extremeness statistics
        extremeness = results['extremeness']
        is_extreme = results['is_extreme']
        
        summary_rows.append({
            'model': model_name,
            'n_obs': len(results_df),
            'n_extreme': np.sum(is_extreme),
            'pct_extreme': np.mean(is_extreme) * 100,
            'extremeness_mean': extremeness.mean(),
            'extremeness_std': extremeness.std(),
            'extremeness_min': extremeness.min(),
            'extremeness_max': extremeness.max(),
            'erp_normal_mean': results_df[results_df['is_extreme'] == False]['ERP'].mean(),
            'erp_extreme_mean': results_df[results_df['is_extreme'] == True]['ERP'].mean(),
            'erp_mean_diff': (results_df[results_df['is_extreme'] == True]['ERP'].mean() - 
                             results_df[results_df['is_extreme'] == False]['ERP'].mean())
        })
    
    return pd.DataFrame(summary_rows)

