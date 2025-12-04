#!/usr/bin/env python3
"""
Systematic HMM Model Testing

Tests all combinations of:
- Variable sets: All 4 variables + all 6 combinations of 2 variables
- Regime numbers: K = 2, 3, 4, 5, 6, 7, 8, 9, 10

Compares models using AIC and BIC to identify optimal specifications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import warnings
from itertools import combinations
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Add current directory and shared utilities to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REGIMES_DIR = SCRIPT_DIR.parent
SECTION_DIR = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REGIMES_DIR, SECTION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from hmm_model import HMMRegimeModel
from path_utils import get_data_dir


# Define all variable combinations to test
ALL_VARIABLES = [
    'growth_factor',
    'inflation_factor',
    'monetary_policy_factor',
    'market_volatility_factor'
]

def get_variable_combinations() -> List[Tuple[str, List[str]]]:
    """
    Generate all variable combinations to test.
    
    Returns:
    --------
    List of tuples: (combination_name, variable_list)
    """
    combinations_list = []
    
    # All 4 variables
    combinations_list.append(('all_4vars', ALL_VARIABLES))
    
    # All combinations of 2 variables (6 total)
    for combo in combinations(ALL_VARIABLES, 2):
        # Create readable name
        var_names = [v.replace('_factor', '') for v in combo]
        combo_name = '_'.join(sorted(var_names))
        combinations_list.append((f'2vars_{combo_name}', list(combo)))
    
    return combinations_list


def test_hmm_model(
    data: pd.DataFrame,
    variables: List[str],
    k_values: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
    n_init: int = 5
) -> pd.DataFrame:
    """
    Test HMM model for a given variable combination across different K values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Combined macro and ERP data
    variables : List[str]
        List of variable names to use
    k_values : List[int]
        List of K (number of regimes) values to test
    n_init : int
        Number of random initializations per K
    
    Returns:
    --------
    pd.DataFrame: Results with AIC, BIC, log-likelihood for each K
    """
    results = []
    
    # Prepare features
    feature_data = data[variables].dropna()
    if len(feature_data) == 0:
        return pd.DataFrame()
    
    scaler = StandardScaler()
    features = scaler.fit_transform(feature_data.values)
    
    for k in k_values:
        try:
            model = HMMRegimeModel(
                n_regimes=k,
                variables=variables,
                random_state=42
            )
            model.scaler = scaler
            
            # Fit model
            model.fit(features, n_init=n_init)
            
            # Calculate metrics
            metrics = model.calculate_model_metrics(features)
            
            results.append({
                'K': k,
                'AIC': metrics['AIC'],
                'BIC': metrics['BIC'],
                'log_likelihood': metrics['log_likelihood'],
                'n_params': metrics['n_params'],
                'n_samples': metrics['n_samples']
            })
            
        except Exception as e:
            print(f"    Error fitting K={k}: {e}")
            continue
    
    return pd.DataFrame(results)


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load macro and ERP data."""
    print("Loading data...")
    
    # Load macro data
    macro_file = data_dir / 'macro_final' / 'final_macro.csv'
    if not macro_file.exists():
        raise FileNotFoundError(f"Macro file not found: {macro_file}")
    
    macro_df = pd.read_csv(macro_file)
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df = macro_df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(macro_df)} macro observations")
    
    # Load ERP data
    erp_file = None
    for path in [
        data_dir / 'macro_processed' / 'equity_risk_pr.csv',
        data_dir.parent / 'main_project' / 'data' / 'macro_processed' / 'equity_risk_pr.csv'
    ]:
        if path.exists():
            erp_file = path
            break
    
    if erp_file is None:
        raise FileNotFoundError("ERP file not found")
    
    erp_df = pd.read_csv(erp_file)
    erp_df['date'] = pd.to_datetime(erp_df['date'])
    erp_df = erp_df.rename(columns={'ERP': 'erp'})
    erp_df = erp_df.dropna(subset=['erp']).reset_index(drop=True)
    print(f"  Loaded {len(erp_df)} ERP observations")
    
    # Combine data
    macro_df['date_month'] = pd.to_datetime(macro_df['date']).dt.to_period('M').dt.end_time
    erp_df['date_month'] = pd.to_datetime(erp_df['date']).dt.to_period('M').dt.end_time
    
    combined = pd.merge(
        macro_df,
        erp_df[['date_month', 'erp']],
        left_on='date_month',
        right_on='date_month',
        how='inner'
    )
    
    combined = combined.sort_values('date').reset_index(drop=True)
    print(f"  Combined dataset: {len(combined)} observations")
    
    return macro_df, combined


def main():
    """Run systematic HMM model testing."""
    print("="*80)
    print("SYSTEMATIC HMM MODEL TESTING")
    print("="*80)
    print("\nTesting all variable combinations and regime numbers:")
    print("  - Variable sets: All 4 variables + all 6 combinations of 2 variables")
    print("  - Regime numbers: K = 2, 3, 4, 5, 6")
    print("  - Comparison metrics: AIC and BIC")
    print()
    
    # Set up paths
    data_dir = get_data_dir(__file__)
    output_dir = SCRIPT_DIR / 'results_systematic'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    macro_df, combined_data = load_data(data_dir)
    
    # Get all variable combinations
    var_combinations = get_variable_combinations()
    
    print(f"\nTesting {len(var_combinations)} variable combinations...")
    print(f"  Each combination tested with K = 2, 3, 4, 5, 6, 7, 8, 9, 10")
    print(f"  Total models to test: {len(var_combinations) * 9}")
    print()
    
    # Store all results
    all_results = []
    
    # Test each variable combination
    for combo_idx, (combo_name, variables) in enumerate(var_combinations, 1):
        print(f"\n{'='*80}")
        print(f"Combination {combo_idx}/{len(var_combinations)}: {combo_name}")
        print(f"Variables: {', '.join(variables)}")
        print(f"{'='*80}")
        
        # Test different K values
        results_df = test_hmm_model(
            combined_data,
            variables,
            k_values=[2, 3, 4, 5, 6, 7, 8, 9, 10],
            n_init=5
        )
        
        if len(results_df) > 0:
            # Add combination info
            results_df['combination'] = combo_name
            results_df['variables'] = ', '.join(variables)
            results_df['n_variables'] = len(variables)
            
            # Find best K by AIC and BIC
            best_aic = results_df.loc[results_df['AIC'].idxmin()]
            best_bic = results_df.loc[results_df['BIC'].idxmin()]
            
            results_df['best_AIC'] = (results_df['K'] == best_aic['K'])
            results_df['best_BIC'] = (results_df['K'] == best_bic['K'])
            
            all_results.append(results_df)
            
            print(f"\n  Results:")
            print(f"    Best AIC: K={best_aic['K']}, AIC={best_aic['AIC']:.2f}")
            print(f"    Best BIC: K={best_bic['K']}, BIC={best_bic['BIC']:.2f}")
        else:
            print(f"  No results for this combination")
    
    # Combine all results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Save comprehensive results
        results_file = output_dir / 'all_model_results.csv'
        final_results.to_csv(results_file, index=False)
        print(f"\n{'='*80}")
        print(f"Saved all results to: {results_file}")
        
        # Create summary: best model for each combination
        summary = []
        for combo_name in final_results['combination'].unique():
            combo_results = final_results[final_results['combination'] == combo_name]
            best_aic = combo_results.loc[combo_results['AIC'].idxmin()]
            best_bic = combo_results.loc[combo_results['BIC'].idxmin()]
            
            summary.append({
                'combination': combo_name,
                'variables': best_aic['variables'],
                'n_variables': best_aic['n_variables'],
                'best_K_AIC': best_aic['K'],
                'best_AIC': best_aic['AIC'],
                'best_K_BIC': best_bic['K'],
                'best_BIC': best_bic['BIC'],
                'AIC_loglik': best_aic['log_likelihood'],
                'BIC_loglik': best_bic['log_likelihood']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_file = output_dir / 'model_comparison_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to: {summary_file}")
        
        # Find overall best models
        overall_best_aic = final_results.loc[final_results['AIC'].idxmin()]
        overall_best_bic = final_results.loc[final_results['BIC'].idxmin()]
        
        print(f"\n{'='*80}")
        print("OVERALL BEST MODELS")
        print(f"{'='*80}")
        print(f"\nBest AIC:")
        print(f"  Combination: {overall_best_aic['combination']}")
        print(f"  Variables: {overall_best_aic['variables']}")
        print(f"  K: {overall_best_aic['K']}")
        print(f"  AIC: {overall_best_aic['AIC']:.2f}")
        print(f"  BIC: {overall_best_aic['BIC']:.2f}")
        
        print(f"\nBest BIC:")
        print(f"  Combination: {overall_best_bic['combination']}")
        print(f"  Variables: {overall_best_bic['variables']}")
        print(f"  K: {overall_best_bic['K']}")
        print(f"  AIC: {overall_best_bic['AIC']:.2f}")
        print(f"  BIC: {overall_best_bic['BIC']:.2f}")
        
        # Save best models info
        best_models = pd.DataFrame([
            {
                'criterion': 'AIC',
                'combination': overall_best_aic['combination'],
                'variables': overall_best_aic['variables'],
                'K': overall_best_aic['K'],
                'AIC': overall_best_aic['AIC'],
                'BIC': overall_best_aic['BIC'],
                'log_likelihood': overall_best_aic['log_likelihood']
            },
            {
                'criterion': 'BIC',
                'combination': overall_best_bic['combination'],
                'variables': overall_best_bic['variables'],
                'K': overall_best_bic['K'],
                'AIC': overall_best_bic['AIC'],
                'BIC': overall_best_bic['BIC'],
                'log_likelihood': overall_best_bic['log_likelihood']
            }
        ])
        best_models_file = output_dir / 'best_models.csv'
        best_models.to_csv(best_models_file, index=False)
        print(f"\nSaved best models to: {best_models_file}")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {output_dir}")
    else:
        print("\nNo results generated. Check data loading and model fitting.")


if __name__ == "__main__":
    main()

