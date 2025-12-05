"""
Create black and white LaTeX-style tables for slides from data files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# Use LaTeX-style fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def parse_in_out_sample_file(file_path: str) -> tuple:
    """
    Parse the in_out_sample.txt file and return DataFrames for in-sample and out-of-sample.
    
    Parameters:
    -----------
    file_path : str
        Path to the in_out_sample.txt file
    
    Returns:
    --------
    tuple
        (in_sample_df, out_of_sample_df)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the sections
    in_sample_start = None
    out_sample_start = None
    
    for i, line in enumerate(lines):
        if 'IN SAMPLE' in line:
            in_sample_start = i + 1
        elif 'OUT OF SAMPLE' in line:
            out_sample_start = i + 1
    
    # Parse in-sample
    in_sample_data = []
    if in_sample_start:
        for i in range(in_sample_start, len(lines)):
            line = lines[i].strip()
            if not line or 'OUT OF SAMPLE' in line:
                break
            if line and not line.startswith('='):
                parts = line.split()
                if len(parts) >= 4 and parts[0] != 'Model':
                    in_sample_data.append({
                        'Model': ' '.join(parts[:-3]),
                        'Regime': int(parts[-3]),
                        'R²': float(parts[-2]),
                        'RMSE': float(parts[-1])
                    })
    
    # Parse out-of-sample
    out_sample_data = []
    if out_sample_start:
        for i in range(out_sample_start, len(lines)):
            line = lines[i].strip()
            if not line:
                break
            if line and not line.startswith('='):
                parts = line.split()
                if len(parts) >= 4 and parts[0] != 'Model':
                    out_sample_data.append({
                        'Model': ' '.join(parts[:-3]),
                        'Regime': int(parts[-3]),
                        'R²': float(parts[-2]),
                        'RMSE': float(parts[-1])
                    })
    
    in_sample_df = pd.DataFrame(in_sample_data)
    out_sample_df = pd.DataFrame(out_sample_data)
    
    return in_sample_df, out_sample_df


def plot_in_out_sample_table(in_sample_df: pd.DataFrame, 
                              out_sample_df: pd.DataFrame,
                              output_path: str = None,
                              figsize: tuple = (10, 6)):
    """
    Create a LaTeX-style black and white table for in-sample and out-of-sample metrics.
    
    Parameters:
    -----------
    in_sample_df : pd.DataFrame
        In-sample metrics DataFrame
    out_sample_df : pd.DataFrame
        Out-of-sample metrics DataFrame
    output_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    
    # Format data for display
    def format_table_data(df):
        table_data = []
        for _, row in df.iterrows():
            table_data.append([
                row['Model'],
                f"R{row['Regime']}",
                f"{row['R²']:.4f}",
                f"{row['RMSE']:.4f}"
            ])
        return table_data
    
    # In-sample table
    in_table_data = format_table_data(in_sample_df)
    in_headers = ['Model', 'Regime', r'$R^2$', 'RMSE']
    
    in_table = ax1.table(
        cellText=in_table_data,
        colLabels=in_headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style in-sample table - LaTeX style with larger fonts
    in_table.auto_set_font_size(False)
    in_table.set_fontsize(15)
    in_table.scale(1, 1.6)
    
    # Header styling - simple black header
    for i in range(len(in_headers)):
        cell = in_table[(0, i)]
        cell.set_facecolor('#000000')
        cell.set_text_props(weight='bold', color='white', ha='center', fontsize=16)
        cell.set_height(0.12)
        cell.set_edgecolor('#000000')
        cell.set_linewidth(1.0)
    
    # Data row styling - all white, clean borders
    for i in range(1, len(in_table_data) + 1):
        for j in range(len(in_headers)):
            cell = in_table[(i, j)]
            cell.set_facecolor('white')
            cell.set_text_props(ha='center', color='black', fontsize=15)
            cell.set_edgecolor('#000000')
            cell.set_linewidth(0.8)
    
    ax1.axis('off')
    ax1.set_title('In-Sample', fontsize=14, fontweight='bold', pad=15)
    
    # Out-of-sample table
    out_table_data = format_table_data(out_sample_df)
    out_headers = ['Model', 'Regime', r'$R^2$', 'RMSE']
    
    out_table = ax2.table(
        cellText=out_table_data,
        colLabels=out_headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style out-of-sample table - LaTeX style with larger fonts
    out_table.auto_set_font_size(False)
    out_table.set_fontsize(15)
    out_table.scale(1, 1.6)
    
    # Header styling - simple black header
    for i in range(len(out_headers)):
        cell = out_table[(0, i)]
        cell.set_facecolor('#000000')
        cell.set_text_props(weight='bold', color='white', ha='center', fontsize=16)
        cell.set_height(0.12)
        cell.set_edgecolor('#000000')
        cell.set_linewidth(1.0)
    
    # Data row styling - all white, clean borders
    for i in range(1, len(out_table_data) + 1):
        for j in range(len(out_headers)):
            cell = out_table[(i, j)]
            cell.set_facecolor('white')
            cell.set_text_props(ha='center', color='black', fontsize=15)
            cell.set_edgecolor('#000000')
            cell.set_linewidth(0.8)
    
    ax2.axis('off')
    ax2.set_title('Out-of-Sample', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved table to {output_path}")
    
    return fig


def compute_performance_metrics(returns: pd.Series) -> dict:
    """
    Compute performance metrics for a strategy.
    """
    if returns.empty or len(returns) == 0:
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    # Annualized return
    n_periods = len(returns)
    total_return = (1 + returns).prod() - 1
    periods_per_year = 12  # Monthly data
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # Annualized volatility
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def compute_fixed_benchmark_metrics(results_dir: Path, save_csvs: bool = True):
    """
    Compute fixed portfolio benchmark metrics for s3_forecasting strategies.
    
    Parameters:
    -----------
    results_dir : Path
        Directory containing s3_forecasting results CSV files
    save_csvs : bool
        Whether to save benchmark returns to CSV files
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with benchmark metrics
    """
    # Load market data - results_dir is s3_forecasting/results
    # Go from tables_slides/ -> workspace root -> main_project/data/macro_processed
    script_dir = Path(__file__).parent  # tables_slides
    workspace_root = script_dir.parent  # workspace root
    data_dir = workspace_root / "main_project" / "data" / "macro_processed"
    
    # Load equity returns
    sp500 = pd.read_csv(data_dir / "sp500_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    equity_returns = (sp500["pct_change_mom"] / 100.0).resample("ME").last()
    
    # Load bond returns
    tbill = pd.read_csv(data_dir / "3m_yield_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    bond_returns = (tbill["value"] / 100.0 / 12.0).resample("ME").last()
    
    # Strategy files
    strategy_files = {
        'XGBoost': 'xgboost_returns_monthly.csv',
        'LSTM': 'lstm_returns_monthly.csv',
        'XGBoost + Groq': 'xgboost_groq_returns_monthly.csv',
        'XGBoost + OpenAI': 'xgboost_openai_returns_monthly.csv'
    }
    
    benchmark_metrics = []
    
    for strategy_name, filename in strategy_files.items():
        file_path = results_dir / filename
        
        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping")
            continue
        
        # Read weights
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.set_index('date')
        
        if 'weight' not in df.columns:
            print(f"Warning: No weight column in {filename}, skipping")
            continue
        
        # Calculate average weight
        avg_weight = df['weight'].mean()
        equity_pct = int(round(avg_weight * 100))
        bond_pct = 100 - equity_pct
        
        # Get common dates
        common_dates = df.index.intersection(equity_returns.index).intersection(bond_returns.index)
        
        if len(common_dates) == 0:
            print(f"Warning: No common dates for {strategy_name}, skipping")
            continue
        
        # Compute fixed portfolio returns
        equity_aligned = equity_returns.reindex(common_dates)
        bond_aligned = bond_returns.reindex(common_dates)
        fixed_returns = avg_weight * equity_aligned + (1 - avg_weight) * bond_aligned
        
        # Save benchmark returns to CSV
        if save_csvs:
            # Create filename: [model name] Bnchmrk ([eq_pct/tbill_pct])
            # Use underscore instead of spaces and parentheses for filesystem compatibility
            clean_name = strategy_name.replace(' + ', '_').replace(' ', '_')
            benchmark_filename = f"{clean_name}_Bnchmrk_{equity_pct}_{bond_pct}_monthly.csv"
            benchmark_path = results_dir / benchmark_filename
            
            benchmark_df = pd.DataFrame({
                'date': fixed_returns.index,
                'return': fixed_returns.values,
                'weight': avg_weight,  # Constant weight
                'equity_weight': avg_weight,
                'bond_weight': 1 - avg_weight
            })
            benchmark_df.to_csv(benchmark_path, index=False)
            # Print with readable format
            readable_name = f"{strategy_name} Bnchmrk ({equity_pct}/{bond_pct})"
            print(f"✓ Saved benchmark returns: {readable_name}")
        
        # Compute metrics
        metrics = compute_performance_metrics(fixed_returns)
        
        benchmark_metrics.append({
            'strategy': f"{strategy_name} Benchmark",
            'sharpe_ratio': metrics['sharpe_ratio'],
            'annualized_return': metrics['annualized_return'],
            'annualized_volatility': metrics['annualized_volatility'],
            'max_drawdown': metrics['max_drawdown']
        })
    
    return pd.DataFrame(benchmark_metrics)


def plot_performance_comparison_table(csv_path: str,
                                      output_path: str = None,
                                      figsize: tuple = (10, 4),
                                      include_benchmarks: bool = True):
    """
    Create a LaTeX-style black and white table for performance comparison.
    
    Parameters:
    -----------
    csv_path : str
        Path to the performance comparison CSV file
    output_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    include_benchmarks : bool
        Whether to include fixed portfolio benchmarks
    """
    df = pd.read_csv(csv_path)
    
    # Add fixed benchmarks if requested
    if include_benchmarks:
        # csv_path is in tables_slides/, need to find s3_forecasting/results
        script_dir = Path(csv_path).parent  # tables_slides
        workspace_root = script_dir.parent  # workspace root
        results_dir = workspace_root / "main_project" / "s3_forecasting" / "results"
        try:
            benchmark_df = compute_fixed_benchmark_metrics(results_dir, save_csvs=True)
            if not benchmark_df.empty:
                df = pd.concat([df, benchmark_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not compute benchmarks: {e}")
            import traceback
            traceback.print_exc()
    
    # Format data for display
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['strategy'],
            f"{row['sharpe_ratio']:.3f}",
            f"{row['annualized_return']:.3f}",
            f"{row['annualized_volatility']:.3f}",
            f"{row['max_drawdown']:.3f}"
        ])
    
    headers = ['Strategy', 'Sharpe Ratio', 'Ann. Return', 'Ann. Volatility', 'Max Drawdown']
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style table - LaTeX style with larger fonts
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 1.8)
    
    # Header styling - simple black header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#000000')
        cell.set_text_props(weight='bold', color='white', ha='center', fontsize=16)
        cell.set_height(0.12)
        cell.set_edgecolor('#000000')
        cell.set_linewidth(1.0)
    
    # Data row styling - all white, clean borders
    # Highlight benchmarks with slightly different styling
    for i in range(1, len(table_data) + 1):
        is_benchmark = 'Benchmark' in table_data[i-1][0]
        for j in range(len(headers)):
            cell = table[(i, j)]
            if is_benchmark:
                cell.set_facecolor('#F5F5F5')  # Very light gray for benchmarks
            else:
                cell.set_facecolor('white')
            cell.set_text_props(ha='center', color='black', fontsize=15)
            cell.set_edgecolor('#000000')
            cell.set_linewidth(0.8)
    
    ax.axis('off')
    ax.set_title('Performance Comparison', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved table to {output_path}")
    
    return fig


def plot_hmm_2x2_performance_table(csv_path: str,
                                   output_path: str = None,
                                   figsize: tuple = (10, 3)):
    """
    Create a LaTeX-style black and white table for HMM and 2x2 strategy performance.
    
    Parameters:
    -----------
    csv_path : str
        Path to the performance comparison CSV file
    output_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    df = pd.read_csv(csv_path)
    
    # Filter for HMM and 2x2 strategies only
    hmm_2x2_df = df[df['strategy'].str.contains('hmm|2x2', case=False, na=False)].copy()
    
    if hmm_2x2_df.empty:
        print("Warning: No HMM or 2x2 strategies found in the CSV file")
        return None
    
    # Clean up strategy names
    def clean_strategy_name(name):
        if 'hmm' in name.lower():
            return 'HMM Based'
        elif '2x2' in name.lower():
            return '2x2 Based'
        else:
            return name
    
    hmm_2x2_df['strategy'] = hmm_2x2_df['strategy'].apply(clean_strategy_name)
    
    # Format data for display
    table_data = []
    for _, row in hmm_2x2_df.iterrows():
        table_data.append([
            row['strategy'],
            f"{row['sharpe_ratio']:.3f}",
            f"{row['annualized_return']:.3f}",
            f"{row['annualized_volatility']:.3f}",
            f"{row['max_drawdown']:.3f}"
        ])
    
    headers = ['Strategy', 'Sharpe Ratio', 'Ann. Return', 'Ann. Volatility', 'Max Drawdown']
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style table - LaTeX style with larger fonts
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2.0)
    
    # Header styling - simple black header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#000000')
        cell.set_text_props(weight='bold', color='white', ha='center', fontsize=16)
        cell.set_height(0.12)
        cell.set_edgecolor('#000000')
        cell.set_linewidth(1.0)
    
    # Data row styling - all white, clean borders
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor('white')
            cell.set_text_props(ha='center', color='black', fontsize=15)
            cell.set_edgecolor('#000000')
            cell.set_linewidth(0.8)
    
    ax.axis('off')
    ax.set_title('HMM and 2x2 Strategy Performance', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved table to {output_path}")
    
    return fig


def update_performance_csvs_with_benchmarks():
    """Update performance CSV files to include benchmark metrics."""
    base_dir = Path(__file__).parent
    workspace_root = base_dir.parent
    results_dir = workspace_root / "main_project" / "s3_forecasting" / "results"
    
    # Compute benchmark metrics
    benchmark_df = compute_fixed_benchmark_metrics(results_dir, save_csvs=True)
    
    if benchmark_df.empty:
        print("Warning: No benchmark metrics computed")
        return
    
    # Update monthly CSV
    monthly_csv = base_dir / 'performance_comparison_all_strategies_monthly.csv'
    if monthly_csv.exists():
        monthly_df = pd.read_csv(monthly_csv)
        # Add benchmarks if not already present
        existing_strategies = set(monthly_df['strategy'].values)
        new_benchmarks = benchmark_df[~benchmark_df['strategy'].isin(existing_strategies)]
        if not new_benchmarks.empty:
            updated_monthly = pd.concat([monthly_df, new_benchmarks], ignore_index=True)
            updated_monthly.to_csv(monthly_csv, index=False)
            print(f"✓ Updated {monthly_csv.name} with benchmarks")
    
    # Update quarterly CSV (if exists)
    quarterly_csv = base_dir / 'performance_comparison_all_strategies.csv'
    if quarterly_csv.exists():
        quarterly_df = pd.read_csv(quarterly_csv)
        # Add benchmarks if not already present
        existing_strategies = set(quarterly_df['strategy'].values)
        new_benchmarks = benchmark_df[~benchmark_df['strategy'].isin(existing_strategies)]
        if not new_benchmarks.empty:
            updated_quarterly = pd.concat([quarterly_df, new_benchmarks], ignore_index=True)
            updated_quarterly.to_csv(quarterly_csv, index=False)
            print(f"✓ Updated {quarterly_csv.name} with benchmarks")


def main():
    """Main function to create all tables."""
    base_dir = Path(__file__).parent
    
    # Update CSV files with benchmarks first
    print("Updating CSV files with benchmark metrics...")
    update_performance_csvs_with_benchmarks()
    
    # Create in/out-of-sample table
    print("\nCreating in/out-of-sample table...")
    in_sample_df, out_sample_df = parse_in_out_sample_file(base_dir / 'in_out_sample.txt')
    plot_in_out_sample_table(
        in_sample_df,
        out_sample_df,
        output_path=base_dir / 'in_out_sample_table.png'
    )
    
    # Create performance comparison table (s3 forecasting)
    print("\nCreating performance comparison table (s3 forecasting)...")
    plot_performance_comparison_table(
        base_dir / 'performance_comparison_all_strategies_monthly.csv',
        output_path=base_dir / 'performance_comparison_table.png',
        include_benchmarks=True
    )
    
    # Create HMM and 2x2 performance table
    print("\nCreating HMM and 2x2 performance table...")
    hmm_2x2_csv = Path(__file__).parent.parent / 'main_project' / 's2_regimeness' / 'trading_strategy' / 'results' / 'performance_comparison_all_strategies.csv'
    if hmm_2x2_csv.exists():
        plot_hmm_2x2_performance_table(
            hmm_2x2_csv,
            output_path=base_dir / 'hmm_2x2_performance_table.png'
        )
    else:
        print(f"Warning: Could not find HMM/2x2 performance CSV at {hmm_2x2_csv}")
    
    print("\n✓ All tables created successfully!")


if __name__ == "__main__":
    main()

