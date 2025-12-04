"""
Data loading utilities for ERP forecasting and trading.
Uses the same macro variables as conditional regressions.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def _get_base_dir(base_dir: Optional[Path] = None) -> Path:
    """Get base directory (main_project)."""
    if base_dir is not None:
        base_dir = Path(base_dir)
        if base_dir.name == "s3_forecasting":
            return base_dir.parent
        return base_dir
    # Default: go up from s3_forecasting -> main_project
    return Path(__file__).parent.parent


def load_erp_data(base_dir: Optional[Path] = None) -> pd.Series:
    """
    Load ERP (Equity Risk Premium) data.
    
    Returns:
    --------
    pd.Series
        ERP values indexed by date
    """
    base_dir = _get_base_dir(base_dir)
    erp_path = base_dir / 'data' / 'macro_processed' / 'equity_risk_pr.csv'
    
    if not erp_path.exists():
        raise FileNotFoundError(f"ERP file not found: {erp_path}")
    
    erp_df = pd.read_csv(erp_path, parse_dates=['date'])
    erp_df['date'] = pd.to_datetime(erp_df['date'])
    
    # Use ERP column
    if 'ERP' in erp_df.columns:
        erp_series = erp_df.set_index('date')['ERP']
    elif 'erp' in erp_df.columns:
        erp_series = erp_df.set_index('date')['erp']
    else:
        raise ValueError("ERP column not found")
    
    # Convert to monthly (end of month)
    erp_series = erp_series.resample('ME').last()
    erp_series = erp_series.sort_index()
    
    return erp_series


def load_market_data(base_dir: Optional[Path] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Load equity and bond returns for trading.
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        (equity_returns, bond_returns) both indexed by date
    """
    base_dir = _get_base_dir(base_dir)
    data_dir = base_dir / "data" / "macro_processed"
    
    # Load equity returns
    sp500 = pd.read_csv(data_dir / "sp500_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    equity_returns = (sp500["pct_change_mom"] / 100.0).resample("ME").last()
    
    # Load bond returns
    tbill = pd.read_csv(data_dir / "3m_yield_processed.csv", parse_dates=["date"]).set_index("date").sort_index()
    bond_returns = (tbill["value"] / 100.0 / 12.0).resample("ME").last()
    
    # Align
    aligned = pd.DataFrame({"equity_return": equity_returns, "bond_return": bond_returns}).dropna()
    equity_returns = aligned["equity_return"]
    bond_returns = aligned["bond_return"]
    
    return equity_returns, bond_returns


def load_macro_features(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load macro features for ERP forecasting.
    Uses the same variables as conditional regressions.
    
    Returns:
    --------
    pd.DataFrame
        Macro features indexed by date
    """
    base_dir = _get_base_dir(base_dir)
    macro_data_dir = base_dir / 'data' / 'macro_processed_full'
    
    # Same macro variables as used in conditional regressions
    macro_dirs = {
        'ec_growth': [
            'industrial_production_processed.csv',
            'retail_sales_processed.csv',
            'tot_business_inventories_processed.csv',
            'export_price_index_processed.csv',
            'import_price_index_processed.csv',
            'unemployment_processed.csv'
        ],
        'inflation': [
            'cpi_processed.csv',
            'PCE_price_index_processed.csv',
            'PPI_inflation_processed.csv'
        ],
        'mkt_vol': [
            'nat_fin_condition_indx_processed_monthly.csv',
            '10y_2y_spread_processed_monthly.csv'
        ],
        'mon_policy': [
            '10y_treasury_const_maturity_rate_processed.csv',
            'fed_reserve_discount_rate_processed.csv',
            'fedfunds_processed.csv',
            'm2_real_money_supply_processed.csv'
        ]
    }
    
    all_data = []
    
    for category, files in macro_dirs.items():
        category_dir = macro_data_dir / category
        
        for filename in files:
            file_path = category_dir / filename
            
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                
                # Use 'value' column if available, otherwise use first numeric column
                if 'value' in df.columns:
                    value_col = 'value'
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        continue
                    value_col = numeric_cols[0]
                
                # Create variable name from filename
                var_name = filename.replace('_processed 2.csv', '').replace('_processed_monthly.csv', '').replace('_processed.csv', '')
                var_name = var_name.replace(' ', '_')
                
                # Select date and value
                df_subset = df[['date', value_col]].copy()
                df_subset.columns = ['date', var_name]
                
                # Convert to monthly if not already
                df_subset['date'] = pd.to_datetime(df_subset['date'])
                df_subset = df_subset.set_index('date').sort_index()
                df_subset = df_subset.resample('ME').last()
                df_subset = df_subset.reset_index()
                
                all_data.append(df_subset)
                print(f"  ✓ Loaded {var_name}: {len(df_subset)} observations")
                
            except Exception as e:
                print(f"Warning: Error loading {filename}: {e}")
                continue
    
    # Merge all macro variables
    print("\nMerging all macro variables...")
    if not all_data:
        raise ValueError("No macro variables loaded!")
    
    merged = all_data[0]
    for df in all_data[1:]:
        merged = pd.merge(merged, df, on='date', how='outer', suffixes=('', '_drop'))
        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.str.endswith('_drop')]
    
    merged = merged.sort_values('date').reset_index(drop=True)
    merged = merged.dropna(subset=['date'])
    
    print(f"  ✓ Merged dataset: {len(merged)} observations")
    print(f"  ✓ Variables: {len(merged.columns) - 1} macro variables")
    print(f"  ✓ Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.set_index('date').sort_index()
    
    # Already monthly from resample('ME').last() above
    return merged


def load_sentiment_groq(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Groq sentiment data.
    
    Returns:
    --------
    pd.DataFrame
        Sentiment scores indexed by date
    """
    base_dir = _get_base_dir(base_dir)
    sentiment_path = base_dir / 's3_forecasting' / 'news_data' / 'sentiment_groq.csv'
    
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Groq sentiment file not found: {sentiment_path}")
    
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=['date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df = sentiment_df.set_index('date').sort_index()
    
    # Convert to monthly (end of month) - take last value of month
    sentiment_df = sentiment_df.resample('ME').last()
    
    # Return only numeric columns (sentiment scores)
    return sentiment_df.select_dtypes(include=[np.number])


def load_sentiment_openai(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load OpenAI sentiment data.
    
    Returns:
    --------
    pd.DataFrame
        Sentiment scores indexed by date
    """
    base_dir = _get_base_dir(base_dir)
    sentiment_path = base_dir / 's3_forecasting' / 'news_data' / 'sentiment_openai.csv'
    
    if not sentiment_path.exists():
        raise FileNotFoundError(f"OpenAI sentiment file not found: {sentiment_path}")
    
    sentiment_df = pd.read_csv(sentiment_path)
    
    # Handle different column names
    if 'month' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['month'])
    elif 'date' in sentiment_df.columns:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    else:
        raise ValueError("Could not find date column in OpenAI sentiment file")
    
    sentiment_df = sentiment_df.set_index('date').sort_index()
    
    # Already monthly, but ensure month-end
    sentiment_df = sentiment_df.resample('ME').last()
    
    # Return sentiment_score column if it exists, otherwise return all numeric columns
    if 'sentiment_score' in sentiment_df.columns:
        return sentiment_df[['sentiment_score']]
    else:
        # Return all numeric columns
        return sentiment_df.select_dtypes(include=[np.number])

