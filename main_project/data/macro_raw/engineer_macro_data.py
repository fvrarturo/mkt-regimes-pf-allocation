"""
Macro Data Engineering Script

This script performs comprehensive data engineering on macroeconomic data following
academic and research best practices. It includes:

1. Percentage Changes (MoM, YoY) for level variables
2. Log Transformations for variables with exponential growth
3. First Differences for rates and spreads
4. Normalization/Standardization (Z-score and Min-Max)
5. Missing data handling

Based on academic best practices:
- Stock & Watson (2011): "Introduction to Econometrics"
- Hamilton (1994): "Time Series Analysis"
- Diebold (2017): "Forecasting: Principles and Practice"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MacroDataEngineer:
    """
    Comprehensive macro data engineering class following academic best practices.
    """
    
    def __init__(self, data_dir: Path, output_dir: Optional[Path] = None):
        """
        Initialize the data engineer.
        
        Parameters:
        -----------
        data_dir : Path
            Directory containing raw macro data
        output_dir : Path, optional
            Directory to save processed data. If None, creates 'processed' subdirectory.
        """
        self.data_dir = Path(data_dir)
        if output_dir is None:
            self.output_dir = self.data_dir.parent / 'macro_processed'
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define variable categories for appropriate transformations
        self.level_variables = {
            'ec_growth': ['gdp', 'real_gdp', 'industrial_production', 'retail_sales', 
                         'tot_business_inventories', 'export_price_index', 'import_price_index'],
            'inflation': ['cpi', 'PCE_price_index', 'PPI_inflation'],
            'mon_policy': ['m2_real_money_supply'],
            'other': ['sp500']
        }
        
        self.rate_variables = {
            'ec_growth': ['unemployment'],
            'mon_policy': ['fedfunds', 'fed_reserve_discount_rate', 
                          '10y_treasury_const_maturity_rate'],
            'other': ['10y_yield', '2y_yield', '3m_yield']
        }
        
        self.spread_variables = {
            'mkt_vol': ['10y_2y_spread'],
            'other': ['bofa_highyield_spread']
        }
        
        self.volatility_variables = {
            'mkt_vol': ['vix', '3month_vol_index_sp500', 'nasdaq_vol_indx']
        }
        
        self.index_variables = {
            'mkt_vol': ['nat_fin_condition_indx']
        }
        
    def load_macro_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a macro data CSV file and standardize format.
        
        Parameters:
        -----------
        file_path : Path
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame or None
            Standardized dataframe with 'date' and 'value' columns
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle date column (case-insensitive)
            date_col = None
            for col in df.columns:
                if col.lower() in ['observation_date', 'date']:
                    date_col = col
                    break
            
            if date_col:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            else:
                print(f"  Warning: No date column found in {file_path.name}")
                return None
            
            # Get value column (first non-date column, case-insensitive)
            exclude_cols = ['date', 'observation_date', 'Date']
            value_cols = [c for c in df.columns if c not in exclude_cols and c.lower() not in ['date', 'observation_date']]
            if not value_cols:
                print(f"  Warning: No value column found in {file_path.name}")
                return None
            
            value_col = value_cols[0]
            
            # Create standardized dataframe
            df_clean = df[['date', value_col]].copy()
            df_clean = df_clean.rename(columns={value_col: 'value'})
            
            # Convert value to numeric, handling empty strings
            df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
            
            # Remove rows with missing dates or values
            df_clean = df_clean.dropna(subset=['date', 'value'])
            
            # Sort by date
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
            
            # Set date as index for time series operations
            df_clean = df_clean.set_index('date')
            
            return df_clean
            
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            return None
    
    def calculate_pct_change(self, df: pd.DataFrame, periods: List[int] = [1, 12]) -> pd.DataFrame:
        """
        Calculate percentage changes (MoM, YoY, etc.).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'value' column
        periods : List[int]
            Periods for percentage change (1=MoM, 12=YoY for monthly data)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional pct_change columns
        """
        df = df.copy()
        
        for period in periods:
            if period == 1:
                col_name = 'pct_change_mom'
            elif period == 12:
                col_name = 'pct_change_yoy'
            elif period == 3:
                col_name = 'pct_change_qoq'
            else:
                col_name = f'pct_change_{period}'
            
            df[col_name] = df['value'].pct_change(periods=period) * 100
        
        return df
    
    def calculate_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log transformation (log levels and log differences).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'value' column (must be positive)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with log transformations
        """
        df = df.copy()
        
        # Only apply if all values are positive
        if (df['value'] > 0).all():
            df['log_value'] = np.log(df['value'])
            df['log_diff'] = df['log_value'].diff()
            df['log_pct_change'] = df['log_value'].pct_change() * 100
        else:
            print(f"    Warning: Cannot apply log transform (non-positive values present)")
        
        return df
    
    def calculate_first_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate first differences (for rates and spreads).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'value' column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with first difference column
        """
        df = df.copy()
        df['first_diff'] = df['value'].diff()
        return df
    
    def normalize_zscore(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply Z-score normalization (standardization).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns to normalize
        columns : List[str]
            Column names to normalize
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized columns (prefix: 'zscore_')
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    df[f'zscore_{col}'] = (df[col] - mean_val) / std_val
                else:
                    print(f"    Warning: Cannot normalize {col} (std=0)")
        
        return df
    
    def normalize_minmax(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply Min-Max normalization (scaling to [0, 1]).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns to normalize
        columns : List[str]
            Column names to normalize
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized columns (prefix: 'minmax_')
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val
                
                if range_val > 0:
                    df[f'minmax_{col}'] = (df[col] - min_val) / range_val
                else:
                    print(f"    Warning: Cannot normalize {col} (range=0)")
        
        return df
    
    def get_variable_category(self, var_name: str, subdir: str) -> str:
        """
        Determine the category of a variable for appropriate transformation.
        
        Parameters:
        -----------
        var_name : str
            Variable name (filename without extension)
        subdir : str
            Subdirectory name
            
        Returns:
        --------
        str
            Category: 'level', 'rate', 'spread', 'volatility', 'index'
        """
        var_name_lower = var_name.lower()
        
        # Check all categories
        for category, vars_dict in [
            ('level', self.level_variables),
            ('rate', self.rate_variables),
            ('spread', self.spread_variables),
            ('volatility', self.volatility_variables),
            ('index', self.index_variables)
        ]:
            for subdir_key, var_list in vars_dict.items():
                if subdir == subdir_key and var_name_lower in [v.lower() for v in var_list]:
                    return category
        
        # Default: treat as level variable
        return 'level'
    
    def process_variable(self, file_path: Path, subdir: str) -> Optional[pd.DataFrame]:
        """
        Process a single macro variable with appropriate transformations.
        
        Parameters:
        -----------
        file_path : Path
            Path to CSV file
        subdir : str
            Subdirectory name (category)
            
        Returns:
        --------
        pd.DataFrame or None
            Processed dataframe with all transformations
        """
        var_name = file_path.stem
        print(f"\n  Processing: {var_name}")
        
        # Load data
        df = self.load_macro_file(file_path)
        if df is None or len(df) == 0:
            return None
        
        print(f"    Loaded {len(df)} observations ({df.index.min()} to {df.index.max()})")
        
        # Determine variable category
        category = self.get_variable_category(var_name, subdir)
        print(f"    Category: {category}")
        
        # Apply transformations based on category
        if category == 'level':
            # Level variables: percentage changes and log transformation
            df = self.calculate_pct_change(df, periods=[1, 12])  # MoM and YoY
            df = self.calculate_log_transform(df)
            
            # Normalize key columns
            norm_cols = ['value', 'pct_change_mom', 'pct_change_yoy']
            if 'log_value' in df.columns:
                norm_cols.append('log_value')
            df = self.normalize_zscore(df, norm_cols)
            df = self.normalize_minmax(df, norm_cols)
            
        elif category == 'rate':
            # Rates: first differences and percentage changes
            df = self.calculate_first_difference(df)
            df = self.calculate_pct_change(df, periods=[1, 12])
            
            # Normalize
            norm_cols = ['value', 'first_diff', 'pct_change_mom']
            df = self.normalize_zscore(df, norm_cols)
            df = self.normalize_minmax(df, norm_cols)
            
        elif category == 'spread':
            # Spreads: first differences
            df = self.calculate_first_difference(df)
            df = self.calculate_pct_change(df, periods=[1])
            
            # Normalize
            norm_cols = ['value', 'first_diff', 'pct_change_mom']
            df = self.normalize_zscore(df, norm_cols)
            df = self.normalize_minmax(df, norm_cols)
            
        elif category == 'volatility':
            # Volatility: log transformation (volatility is log-normally distributed)
            df = self.calculate_log_transform(df)
            df = self.calculate_pct_change(df, periods=[1])
            
            # Normalize
            norm_cols = ['value']
            if 'log_value' in df.columns:
                norm_cols.append('log_value')
            norm_cols.append('pct_change_mom')
            df = self.normalize_zscore(df, norm_cols)
            df = self.normalize_minmax(df, norm_cols)
            
        elif category == 'index':
            # Index variables: percentage changes
            df = self.calculate_pct_change(df, periods=[1, 12])
            
            # Normalize
            norm_cols = ['value', 'pct_change_mom', 'pct_change_yoy']
            df = self.normalize_zscore(df, norm_cols)
            df = self.normalize_minmax(df, norm_cols)
        
        # Reset index to have date as column
        df = df.reset_index()
        
        return df
    
    def process_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process all macro data files.
        
        Returns:
        --------
        Dict[str, Dict[str, pd.DataFrame]]
            Nested dictionary: {subdir: {var_name: processed_df}}
        """
        print("="*80)
        print("MACRO DATA ENGINEERING")
        print("="*80)
        
        processed_data = {}
        
        # Iterate through subdirectories
        for subdir in ['ec_growth', 'inflation', 'mkt_vol', 'mon_policy', 'other']:
            subdir_path = self.data_dir / subdir
            
            if not subdir_path.exists():
                print(f"\n  Subdirectory not found: {subdir}")
                continue
            
            print(f"\n{'='*80}")
            print(f"Processing: {subdir.upper()}")
            print(f"{'='*80}")
            
            processed_data[subdir] = {}
            
            # Get all CSV files
            csv_files = list(subdir_path.glob('*.csv'))
            
            for file_path in sorted(csv_files):
                if file_path.name == '.DS_Store':
                    continue
                
                df_processed = self.process_variable(file_path, subdir)
                
                if df_processed is not None and len(df_processed) > 0:
                    var_name = file_path.stem
                    processed_data[subdir][var_name] = df_processed
                    
                    # Save processed data
                    output_subdir = self.output_dir / subdir
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    output_file = output_subdir / f"{var_name}_processed.csv"
                    df_processed.to_csv(output_file, index=False)
                    print(f"    Saved: {output_file}")
        
        return processed_data
    
    def create_summary_report(self, processed_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Create a summary report of processed data.
        
        Parameters:
        -----------
        processed_data : Dict[str, Dict[str, pd.DataFrame]]
            Processed data dictionary
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MACRO DATA ENGINEERING SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        total_vars = 0
        for subdir, vars_dict in processed_data.items():
            report_lines.append(f"{subdir.upper()}:")
            report_lines.append("-" * 40)
            
            for var_name, df in vars_dict.items():
                total_vars += 1
                report_lines.append(f"  {var_name}:")
                report_lines.append(f"    Observations: {len(df)}")
                report_lines.append(f"    Date range: {df['date'].min()} to {df['date'].max()}")
                report_lines.append(f"    Columns: {', '.join(df.columns)}")
                report_lines.append("")
        
        report_lines.append(f"Total variables processed: {total_vars}")
        report_lines.append("")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / "processing_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\nSummary report saved to: {report_file}")


def main():
    """Main execution function."""
    # Set up paths
    base_dir = Path(__file__).parent
    data_dir = base_dir  # Macro data is in the same directory as this script
    
    # Initialize engineer
    engineer = MacroDataEngineer(data_dir)
    
    # Process all data
    processed_data = engineer.process_all_data()
    
    # Create summary report
    engineer.create_summary_report(processed_data)
    
    print("\n" + "="*80)
    print("DATA ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nProcessed data saved to: {engineer.output_dir}")


if __name__ == '__main__':
    main()

