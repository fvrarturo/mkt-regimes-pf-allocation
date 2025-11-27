"""
Compute Equity Risk Premium (ERP) from S&P 500 and 3M Treasury yield data.

ERP = Stock return - Risk-free return (bond return)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compute_equity_risk_premium(
    sp500_path: Path,
    yield_3m_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Compute monthly Equity Risk Premium (ERP).
    
    Parameters:
    -----------
    sp500_path : Path
        Path to S&P 500 processed CSV file
    yield_3m_path : Path
        Path to 3M Treasury yield processed CSV file
    output_path : Path
        Path to save the output CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with date, ERP, and related columns
    """
    
    # Load S&P 500 data (monthly)
    print("Loading S&P 500 data...")
    sp500 = pd.read_csv(sp500_path, parse_dates=["date"])
    sp500 = sp500.set_index("date").sort_index()
    
    # Extract monthly return (already in percentage)
    sp500["stock_return"] = sp500["pct_change_mom"] / 100.0  # Convert to decimal
    sp500_monthly = sp500.resample("ME").last()[["stock_return", "value"]]
    sp500_monthly.rename(columns={"value": "sp500_value"}, inplace=True)
    
    # Load 3M Treasury yield data (daily)
    print("Loading 3M Treasury yield data...")
    yield_3m = pd.read_csv(yield_3m_path, parse_dates=["date"])
    yield_3m = yield_3m.set_index("date").sort_index()
    
    # Convert daily annualized yield to daily simple return
    # Annual yield / 100 / 252 trading days = daily return
    yield_3m["r_daily"] = (yield_3m["value"] / 100.0) / 252.0
    
    # Compound daily returns within each month to get monthly risk-free return
    yield_monthly = yield_3m["r_daily"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
    yield_monthly = yield_monthly.to_frame(name="risk_free_return")
    
    # Get last yield value of each month for reference
    yield_3m_monthly_value = yield_3m["value"].resample("ME").last().to_frame(name="yield_3m_value")
    
    # Merge stock and bond returns
    print("Merging data and computing ERP...")
    df = sp500_monthly.join(yield_monthly, how="inner")
    df = df.join(yield_3m_monthly_value, how="inner")
    
    # Compute Equity Risk Premium (excess return)
    df["ERP"] = df["stock_return"] - df["risk_free_return"]
    
    # Reset index to have date as column
    df = df.reset_index()
    
    # Select and reorder columns
    df_output = df[["date", "sp500_value", "yield_3m_value", "stock_return", 
                    "risk_free_return", "ERP"]].copy()
    
    # Save to CSV
    print(f"Saving ERP data to {output_path}...")
    df_output.to_csv(output_path, index=False)
    
    print(f"\nERP computation complete!")
    print(f"Date range: {df_output['date'].min()} to {df_output['date'].max()}")
    print(f"Total observations: {len(df_output)}")
    print(f"\nERP Statistics:")
    print(f"  Mean: {df_output['ERP'].mean():.4f} ({df_output['ERP'].mean()*100:.2f}%)")
    print(f"  Std:  {df_output['ERP'].std():.4f} ({df_output['ERP'].std()*100:.2f}%)")
    print(f"  Min:  {df_output['ERP'].min():.4f} ({df_output['ERP'].min()*100:.2f}%)")
    print(f"  Max:  {df_output['ERP'].max():.4f} ({df_output['ERP'].max()*100:.2f}%)")
    
    return df_output


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent
    sp500_path = base_dir / "sp500_processed.csv"
    yield_3m_path = base_dir / "3m_yield_processed.csv"
    output_path = base_dir / "equity_risk_pr.csv"
    
    # Compute ERP
    erp_df = compute_equity_risk_premium(
        sp500_path=sp500_path,
        yield_3m_path=yield_3m_path,
        output_path=output_path
    )

