"""
Data Download Module
Downloads market data from Yahoo Finance (VIX, Treasury yields, Dollar index futures).
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional


# ============================
# Parameters
# ============================
START_DATE = "1980-01-01"
END_DATE = None  # None -> latest

# Yahoo tickers
TICKERS = {
    "VIX": "^VIX",         # CBOE Volatility Index
    "UST_10Y": "^TNX",     # 10-year Treasury yield index (x10)
    "DX_FUT": "DX=F",      # Dollar index futures (front contract)
}


def download_series_yahoo(
    ticker: str,
    name: str,
    start: str = START_DATE,
    end: Optional[str] = END_DATE
) -> pd.Series:
    """
    Robustly download Adjusted Close price as a Series from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Yahoo Finance ticker symbol
    name : str
        Name for the resulting Series
    start : str
        Start date (YYYY-MM-DD format)
    end : str, optional
        End date (YYYY-MM-DD format), None for latest
    
    Returns:
    --------
    pd.Series
        Adjusted Close prices with specified name
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # df["Adj Close"] can be a Series or a DataFrame (if MultiIndex columns)
    adj = df["Adj Close"]

    # If it's a DataFrame (MultiIndex columns), take the first column
    if isinstance(adj, pd.DataFrame):
        adj = adj.iloc[:, 0]

    # Make sure it's a Series with a nice name
    adj = adj.astype("float64")
    adj.name = name
    return adj


def download_yahoo_data(
    start_date: str = START_DATE,
    end_date: Optional[str] = END_DATE
) -> pd.DataFrame:
    """
    Download all Yahoo Finance data and combine into a DataFrame.
    
    Parameters:
    -----------
    start_date : str
        Start date for data download
    end_date : str, optional
        End date for data download
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with VIX, UST10Y, and DX_FUT
    """
    # Download VIX
    vix = download_series_yahoo(TICKERS["VIX"], "VIX", start=start_date, end=end_date)

    # Download Treasury Yields
    y10_raw = download_series_yahoo(TICKERS["UST_10Y"], "UST10Y_raw", start=start_date, end=end_date)

    # Yahoo yields quoted as "yield * 10" (e.g. 46.50 = 4.65%)
    y10 = (y10_raw / 10.0).rename("UST10Y_pct")

    # Download Dollar index futures
    dx_fut = download_series_yahoo(TICKERS["DX_FUT"], "DX_FUT", start=start_date, end=end_date)

    # Combine everything
    macro = pd.concat([vix, y10, dx_fut], axis=1).sort_index()

    # Filter to start date and drop rows with all NaN
    macro = macro.loc[macro.index >= pd.to_datetime(start_date)]
    macro = macro.dropna(how="all")

    return macro


def save_yahoo_data(
    output_path: Path,
    start_date: str = START_DATE,
    end_date: Optional[str] = END_DATE
) -> pd.DataFrame:
    """
    Download and save Yahoo Finance data to CSV.
    
    Parameters:
    -----------
    output_path : Path
        Path to save the CSV file
    start_date : str
        Start date for data download
    end_date : str, optional
        End date for data download
    
    Returns:
    --------
    pd.DataFrame
        Downloaded and processed DataFrame
    """
    macro = download_yahoo_data(start_date, end_date)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(output_path)
    
    print(f"Saved Yahoo Finance data to {output_path}")
    print(f"\nData shape: {macro.shape}")
    print(f"Date range: {macro.index.min()} to {macro.index.max()}")
    print(f"\nLast few rows:")
    print(macro.tail())
    
    return macro


if __name__ == "__main__":
    # Example usage
    output_file = Path(__file__).parent / "macro_yahoo_vix_yieldcurve_dxy.csv"
    save_yahoo_data(output_file)

