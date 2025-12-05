"""
Main script for downloading Yahoo Finance data.
Equivalent to get data.ipynb
"""

from pathlib import Path
from data_download import save_yahoo_data

if __name__ == "__main__":
    # Set output path
    output_file = Path(__file__).parent / "macro_yahoo_vix_yieldcurve_dxy.csv"
    
    # Download and save data
    save_yahoo_data(output_file)

