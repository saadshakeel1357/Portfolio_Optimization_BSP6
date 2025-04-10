import yfinance as yf         # For downloading financial data from Yahoo Finance.
import numpy as np             # For numerical operations.
import pandas as pd            # For data manipulation.
from datetime import datetime  # For date manipulation.

def download_to_csv(ticker: str) -> pd.DataFrame:
    """
    Downloads the maximum available historical data for a given ticker from Yahoo Finance,
    selects specific columns, formats the index as dates, and saves the data as a tab-separated CSV file.
    
    Parameters:
        ticker (str): The ticker symbol for the asset.
    
    Returns:
        pd.DataFrame: Processed data containing 'close', 'open', 'high', and 'low'.
    """
    # Specify auto_adjust explicitly to remove the warning.
    data = yf.download(ticker, period="max", auto_adjust=False)
    
    # Select only the required columns.
    data = data[['Close', 'Open', 'High', 'Low']]
    
    # Format the index as dates in the 'YYYY-MM-DD' format.
    data.index = data.index.strftime('%Y-%m-%d')
    data.index.names = ['time']
    
    # Rename the columns to lowercase.
    data = data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'})
    
    # Save the processed data to a tab-separated CSV file.
    data.to_csv(f"{ticker}.csv", sep='\t')
    return data

def analyze_indices_returns():
    """
    Downloads data for three indices (^GSPC, LQD, IEF), computes their daily simple returns,
    saves the returns into CSV files, and computes the average daily return and volatility (standard deviation)
    for each index.
    
    Changes in this version to fix warnings:
      - Added the parameter auto_adjust=False to yf.download() to explicitly set the adjustment behavior.
      - Used the .item() method to extract a scalar value from the Series returned by .mean() and .std(),
        which prevents the FutureWarning when using float() directly.
    """
    indices = ["^GSPC", "LQD", "IEF"]
    print("Downloading data for indices and computing returns, average returns, and volatility...")
    
    for ticker in indices:
        # Download historical price data and save it to a CSV file.
        data = download_to_csv(ticker)
        
        # Compute daily simple returns from the closing prices.
        # Drop the first value (NaN) because there's no previous day to compare.
        returns = data['close'].pct_change().dropna()
        
        # Create a "safe" ticker name for the filename by removing special characters like '^'.
        safe_ticker = ticker.replace("^", "")
        
        # Save the computed returns to a CSV file.
        returns.to_csv(f"{safe_ticker}_returns.csv", sep='\t', header=True)
        
        # Compute average (mean) return and volatility (standard deviation).
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Convert the one-element Series to a scalar using .item() to avoid a FutureWarning.
        avg_return_float = avg_return.item()
        volatility_float = volatility.item()
        
        # Print the computed statistics for the current index.
        print(f"\n--- Statistics for {ticker} ---")
        print(f"Average Daily Return: {avg_return_float:.6f}")
        print(f"Volatility (Standard Deviation): {volatility_float:.6f}")

def main():
    """
    Main function to trigger the analysis of indices returns.
    
    This function downloads the data for each specified index, computes daily returns,
    saves the returns into CSV files, and prints the average daily return and volatility.
    """
    print("Analyzing the returns of the indices: ^GSPC, LQD, IEF")
    analyze_indices_returns()

if __name__ == '__main__':
    main()
