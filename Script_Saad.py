import yfinance as yf         # For downloading financial data from Yahoo Finance.
import numpy as np             # For numerical operations.
import pandas as pd            # For data manipulation.
from datetime import datetime  # For date manipulation.


"""
This script downloads historical data for three financial indices from Yahoo Finance,
computes their daily simple returns, saves the returns into CSV files, and computes the average daily return
and volatility (standard deviation) for each index.
It also allows the user to generate a CSV file with parameters suitable for a Geometric Brownian Motion (GBM) model.


This script is not yet optimized. Ideally, I would have two separate scripts:
1. One for downloading the data and computing returns.
2. Another for the GBM parameters CSV generation.

But for now the code works fine.
"""


##### DATASETS AND DATES #####
# (^GSPC, LQD, IEF) is dataset 1. S&P 500 (^GSPC), iShares iBoxx $ Inv Grade Corporate Bond ETF (LQD), and iShares 7-10 Year Treasury Bond ETF (IEF).
### start_date="2002-07-30", end_date="2025-05-22"

# (^STOXX50E, EUCO.L, VGEA.DE) is dataset 2. EURO STOXX 50 I (^STOXX50E), SPDR Bloomberg Euro Corporate Bond UCITS ETF (EUCO.L), Vanguard EUR Eurozone Government Bond UCITS ETF EUR Accumulation (VGEA.DE)
### start_date="2019-02-19", end_date="2025-04-22"

# (URTH, IBND, 1677.T) is dataset 3. iShares MSCI World ETF (URTH), SPDR Bloomberg International Corporate Bond ETF (IBND), Listed Index Fund International Bond (FTSE WGBI) Monthly Dividend Payment (1677.T)
### start_date="2012-01-12", end_date="2024-09-04"

# (DAX, EXHF.DE, VECP.L) is dataset 4. DAX (^DAX), iShares Euro Government Bond Capped 1.5-10.5yr UCITS ETF (DE) (EXHF.DE), Vanguard EUR Corporate Bond UCITS ETF (VECP.L)
### start_date="2016-02-24", end_date="2025-02-26"

# (^HSI, 3141.HK, IGEA.L) is dataset 5. Hang Seng Index (^HSI), iShares Asia High Yield Bond ETF (3141.HK), iShares Emerging Asia Local Govt Bond UCITS ETF USD (Dist) (IGEA.L)
### start_date="2014-11-07", end_date="2023-11-08"




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
    
    Returns:
        dict: A dictionary where each key is a ticker and the value is another dictionary with:
              'initial_value', 'drift', and 'volatility'
    """
    indices = ["2819.HK", "2829.HK", "3122.HK"]
    results = {}
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
        avg_return = returns.mean().item()   # Convert to a scalar
        volatility = returns.std().item()      # Convert to a scalar
        
        # Extract initial value from the price data and convert it to a float.
        initial_value = float(data['close'].iloc[0])
        
        # Store the extracted values in our results dictionary.
        results[ticker] = {
            "initial_value": initial_value,
            "drift": avg_return,
            "volatility": volatility
        }
        
        # Print the computed statistics for the current index.
        print(f"\n--- Statistics for {ticker} ---")
        print(f"Initial Value: {initial_value}")
        print(f"Average Daily Return (Drift): {avg_return:.6f}")
        print(f"Volatility (Standard Deviation): {volatility:.6f}")
    
    return results

def create_gbm_parameter_csv(index_ticker, initial_value, drift, volatility, 
                               time_horizon=1, steps=252, output_file="gbm_parameters.csv"):
    """
    Creates a CSV file containing parameters suitable for the GBM model input.
    
    The CSV file has three columns: 'type', 'parameter', and 'value'. For this example,
    the 'type' field is fixed as 'Stock'.
    
    Parameters:
        index_ticker (str): The ticker used (for clarity, though not stored in CSV).
        initial_value (float): The starting price for the GBM simulation.
        drift (float): The average daily return to be used as the drift.
        volatility (float): The standard deviation of returns.
        time_horizon (float): The simulation time horizon in years (default is 1).
        steps (int): The number of simulation steps (default is 252).
        output_file (str): The name of the output CSV file.
    """
    df = pd.DataFrame({
        "type": ["Stock"] * 5,
        "parameter": ["initial_value", "drift", "volatility", "time_horizon", "steps"],
        "value": [initial_value, drift, volatility, time_horizon, steps]
    })
    df.to_csv(output_file, index=False)
    print(f"GBM parameter CSV saved to {output_file}.")

def main():
    """
    Main function to trigger the indices returns analysis and then convert the results
    into a GBM parameter CSV file.
    
    Workflow:
      1. Analyze indices returns for ^GSPC, LQD, and IEF.
      2. Ask the user if they wish to generate a GBM parameters CSV from one of these indices.
      3. If yes, prompt for the index ticker (one of the available indices), and optionally
         for time horizon and steps.
      4. Generate the GBM parameters CSV file.
    """
    print("Analyzing the returns of the indices: ^GSPC, LQD, IEF")
    results = analyze_indices_returns()  # returns a dictionary with computed parameters for each index.
    
    # Ask the user if they want to generate a GBM parameters CSV.
    user_response = input("\nWould you like to generate a GBM parameters CSV from one of these indices? (yes/no): ").strip().lower()
    if user_response in ['yes', 'y']:
        print("Please enter the index ticker to use (e.g., ^GSPC, LQD, IEF):")
        chosen_ticker = input().strip()
        if chosen_ticker in results:
            params = results[chosen_ticker]
            # Allow user to optionally define simulation time horizon and steps.
            time_horizon_input = input("Enter the simulation time horizon (in years, default=1): ").strip()
            time_horizon = float(time_horizon_input) if time_horizon_input else 1
            steps_input = input("Enter the number of simulation steps (default=252): ").strip()
            steps = int(steps_input) if steps_input else 252
            
            create_gbm_parameter_csv(chosen_ticker, params["initial_value"], params["drift"], 
                                     params["volatility"], time_horizon, steps)
        else:
            print(f"Ticker {chosen_ticker} was not found in the results.")
    else:
        print("Skipping GBM parameters CSV generation.")

if __name__ == '__main__':
    main()
