import os
import glob
import pandas as pd
import pandas_market_calendars as mcal

"""
This script processes multiple CSV files containing final portfolio values,
aligns them with NYSE trading days, and computes a portfolio value at a given percentage point of each calendar year.
It assumes each CSV has a column "Final Portfolio Value" and no Date column.
It outputs a DataFrame with the selected portfolio value for each year across all files.
It outputs the results to the console.
"""

def value_by_year_percent(filepath, start_date="2002-07-30", end_date="2025-05-22", percent=0.0):
    """
    Reads a CSV and computes the portfolio value closest to a specific percentage through the trading year.
    
    Parameters:
    - filepath: path to the CSV file.
    - percent: float (0.0 to 1.0), where 0.0 = start of year, 1.0 = end of year.

    Returns:
    - A Series mapping each year to the selected portfolio value.
    """
    df = pd.read_csv(filepath, index_col=0)

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index

    if len(trading_days) != len(df):
        raise ValueError(
            f"Length mismatch in {os.path.basename(filepath)}: "
            f"{len(trading_days)} trading days vs. {len(df)} rows"
        )

    df['Date'] = trading_days
    df['Year'] = df['Date'].dt.year

    selected_values = []
    for year, group in df.groupby('Year'):
        group = group.sort_values('Date').reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue  # skip empty years just in case
        target_index = int(round(percent * (n - 1)))
        value = group.loc[target_index, 'Final Portfolio Value']
        selected_values.append((year, value))

    return pd.Series(dict(selected_values))

def compute_all(directory=".", percent=0.0):
    """
    - Finds all files matching final_portfolio_values_*.csv in `directory`.
    - For each file, computes the Final Portfolio Value at the given percentage of each year.
    - Returns a DataFrame whose:
        • index = years (e.g. 2002, 2003, …, 2025)
        • columns = function names (derived from each filename)
        • values = the portfolio value at that percent point.
    """
    pattern = os.path.join(directory, "final_portfolio_values_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        print("No CSV files found with pattern:", pattern)
        return pd.DataFrame()

    results = {}
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").split("_")
        func_name = "_".join(parts[3:])  # e.g., “myfunc”

        try:
            yearly_values = value_by_year_percent(filepath, percent=percent)
            results[func_name] = yearly_values
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    combined = pd.DataFrame(results).sort_index()
    return combined

if __name__ == "__main__":
    # Change this to any value between 0.0 and 1.0 (e.g., 0.5 for mid-year, 1.0 for end-of-year)
    percent = 0.2

    print(f"\n📊 Computing portfolio values at {percent*100:.0f}% through each trading year...\n")

    combined_df = compute_all(directory=".", percent=percent)

    if not combined_df.empty:
        print("\nPortfolio Value by Year (rows = Year, columns = Strategy):\n")
        print(combined_df.to_string())
        # Optional save:
        # combined_df.to_csv(f"portfolio_values_{int(percent*100)}percent.csv")
    else:
        print("No data processed.")
