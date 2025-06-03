import os
import glob
import pandas as pd
import pandas_market_calendars as mcal


"""
This script processes multiple CSV files containing final portfolio values,
aligns them with NYSE trading days, and computes the first final portfolio value for each calendar year.
It assumes each CSV has a column "Final Portfolio Value" and no Date column.
It outputs a DataFrame with the first final portfolio value for each year across all files.
It outputs the results to the console
"""


def first_value_by_year(filepath, start_date="2002-07-30", end_date="2025-05-22"):
    """
    1) Reads a single CSV (with an unnamed 0–N index and a column "Final Portfolio Value").
    2) Builds the exact NYSE trading‐day index from start_date to end_date.
    3) Attaches that index as a Date column.
    4) Groups by calendar year and returns a pd.Series mapping each Year → first Final Portfolio Value.
    """
    # 1. Load the CSV, use the existing integer column as index
    df = pd.read_csv(filepath, index_col=0)

    # 2. Grab the NYSE trading‐day calendar for that full date range
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index

    # 3. Make sure the count matches your CSV’s row count
    if len(trading_days) != len(df):
        raise ValueError(
            f"Length mismatch in {os.path.basename(filepath)}: "
            f"{len(trading_days)} trading days vs. {len(df)} rows"
        )

    # 4. Attach the trading_days index as a real Date column
    df['Date'] = trading_days

    # 5. Extract the calendar year from that Date
    df['Year'] = df['Date'].dt.year

    # 6. Group by Year and take the first value in "Final Portfolio Value"
    first_per_year = df.groupby('Year')['Final Portfolio Value'].first()

    return first_per_year

def compute_all(directory="."):
    """
    - Finds all files matching final_portfolio_values_*.csv in `directory`.
    - For each file, computes the first Final Portfolio Value of each calendar year.
    - Returns a DataFrame whose:
        • index = years (e.g. 2002, 2003, …, 2025)
        • columns = function names (derived from each filename)
        • values = the “first Final Portfolio Value” for that year & that file.
    """
    pattern = os.path.join(directory, "final_portfolio_values_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        print("No CSV files found with pattern:", pattern)
        return pd.DataFrame()

    results = {}
    for filepath in csv_files:
        # e.g. filepath = ".../final_portfolio_values_myfunc.csv"
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").split("_")
        # parts == ["final","portfolio","values","myfunc"]
        func_name = "_".join(parts[3:])  # “myfunc”, or “expected_shortfall”, etc.

        # Compute that file’s “first value per year” Series
        yearly_first = first_value_by_year(filepath)

        results[func_name] = yearly_first

    # Combine into a single DataFrame: rows = years, columns = each func_name
    combined = pd.DataFrame(results).sort_index()
    return combined

if __name__ == "__main__":
    # 1. Compute the table of “first Final Portfolio Value” by year for each CSV
    combined_df = compute_all(directory=".")

    # 2. Print it (and optionally save to CSV if you like)
    if not combined_df.empty:
        print("\nFirst Final Portfolio Value by Year (rows=Year, cols=Function):\n")
        print(combined_df.to_string())
        # If you want to save it:
        # combined_df.to_csv("yearly_first_portfolio_values_all.csv")
    else:
        print("No data processed.")
