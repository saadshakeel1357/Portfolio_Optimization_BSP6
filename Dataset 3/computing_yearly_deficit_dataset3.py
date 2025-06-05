import os
import glob
import pandas as pd
import pandas_market_calendars as mcal

# -------------------------------------------------------------------
# Set your target percentage here (e.g., 0.2 for 20% through each year)
PERCENT = 0.5
# -------------------------------------------------------------------

def value_by_year(filepath, start_date="2012-01-12", end_date="2024-09-04"):
    """
    Reads a CSV and computes the portfolio value closest to PERCENT through the trading year.
    
    Parameters:
    - filepath: path to the CSV file.
    - start_date, end_date: strings in "YYYY-MM-DD" format to define the trading calendar range.
    
    Returns:
    - A Series mapping each year to the selected portfolio value (at PERCENT into that year).
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
        target_index = int(round(PERCENT * (n - 1)))
        value = group.loc[target_index, 'Final Portfolio Value']
        selected_values.append((year, value))

    return pd.Series(dict(selected_values))


def compute_all(directory="."):
    """
    - Finds all files matching final_portfolio_values_*.csv in `directory`.
    - For each file, computes the Final Portfolio Value at PERCENT through each year.
    - Returns a DataFrame whose:
        • index = years (e.g. 2002, 2003, …, 2025)
        • columns = function names (derived from each filename)
        • values = the portfolio value at that PERCENT point.
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
            yearly_values = value_by_year(filepath)
            results[func_name] = yearly_values
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    combined = pd.DataFrame(results).sort_index()
    return combined


if __name__ == "__main__":
    print(f"\n📊 Computing portfolio values at {PERCENT*100:.0f}% through each trading year...\n")

    combined_df = compute_all(directory=".")

    if not combined_df.empty:
        print("\nPortfolio Value by Year (rows = Year, columns = Strategy):\n")
        print(combined_df.to_string())

        # For each strategy (column), sum up (1 - value) whenever value < 1.
        print("\nYearly Deficits per function (sum of 1 - value for years where value < 1):\n")
        for col in combined_df.columns:
            deficit = 0.0
            for year_value in combined_df[col]:
                if year_value < 1:
                    deficit += (1 - year_value)
            print(f"{col}: {deficit:.4f}")
    else:
        print("No data processed.")
