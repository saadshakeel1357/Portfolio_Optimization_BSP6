import os
import glob
import pandas as pd
import pandas_market_calendars as mcal

def value_by_year(filepath, percent, start_date, end_date):
    """
    Reads a CSV and computes the portfolio value closest to `percent` through the trading year.
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
            continue
        target_index = int(round(percent * (n - 1)))
        value = group.loc[target_index, 'Final Portfolio Value']
        selected_values.append((year, value))

    return pd.Series(dict(selected_values))


def compute_all(directory, percent, start_date, end_date):
    """
    Processes all CSV files and returns a DataFrame of portfolio values per strategy per year.
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
        func_name = "_".join(parts[3:])

        try:
            yearly_values = value_by_year(filepath, percent, start_date, end_date)
            results[func_name] = yearly_values
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return pd.DataFrame(results).sort_index()


def run_analysis(percent, start_date, end_date, directory="."):
    """
    Runs the full portfolio analysis pipeline.
    """
    print(f"\n📊 Computing portfolio values at {percent * 100:.0f}% through each trading year...\n")
    combined_df = compute_all(directory, percent, start_date, end_date)

    if not combined_df.empty:
        print("\nPortfolio Value by Year (rows = Year, columns = Strategy):\n")
        print(combined_df.to_string())

        print("\nYearly Deficits per function (sum of 1 - value for years where value < 1):\n")
        for col in combined_df.columns:
            deficit = sum((1 - val) for val in combined_df[col] if val < 1)
            print(f"{col}: {deficit:.4f}")
    else:
        print("No data processed.")


if __name__ == "__main__":
    percent = 0.2
    start_date = "2002-07-30"
    end_date = "2025-05-22"

    run_analysis(percent, start_date, end_date)



