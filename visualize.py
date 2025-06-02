import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_and_combine_csvs(directory="."):
    """
    Finds all files named final_portfolio_values_*.csv in `directory`,
    reads them, and combines into a single DataFrame.
    """
    pattern = os.path.join(directory, "final_portfolio_values_*.csv")
    csv_files = glob.glob(pattern)

    # Dictionary to hold each series
    series_dict = {}

    for filepath in csv_files:
        # Extract the function-name part:
        # e.g. "final_portfolio_values_myfunc.csv" -> "myfunc"
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").split("_")
        # Last part(s) after "final", "portfolio", "values" form the function name.
        func_name = "_".join(parts[3:])

        # Read CSV, using the first column as the index
        df = pd.read_csv(filepath, index_col=0)
        # Assume the column is called "Final Portfolio Value"
        series = df["Final Portfolio Value"]
        series_dict[func_name] = series

    # Combine all series into one DataFrame (index = row numbers)
    combined_df = pd.DataFrame(series_dict)
    return combined_df

def plot_combined(df):
    """
    Plots each column of df as a separate line on the same axes.
    """
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.xlabel("Index (e.g., Time Step)")
    plt.ylabel("Final Portfolio Value")
    plt.title("Comparison of Portfolio Value Across Functions")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change "." to a specific folder if your CSVs aren't in the script's directory
    combined = load_and_combine_csvs(directory=".")
    plot_combined(combined)
