import pandas as pd

"""
This script reads multiple TSV (tab-separated) files containing close prices for different financial instruments,
aligns their dates, and merges them into a single DataFrame.
The final output will have the columns ordered by the ticker symbols of the instruments.
It assumes the first row of each file contains a header and the second row contains the ticker symbol.
"""

def extract_ticker(filepath):
    """
    Read the second line of the TSV file and extract the ticker symbol.
    """
    with open(filepath, 'r') as f:
        f.readline()               # skip header line
        second_line = f.readline().strip()
    # assume format: "Ticker<TAB><SYMBOL>"
    return second_line.split('\t')[1]


def load_close_series(filepath):
    """
    Load a TSV file, keep only the 'close' column, rename it to the ticker.
    """
    ticker = extract_ticker(filepath)
    df = pd.read_csv(
        filepath,
        sep='\t',            # tab-separated input
        skiprows=[1],        # skip ticker info line
        header=0,            # first row contains column names
        index_col=0,         # use time column as index
        parse_dates=True     # parse index as dates
    )
    return df[['close']].rename(columns={'close': ticker})


def sync_and_merge(dataframes):
    """
    Align all series to the same start date and merge into one DataFrame.
    """
    # find the latest start date among all series
    max_start = max(df.index.min() for df in dataframes)
    # truncate each series to dates >= max_start
    truncated = [df.loc[df.index >= max_start] for df in dataframes]
    # inner join on dates and drop any rows with missing data
    merged = pd.concat(truncated, axis=1).sort_index().dropna()
    return merged


def main():
    # list of input TSV files
    file_paths = [
        'EUCO.L.csv',     # corporate bonds
        '^STOXX50E.csv',   # stock index
        'VGEA.DE.csv',    # sovereign bonds
    ]

    # load each series into a list
    series_list = [load_close_series(fp) for fp in file_paths]

    # align and merge all series
    merged = sync_and_merge(series_list)

    # print the header (column names)
    print("Column headers:", merged.columns.tolist())

    # save the merged data to CSV (comma-separated)
    output_file = 'merged_close_prices.csv'
    merged.to_csv(output_file, index=False, header=False)

    print(f"Wrote {output_file} with shape {merged.shape}")


if __name__ == '__main__':
    main()
