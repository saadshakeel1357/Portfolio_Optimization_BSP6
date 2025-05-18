import pandas as pd

# 1) List your three files here (full paths or relative)
file_paths = [
    'LQD.csv',   #corporate
    '^GSPC.csv', #S&P 500 
    'IEF.csv',   #treasury
]

### tasks for today:

# compute value of the portfolio fo each returns file. 
    # old value * (1 + return file value) #
# 1.009 * (1 + 0.004) example for the IEF_returns

# and each line of the new processed file is multiplied 
# by the last line value for each asset/bond in the K3turnover_adaptiveNikkeiLO file


# and then create one single file that contains the sum of all
# assets/bonds for each day(row)





# Will hold each ticker's DataFrame
dfs = []

for fp in file_paths:
    # ---- grab the ticker from the second line ("Ticker  IEF  IEF ...") ----
    with open(fp, 'r') as f:
        f.readline()                        # skip first header row
        ticker = f.readline().split()[1]    # second line, second token
    
    # ---- read the data, skipping that second header row ----
    df = pd.read_csv(
        fp,
        sep='\t',             # or sep=',' if comma-delimited
        skiprows=[1],         # drop the "Ticker ..." line
        header=0,             # first line as column names
        index_col=0,          # first column ("time") as index
        parse_dates=True
    )
    
    # ---- keep only the 'close' column and rename it to the ticker ----
    df = df[['close']].rename(columns={'close': ticker})
    dfs.append(df)

# ---- find the latest of the three start‐dates ----
start_dates = [df.index.min() for df in dfs]
max_start = max(start_dates)

# ---- truncate each series to dates >= max_start ----
dfs_truncated = [df.loc[df.index >= max_start] for df in dfs]

# ---- merge on the date index, only keeping dates common to all three ----
merged = pd.concat(dfs_truncated, axis=1).sort_index().dropna()

# ---- reset index so that 'Date' becomes the leftmost column ----
merged = merged.reset_index().rename(columns={'index': 'Date'})

# ---- write out the new CSV ----
#    This will include a header row: Date,LQD,^GSPC,IEF
merged.to_csv('merged_close_prices.csv', index=False)

print("Wrote merged_close_matrix.csv with shape", merged.shape)
