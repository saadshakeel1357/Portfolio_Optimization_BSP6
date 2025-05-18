import pandas as pd

# Step 1: Load the returns data
df_returns = pd.read_csv('combined_returns.csv')
print("Initial returns data head:")
print(df_returns.head())

# Step 2: Compute cumulative portfolio values
df_values = (1 + df_returns).cumprod()
print("\nCumulative portfolio values head:")
print(df_values.head())

# Step 3: Read turnover Excel and inspect its columns
turn_df = pd.read_excel('K3turnover_adaptiveNikkeiLO.xlsx')
print("\nTurnover DataFrame columns:")
print(turn_df.columns.tolist())

# Grab the last entry of the *first three* columns, regardless of their names:
turn1 = turn_df.iloc[-1, 0]
turn2 = turn_df.iloc[-1, 1]
turn3 = turn_df.iloc[-1, 2]
print("\nTurnover factors (by position):")
print(f"  Column #1 ➔ {turn1}")
print(f"  Column #2 ➔ {turn2}")
print(f"  Column #3 ➔ {turn3}")

# Step 4: Apply each factor to the matching df_values column
col_names = df_values.columns.tolist()
df_final = df_values.copy()
df_final[col_names[0]] *= turn1
df_final[col_names[1]] *= turn2
df_final[col_names[2]] *= turn3

print("\nValues after applying turnover head:")
print(df_final.head())

# Step 5: Sum up each column
totals = df_final.sum()
print("\nTotal portfolio value for each asset:")
print(totals)

# Step 6: Save processed time series
df_final.to_csv('processed_portfolio_values.csv', index=False)

# Step 7: Save totals as a one-row CSV with tickers as headers
# This makes the CSV look like: LQD,GSPC,IEF \n 2304.6,4181.2,2387.6
totals_row = pd.DataFrame([totals.values], columns=totals.index)
totals_row.to_csv('portfolio_totals.csv', index=False)

print("\nAll done—files saved as:")
print(" • processed_portfolio_values.csv")
print(" • portfolio_totals.csv  (with LQD, GSPC, IEF as headers)")