import pandas as pd
import pandas_market_calendars as mcal

"""
This script processes a CSV file containing final portfolio values, aligns it with NYSE trading days,
and computes the first final portfolio value for each calendar year.
It assumes the CSV has a column "Final Portfolio Value" and no Date column.
It outputs the first final portfolio value for each year in the specified date range.
"""


# 1. Load your CSV
df = pd.read_csv('final_portfolio_values_mean_semivariance.csv')

# 2. Grab the NYSE calendar and pull all trading days between 2002-07-30 and 2025-05-22
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2002-07-30', end_date='2025-05-22')

# schedule.index is a DatetimeIndex of every trading day
trading_days = schedule.index

# 3. Check that we now have exactly as many trading days as rows in your CSV
if len(trading_days) != len(df):
    raise ValueError(
        f"Still a length mismatch: {len(trading_days)} trading days vs. {len(df)} rows"
    )

# 4. Assign those trading dates to your DataFrame
df['Date'] = trading_days

# 5. Pull out the year
df['Year'] = df['Date'].dt.year

# 6. Group by year
grouped = df.groupby('Year')

# 7. Loop over each year, print its first Final Portfolio Value, and (optionally) save
for year, group_df in grouped:
    first_value = group_df['Final Portfolio Value'].iloc[0]
    print(f"First Final Portfolio Value for {year}: {first_value}")
