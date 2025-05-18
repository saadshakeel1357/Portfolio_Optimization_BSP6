import pandas as pd

# Load the CSV files with tab delimiter and set column names
gspc_df = pd.read_csv("GSPC_returns.csv", sep="\t", names=["Date", "GSPC"], header=None, skiprows=1)
ief_df = pd.read_csv("IEF_returns.csv", sep="\t", names=["Date", "IEF"], header=None, skiprows=1)
lqd_df = pd.read_csv("LQD_returns.csv", sep="\t", names=["Date", "LQD"], header=None, skiprows=1)

# Convert Date columns to datetime format
gspc_df["Date"] = pd.to_datetime(gspc_df["Date"])
ief_df["Date"] = pd.to_datetime(ief_df["Date"])
lqd_df["Date"] = pd.to_datetime(lqd_df["Date"])

# Merge all three dataframes on the Date column (inner join to keep only common dates)
combined_df = gspc_df.merge(ief_df, on="Date", how="inner").merge(lqd_df, on="Date", how="inner")

# Sort by date to keep the timeline in order
combined_df = combined_df.sort_values("Date").reset_index(drop=True)

# Drop the Date column and reorder columns as LQD, GSPC, IEF
final_df = combined_df[["LQD", "GSPC", "IEF"]]

# Save to CSV if needed
final_df.to_csv("combined_returns.csv", index=False)

# Display the final result
print(final_df.head())
