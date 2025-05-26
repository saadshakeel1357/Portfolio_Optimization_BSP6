import pandas as pd

def load_returns_csv(filepath):
    """Load returns (futures) data from a CSV file."""
    df_returns = pd.read_csv(filepath)
    print("Initial returns data head:")
    print(df_returns.head())
    return df_returns

def compute_cumulative_returns(df_returns):
    """Compute cumulative portfolio values from returns (futures) data."""
    df_cumulative = (1 + df_returns).cumprod()
    print("\nCumulative portfolio values head:")
    print(df_cumulative.head())
    return df_cumulative

def load_turnover_weights(filepath):
    """Load turnover Excel file and return the last row's first three weights."""
    turn_df = pd.read_excel(filepath)

    # Extract weights from the last row's first three columns
    weight1 = turn_df.iloc[-1, 0]
    weight2 = turn_df.iloc[-1, 1]
    weight3 = turn_df.iloc[-1, 2]

    print("\nWeights (by position):")
    print(f"  Column #1 ➔ {weight1}")
    print(f"  Column #2 ➔ {weight2}")
    print(f"  Column #3 ➔ {weight3}")
    
    return weight1, weight2, weight3

def apply_weights(df_values, weights):
    """Multiply the first three columns of the cumulative values by the given weights."""
    df_weighted = df_values.copy()
    col_names = df_values.columns.tolist()

    # Apply each weight to its corresponding column
    df_weighted[col_names[0]] *= weights[0]
    df_weighted[col_names[1]] *= weights[1]
    df_weighted[col_names[2]] *= weights[2]

    print("\nWeighted cumulative values (head):")
    print(df_weighted.head())
    
    return df_weighted

def compute_final_portfolio_series(df_weighted):
    """Sum across weighted columns to get the final portfolio series."""
    final_series = df_weighted.sum(axis=1)
    print("\nFinal aggregated portfolio values over time:")
    print(final_series)
    return final_series

def save_series_to_csv(series, filepath):
    """Save a pandas Series to a CSV file."""
    df_to_save = series.to_frame(name='Final Portfolio Value')
    df_to_save.to_csv(filepath, index=True)
    print(f"\nFinal series saved to: {filepath}")

# --- Main script execution ---

if __name__ == "__main__":
    # File paths
    returns_file = 'combined_returns.csv'
    turnover_file = 'K3turnover_adaptiveNikkeiLO.xlsx'

    # change output file name according to the risk function being used for easy joining later
    output_file = 'final_portfolio_values_.csv'

    # Step-by-step execution
    df = load_returns_csv(returns_file)
    df_cumulative = compute_cumulative_returns(df)
    weights = load_turnover_weights(turnover_file)
    df_weighted = apply_weights(df_cumulative, weights)
    final_series = compute_final_portfolio_series(df_weighted)
    save_series_to_csv(final_series, output_file)
