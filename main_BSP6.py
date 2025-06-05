# main.py
from ga_provaOutOfSampleLongShort import run_ga
from creating_final_portfolio_values import (
    load_returns_csv,
    compute_cumulative_returns,
    load_turnover_weights,
    apply_weights,
    compute_final_portfolio_series,
    save_series_to_csv
)






if __name__ == "__main__":
    print("calling run_ga()")
    result_turnover = run_ga()     # <-- this is your DataFrame of turnover weights
    print("Last row of result_turnover:")
    print(result_turnover.tail(1))

    # 1) Load returns
    returns_file = 'merged_returns.csv'
    df = load_returns_csv(returns_file)

    # 2) Compute cumulative
    df_cumulative = compute_cumulative_returns(df)

    num_windows = 5
    num_fitness = len(result_turnover) // num_windows
        
    fitness_func_names = [
        'sharpe',
        'sortino',
        'omega',
        'value_at_risk',
        'expected_shortfall',
        'mean_variance',
        'mean_semivariance',
        'mean_mad',
        'minimax',
        'variance_with_skewness',
        'twosided',
        'risk_parity'
    ]
    #  ─── Now loop: for each fitness index w, pick its weights from the LAST window (s=4)
    for w in range(fitness_func_names):
        # Row‐index for (window=4, fitness=w) is: 4 * num_fitness + w
        row_idx = (num_windows - 1) * num_fitness + w
        weight_row = result_turnover.iloc[row_idx]

        w1, w2, w3 = weight_row.iloc[0], weight_row.iloc[1], weight_row.iloc[2]

        # Apply exactly those three weights to columns 0,1,2 of df_cumulative:
        df_weighted = apply_weights(df_cumulative, (w1, w2, w3))

        # Build the final series (sum across all columns) and save to a CSV named by fitness:
        final_series = compute_final_portfolio_series(df_weighted)
        output_file = f'final_portfolio_values_{fitness_func_names[w]}.csv'
        save_series_to_csv(final_series, output_file)
    # ──────────────────────────────────────────────────────────────────────



    # # 3) **Extract weights from the last row of result_turnover**:
    # w1, w2, w3 = load_turnover_weights(result_turnover)

    # # 4) Apply those three weights to the first three columns of df_cumulative
    # df_weighted = apply_weights(df_cumulative, (w1, w2, w3))

    # # 5) Build the final series and save it
    # final_series = compute_final_portfolio_series(df_weighted)
    # output_file = 'final_portfolio_values_expected_shortfall.csv'
    # save_series_to_csv(final_series, output_file)
