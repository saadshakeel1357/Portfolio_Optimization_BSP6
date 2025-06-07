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
from visualize import load_and_combine_csvs, plot_combined
from computing_yearly_deficit import run_analysis  


if __name__ == "__main__":

    percent = 0.2
    start_date = "2002-07-30"
    end_date = "2025-05-22"

    shared_state = {
        "last_avg_weights": None,   # will hold the most recent window’s averaged weights
        "yearly_deficit": None,  # will hold the yearly deficit value if needed
        # … add any other keys you know you’ll need later …
    }
    # ─── 1) Prepare a list to collect each window’s averaged weights ─────────
    collected_weights_per_window = []


    # ─── 2) Define a callback that will be invoked once per walk-forward window ─
    def on_window_complete(avg_weights_array):


        print("Inside window complete fuction \n")
        # print("Received averaged weights from ga code: \n", avg_weights_array)
        collected_weights_per_window.append(avg_weights_array.copy())
        
        # Store those weights in the shared_state dict, so GA can see them if it wants
        shared_state["last_avg_weights"] = avg_weights_array.copy()
        print("shared state in window complete code: \n")
        print(shared_state)

        # computing yearly deficit for testing
        w1, w2, w3 = shared_state["last_avg_weights"]
        # Apply weights
        df_weighted = apply_weights(df_cumulative, (w1, w2, w3))

        # Compute final series
        final_series = compute_final_portfolio_series(df_weighted)

        # Save to CSV with appropriate name
        output_file = f'final_portfolio_values_{fitness_name}.csv'
        save_series_to_csv(final_series, output_file)
        # print(f"Saved final portfolio series to TESTTTTT: {output_file}")

        shared_state["yearly_deficit"] = run_analysis(percent, start_date, end_date, fitness_name)
        print("Yearly deficit computed and stored in shared_state inside window complete code:", shared_state["yearly_deficit"])




    fitness_func_names = [
        'sharpe',
        'sortino_ratio',
        'omega_ratio',
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

    returns_file = 'merged_returns.csv'
    df = load_returns_csv(returns_file)
    df_cumulative = compute_cumulative_returns(df)

    for fitness_name in fitness_func_names:
        print(f"\nRunning GA with fitness function: {fitness_name}")

        # Run genetic algorithm with the current fitness function

        print("before running GA \n" )
        print(shared_state)
        result_turnover = run_ga(fitness_name, on_window_complete, shared_state)
        print("shared state in main code: \n",shared_state["last_avg_weights"])

        print("Last row of result_turnover:")
        print(result_turnover.tail(1))

        # Extract weights from the last row of result_turnover
        w1, w2, w3 = load_turnover_weights(result_turnover)
        # Apply weights
        df_weighted = apply_weights(df_cumulative, (w1, w2, w3))

        # Compute final series
        final_series = compute_final_portfolio_series(df_weighted)

        # Save to CSV with appropriate name
        output_file = f'final_portfolio_values_{fitness_name}.csv'
        save_series_to_csv(final_series, output_file)
        print(f"Saved final portfolio series to: {output_file}")
    
    
    # 6) Load all CSVs and plot final comparison
    combined_df = load_and_combine_csvs(directory=".")
    plot_combined(combined_df)


    # Set the following variables according to each dataset used:
    # See the computing_yearly_deficit.py file for more details
    


    

    # ─── 4) After run_ga has finished, you can inspect collected_weights_per_window: ─
    print("\nAll windows’ averaged weights collected:")
    for idx, wvec in enumerate(collected_weights_per_window, start=1):
        print(f" Window {idx}: {wvec}")