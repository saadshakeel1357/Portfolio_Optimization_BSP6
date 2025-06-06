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

    shared_state = {
        "last_avg_weights": None,   # will hold the most recent window’s averaged weights
        "some_flag": False,         # example flag that GA could toggle
        # … add any other keys you know you’ll need later …
    }
    # ─── 1) Prepare a list to collect each window’s averaged weights ─────────
    collected_weights_per_window = []


    # ─── 2) Define a callback that will be invoked once per walk-forward window ─
    def on_window_complete(avg_weights_array):
        """
        This function is called by run_ga(...) as soon as one window’s final
        averaged weight vector is ready. We simply append it to our list.
        """
        print("→ Received averaged weights for one window:", avg_weights_array)
        collected_weights_per_window.append(avg_weights_array.copy())
        
        # Store those weights in the shared_state dict, so GA can see them if it wants
        shared_state["last_avg_weights"] = avg_weights_array.copy()
        print("shared state in window complete code: \n",shared_state["last_avg_weights"])


        # (Optionally flip a flag or record other info into shared_state here)
        shared_state["some_flag"] = True

    fitness_func_names = [
        'sharpe',
        'sortino',
        # 'omega',
        # 'value_at_risk',
        # 'expected_shortfall',
        # 'mean_variance',
        # 'mean_semivariance',
        # 'mean_mad',
        # 'minimax',
        # 'variance_with_skewness',
        # 'twosided',
        # 'risk_parity'
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
    
    percent = 0.2
    start_date = "2002-07-30"
    end_date = "2025-05-22"

    run_analysis(percent, start_date, end_date)

    # ─── 4) After run_ga has finished, you can inspect collected_weights_per_window: ─
    print("\nAll windows’ averaged weights collected:")
    for idx, wvec in enumerate(collected_weights_per_window, start=1):
        print(f" Window {idx}: {wvec}")