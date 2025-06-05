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

    returns_file = 'merged_returns.csv'
    df = load_returns_csv(returns_file)
    df_cumulative = compute_cumulative_returns(df)

    for fitness_name in fitness_func_names:
        print(f"\nRunning GA with fitness function: {fitness_name}")

        # Run genetic algorithm with the current fitness function
        result_turnover = run_ga(fitness_name)
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
