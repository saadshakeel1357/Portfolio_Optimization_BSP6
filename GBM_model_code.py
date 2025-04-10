import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''This code implements the Geometric Brownian Motion (GBM) model for simulating stock price paths over a specified time horizon,
it is used to model the stochastic behavior of stock prices.

Inputs:
1. CSV file containing parameters for the GBM process (initial stock price, drift, volatility, time horizon, steps).
2. User-specified process type (e.g., 'Stock') to load the corresponding parameters from the file.

Outputs:
1. Simulated stock price paths over time based on the GBM model and its graph.
2. Console output displaying the simulated stock prices.
3. Updated CSV file containing the original data along with the simulated stock prices for the process type.'''

class GBMModel:
    def __init__(self, file_path, process_type):
        # Load parameters from the CSV file
        self.params = self.load_parameters(file_path, process_type)
        self.s0 = self.params["initial_value"]  # Initial stock price
        self.mu = self.params["drift"]  # Drift term (expected return)
        self.sigma = self.params["volatility"]  # Volatility
        self.deltaT = self.params["time_horizon"]  # Time period (e.g., 1 year)
        self.dt = self.params["steps"]  # Time step size

    def load_parameters(self, file_path, process_type):
        # Read the data from the provided CSV file
        data = pd.read_csv(file_path)
        process_data = data[data["type"] == process_type]
        # Convert the relevant rows into a dictionary
        return {row["parameter"]: float(row["value"]) for _, row in process_data.iterrows()}

    def simulate_gbm(self):
        """Simulates stock prices using the Geometric Brownian Motion model."""
        n_step = int(self.dt)  # Number of steps
        dt = self.deltaT / n_step  # Calculate the time step size
        stock_prices = np.zeros(n_step + 1)
        stock_prices[0] = self.s0  # Set initial stock price

        for i in range(1, n_step + 1):
            # Generate a Wiener process increment
            dW = np.random.randn() * np.sqrt(dt)
            
            # Calculate the stock price using GBM formula
            stock_prices[i] = stock_prices[i - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW
            )
        
        # Time vector for plotting
        time_vector = np.linspace(0, self.deltaT, n_step + 1)
        return time_vector, stock_prices


if __name__ == "__main__":
    # Prompt the user for file input
    file_path = input("Enter the path to the file with parameters: ")
    process_type = input("Enter the process type (e.g., 'Stock'): ")

    # Read the CSV file
    try:
        data = pd.read_csv(file_path)

        # Initialize the GBM model
        gbm_model = GBMModel(file_path, process_type)

        # Simulate the GBM stock prices
        time_vector, stock_prices = gbm_model.simulate_gbm()
        print(f"Simulated Stock Prices:\n{stock_prices}")

        # Plot the GBM stock prices
        plt.plot(time_vector, stock_prices, marker='o', label="Simulated Stock Prices")
        plt.title("Geometric Brownian Motion Stock Price Simulation")
        plt.ylabel("Stock Price")
        plt.xlabel("Time (Years)")
        plt.legend()
        plt.grid()
        plt.show()

        # Preserve existing data and append new results
        original_data = pd.read_csv(file_path)

        # Save the results back to the CSV file
        result_data = pd.DataFrame({
            "Time": time_vector,
            "Simulated Stock Prices": stock_prices
        })

        # Combine original data with new results
        updated_data = pd.concat([original_data, result_data], ignore_index=True)
        updated_data.to_csv(file_path, index=False)
        print(f"Updated file saved to {file_path}.")

    except FileNotFoundError:
        print("The specified file was not found. Please check the path and try again.")
    except ValueError as e:
        print(f"Error: {e}")
