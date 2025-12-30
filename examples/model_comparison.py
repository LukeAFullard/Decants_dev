
import pandas as pd
import numpy as np
from decants import DecantBenchmarker, GamDecanter, ProphetDecanter, DoubleMLDecanter, ArimaDecanter

def run_comparison():
    print("Generating synthetic data (N=150)...")
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=150, freq="W")

    # Trend + Seasonality + Covariate Effect
    time_idx = np.arange(150)
    trend = 0.1 * time_idx
    seasonality = 5 * np.sin(2 * np.pi * time_idx / 52)

    # Covariate: "Price"
    price = np.random.normal(100, 10, 150)
    # Effect: -0.5 * Price
    true_effect = -0.5 * price

    noise = np.random.normal(0, 2, 150)

    y = pd.Series(trend + seasonality + true_effect + noise, index=dates)
    X = pd.DataFrame({'price': price}, index=dates)

    print("Defining models to compare...")
    models = {
        "GAM (Splines)": GamDecanter(n_splines=10),
        "Prophet (Bayesian)": ProphetDecanter(),
        "DoubleML (Causal)": DoubleMLDecanter(splitter="kfold", n_splits=5),
        "ARIMA (Linear)": ArimaDecanter(order=(1,0,0))
    }

    print("Running benchmark...")
    bench = DecantBenchmarker()
    summary = bench.benchmark(models, y, X)

    print("\n--- Model Comparison Summary ---")
    print(summary[["Status", "Execution Time (s)", "Variance Reduction", "AIC"]])

    # Highlight best model
    if "Variance Reduction" in summary.columns:
        best_model = summary["Variance Reduction"].idxmax()
        print(f"\nBest model by Variance Reduction: {best_model}")

    print("\nDone. You can access individual results via bench.results['Model Name'].")

if __name__ == "__main__":
    run_comparison()
