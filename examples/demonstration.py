
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decants import GamDecanter, ProphetDecanter, ArimaDecanter, DoubleMLDecanter

def generate_small_dataset(n=120):
    """
    Simulates 10 years of monthly data (N=120).
    Scenario: Retail Sales affected by 'Price' (Covariate) and a latent Trend + Seasonality.
    """
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=n, freq="M")

    # 1. Covariate: Price (Random Walk with drift)
    # Price affects Sales negatively.
    price = np.cumsum(np.random.normal(0, 0.1, n)) + 10

    # 2. Intrinsic Signal: Trend + Seasonality
    time_idx = np.arange(n)
    trend = 0.05 * time_idx # Linear growth
    seasonality = 2 * np.sin(2 * np.pi * time_idx / 12) # Annual cycle

    # 3. True Relation: Sales = 50 + Trend + Seasonality - 2 * Price + Noise
    noise = np.random.normal(0, 1, n)
    sales = 50 + trend + seasonality - 2 * price + noise

    return pd.Series(sales, index=dates), pd.DataFrame({'price': price}, index=dates)

def demo_methods():
    y, X = generate_small_dataset()
    print(f"Dataset Shape: {y.shape} (Simulating 10 years monthly)")

    # --- 1. GAM Decanter ---
    # Good for smooth non-linear effects. Robust on small data if n_splines is low.
    print("\n--- Running GAM Decanter ---")
    gam = GamDecanter(n_splines=10) # 10 splines is appropriate for N=120 (approx 1 per year)
    res_gam = gam.fit_transform(y, X)
    print(f"GAM Adjusted Variance Reduction: {1 - res_gam.adjusted_series.var()/y.var():.2%}")

    # --- 2. Prophet Decanter ---
    # Robust to outliers and seasonality.
    print("\n--- Running Prophet Decanter ---")
    prophet = ProphetDecanter()
    res_prophet = prophet.fit_transform(y, X)
    print(f"Prophet Adjusted Variance Reduction: {1 - res_prophet.adjusted_series.var()/y.var():.2%}")

    # --- 3. Double ML Decanter ---
    # Causal inference. For N=120, we use 'interpolation' (K-Fold) to use all data efficiently.
    print("\n--- Running DoubleML Decanter ---")
    dml = DoubleMLDecanter(splitter="kfold", n_splits=5)
    res_dml = dml.fit_transform(y, X)
    print(f"DoubleML Variance Reduction: {res_dml.stats['variance_reduction']:.2%}")
    print(f"DoubleML Orthogonality: {res_dml.stats['orthogonality']}")

    print("\nDemonstration complete. See 'DecantResult.plot()' for visualizations.")

if __name__ == "__main__":
    demo_methods()
