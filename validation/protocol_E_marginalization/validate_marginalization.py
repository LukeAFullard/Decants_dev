
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from decants import (
    GamDecanter,
    MLDecanter,
    FastLoessDecanter,
    ProphetDecanter,
    DoubleMLDecanter
)

# Suppress warnings
warnings.filterwarnings("ignore")

PROTOCOL_NAME = "Protocol E: Marginalization vs Subtraction"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
N_SAMPLES = 365 * 3  # 3 years of daily data

def generate_nonlinear_weather_data(n=N_SAMPLES, seed=42):
    """
    Simulates Retail Sales driven by Temperature (Non-Linear).
    - Sales peak at 20°C (Comfort zone).
    - Sales drop in cold (0°C) and heat (35°C).
    - Temperature follows a seasonal pattern + anomalies.
    - Trend is linear growth.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    t = np.arange(n)

    # 1. Temperature (Covariate)
    # Seasonal (0 to 30) + Noise
    temp_seasonal = 15 + 15 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    temp_noise = np.random.normal(0, 3, n)
    temperature = temp_seasonal + temp_noise
    # Add a "Heatwave" anomaly in Year 2
    heatwave_mask = (dates >= "2021-07-01") & (dates <= "2021-07-15")
    temperature[heatwave_mask] += 10 # 40C+ spike

    # 2. Non-Linear Effect (Inverted Parabola / Gaussian-like)
    # Peak at 20, drop off.
    # Effect = 100 * exp(- (T - 20)^2 / 50)
    true_effect = 100 * np.exp(- (temperature - 20)**2 / 100)

    # 3. Trend (Intrinsic Signal)
    true_trend = 50 + 0.1 * t # Linear growth

    # 4. Target
    noise = np.random.normal(0, 5, n)
    sales = true_trend + true_effect + noise

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'y': sales,
        'temp': temperature,
        'true_trend': true_trend,
        'true_effect': true_effect
    })
    df.set_index('date', inplace=True)
    return df

def run_comparison(df):
    """
    Runs models and compares 'Adjusted' (Subtraction) vs 'Integrated' (Marginalization).
    """
    y = df['y']
    X = df[['temp']]

    # Models to test
    # Fix: MLDecanter does not take random_state in init, but in estimator
    models = [
        ("GAM", GamDecanter(n_splines=15, strict=True)),
        ("FastLoess", FastLoessDecanter(strict=True)),
        ("ML (RandomForest)", MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), strict=True)),
        ("Prophet", ProphetDecanter(strict=True)),
        ("DoubleML", DoubleMLDecanter(random_state=RANDOM_STATE, allow_future=True, strict=True)) # Allow future for interpolation
    ]

    results = []

    plt.figure(figsize=(15, 10))

    # Plot Raw Data
    plt.subplot(3, 2, 1)
    plt.plot(df.index, df['y'], color='gray', alpha=0.5, label='Original Sales')
    plt.plot(df.index, df['true_trend'], color='black', linestyle='--', label='True Trend (Base)')
    # Expected Sales at "Average" Weather?
    # Avg effect over history:
    avg_effect = df['true_effect'].mean()
    plt.plot(df.index, df['true_trend'] + avg_effect, color='green', linestyle=':', label='True Trend + Avg Effect')
    plt.title("Ground Truth")
    plt.legend()

    subplot_idx = 2

    for name, model in models:
        print(f"Running {name}...")
        try:
            # 1. Standard Fit/Transform (Subtraction)
            res = model.fit_transform(y, X)
            adj_subtraction = res.adjusted_series
            # Subtraction removes the effect of Temp *at that moment*.
            # So Adjusted = Y - f(T).
            # If f(T) is correctly estimated, Adjusted ~ Trend + Noise.

            # 2. Marginalization (Integration)
            # "What would sales be if Temperature was distributed like history?"
            # transform_integrated returns the PREDICTION (Trend + E[Effect]).
            # To get the "Integrated Trend" comparable to Adjusted, we typically look at the output itself.
            # The output of transform_integrated is "Y normalized to climate".
            # So it should be ~ Trend + Avg_Effect.

            # We use the full history as the "Climate" pool
            t_input = df.index
            X_input = X.values # Pass full history pool
            integrated_pred = model.transform_integrated(t_input, X_input, n_samples=50, random_state=RANDOM_STATE)
            integrated_series = pd.Series(integrated_pred, index=df.index)

            # Metrics
            # A. How close is Subtraction to "Base Trend"?
            # Filter NaNs (DoubleML/ML produce NaNs at start/splits)
            mask_sub = ~np.isnan(adj_subtraction)
            rmse_sub = root_mean_squared_error(df.loc[mask_sub, 'true_trend'], adj_subtraction[mask_sub])

            # B. How close is Integration to "Base Trend + Avg Effect"?
            target_integrated = df['true_trend'] + avg_effect
            rmse_int = root_mean_squared_error(target_integrated, integrated_series)

            results.append({
                "Model": name,
                "RMSE_Subtraction": rmse_sub,
                "RMSE_Integration": rmse_int,
                "Diff": rmse_int - rmse_sub
            })

            # Plot
            plt.subplot(3, 2, subplot_idx)
            plt.plot(df.index, df['y'], color='gray', alpha=0.2)
            plt.plot(df.index, df['true_trend'], 'k--', alpha=0.5, label='True Trend')
            plt.plot(df.index, adj_subtraction, label='Subtraction (Adj)', color='blue', linewidth=1)
            plt.plot(df.index, integrated_series, label='Marginalization (Int)', color='red', linewidth=1)
            plt.title(f"{name} (Int RMSE: {rmse_int:.2f})")
            plt.legend(fontsize='small')

            subplot_idx += 1

        except Exception as e:
            print(f"Failed {name}: {e}")
            import traceback
            traceback.print_exc()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "results_marginalization.png"))
    plt.close()

    # Save CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print("\nResults:")
    print(res_df)

if __name__ == "__main__":
    print(f"Generating data for {PROTOCOL_NAME}...")
    df = generate_nonlinear_weather_data()
    run_comparison(df)
