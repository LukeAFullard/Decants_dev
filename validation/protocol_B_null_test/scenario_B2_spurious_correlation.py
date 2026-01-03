import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import sys

# Ensure we can import decants
sys.path.append(os.getcwd())

from decants import (
    DoubleMLDecanter,
    GamDecanter,
    ProphetDecanter,
    MLDecanter,
    ArimaDecanter,
    FastLoessDecanter,
    GPDecanter
)

# Configuration
N_SAMPLES = 120
RANDOM_STATE = 42
OUTPUT_DIR = "validation/protocol_B_null_test"

def generate_random_walks(n, seed=42):
    rng = np.random.default_rng(seed)

    # Random Walk 1 (Y)
    y_innovations = rng.standard_normal(n)
    y = np.cumsum(y_innovations)

    # Random Walk 2 (C) - Independent
    c_innovations = rng.standard_normal(n)
    c = np.cumsum(c_innovations)

    dates = pd.date_range(start="2020-01-01", periods=n, freq="ME")
    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'c1': c
    })
    df.set_index('date', inplace=True)
    return df

def run_validation():
    print("Generating Spurious Correlation Data (Random Walks)...")
    df = generate_random_walks(N_SAMPLES, seed=RANDOM_STATE)

    # Decanters to test
    # FastLoess supports 1 covariate, so we are good with just c1.

    decanters = {
        "DoubleML": DoubleMLDecanter(random_state=RANDOM_STATE),
        "GAM": GamDecanter(),
        "Prophet": ProphetDecanter(),
        "ML (RandomForest)": MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        "ARIMA": ArimaDecanter(order=(1,1,0)), # Use differencing (d=1) to handle non-stationarity? Or let it figure it out?
                                               # Ideally we test if default usage traps them.
                                               # But ARIMA usually requires d to be specified or auto-detected.
                                               # Let's give it d=1 to be fair to ARIMA as a "correct" specification for RW.
        "FastLoess": FastLoessDecanter(),
        "GP": GPDecanter(random_state=RANDOM_STATE)
    }

    results_data = []

    # Calculate Naive Spurious Correlation for baseline
    naive_corr = df['y'].corr(df['c1'])
    print(f"Naive Pearson Correlation (Y vs C): {naive_corr:.4f}")

    print("\nRunning Decanters...")
    for name, decanter in decanters.items():
        print(f"  - Running {name}...")
        try:
            y_series = df['y']
            X_df = df[['c1']]

            result = decanter.fit_transform(y=y_series, X=X_df)

            # Metric 1: Magnitude of Effect vs Magnitude of Signal
            # If the model thinks C is driving Y, Effect will be large.
            # If the model thinks it's just Trend/Noise, Effect will be small.
            # In a spurious regression, typically we find a "significant" coefficient.

            effect = result.covariate_effect.fillna(0)

            # RMSE of Effect (we want it to be 0)
            rmse_effect = np.sqrt(mean_squared_error(np.zeros_like(effect), effect))

            # Normalized RMSE (relative to std of Y)
            std_y = df['y'].std()
            nrmse_effect = rmse_effect / std_y

            # Correlation between Effect and C
            # If Effect = beta * C, then corr is 1.0 (or -1.0).
            # This checks if the model just attributed variance to C linearly.
            corr_effect_c = effect.corr(df['c1'])

            status = "PASS" if nrmse_effect < 0.3 else "WARN"
            # 0.3 is arbitrary but implies effect is < 30% of total variance.
            # High spurious correlation often explains > 50% of variance.

            results_data.append({
                "Model": name,
                "NRMSE (Effect)": nrmse_effect,
                "Corr(Effect, C)": corr_effect_c,
                "Status": status
            })

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['y'], label='Original (Random Walk Y)', color='gray', alpha=0.5)
            plt.plot(df.index, df['c1'], label='Covariate (Random Walk C)', color='blue', alpha=0.3, linestyle='--')
            plt.plot(df.index, result.covariate_effect, label=f'Estimated Effect ({name})', color='red', linewidth=2)
            plt.title(f"Scenario B2: Spurious Correlation - {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"plot_B2_{name.replace(' ', '_')}.png"))
            plt.close()

        except Exception as e:
            print(f"    ERROR in {name}: {e}")
            results_data.append({
                "Model": name,
                "NRMSE (Effect)": np.nan,
                "Corr(Effect, C)": np.nan,
                "Status": f"ERROR: {str(e)}"
            })

    # Summary Report
    results_df = pd.DataFrame(results_data)
    print("\n--- Validation Results (Scenario B2: Spurious Correlation) ---")
    print(results_df.to_markdown(index=False))

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_B2.csv"), index=False)

if __name__ == "__main__":
    run_validation()
