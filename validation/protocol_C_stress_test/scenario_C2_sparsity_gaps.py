import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
OUTPUT_DIR = "validation/protocol_C_stress_test"

def generate_data(n, seed=42):
    rng = np.random.default_rng(seed)
    # Trend + Signal + Noise
    t = np.arange(n)
    trend = 0.05 * t
    c1 = np.sin(t / 5.0) # Smooth covariate
    true_effect = 2.0 * c1
    noise = rng.standard_normal(n) * 0.1
    y = trend + true_effect + noise

    dates = pd.date_range(start="2020-01-01", periods=n, freq="ME")
    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'c1': c1,
        'true_effect': true_effect
    })
    df.set_index('date', inplace=True)
    return df

def create_gap_data(df, gap_start=50, gap_len=24):
    """
    Remove Y values in a contiguous block.
    X remains.
    """
    df_mod = df.copy()
    df_mod.iloc[gap_start:gap_start+gap_len, df_mod.columns.get_loc('y')] = np.nan
    return df_mod

def create_sparse_data(df, drop_prob=0.5, seed=42):
    """
    Remove Y values randomly.
    """
    rng = np.random.default_rng(seed)
    df_mod = df.copy()
    mask = rng.random(len(df)) < drop_prob
    df_mod.loc[mask, 'y'] = np.nan
    return df_mod

def run_validation():
    print("Generating Data (N=120)...")
    df_full = generate_data(N_SAMPLES, seed=RANDOM_STATE)

    # Decanters to test
    decanters = {
        "DoubleML": DoubleMLDecanter(random_state=RANDOM_STATE),
        "GAM": GamDecanter(),
        "Prophet": ProphetDecanter(),
        "ML (RandomForest)": MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        "ARIMA": ArimaDecanter(order=(1,0,0)),
        "FastLoess": FastLoessDecanter(),
        "GP": GPDecanter(random_state=RANDOM_STATE)
    }

    scenarios = {
        "Contiguous Gap (20%)": create_gap_data(df_full),
        "Sparse (50% Missing)": create_sparse_data(df_full)
    }

    results_data = []

    print("\nRunning Sparsity/Gap Tests...")

    for scenario_name, df_test in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")

        # Identify indices where Y is missing (Gap/Sparse regions)
        # This is where we want to check interpolation quality
        missing_mask = df_test['y'].isna()
        missing_idx = df_test.index[missing_mask]

        # Prepare training data (dropna)
        # We assume standard usage: fit on available Y, transform on full X.
        df_train = df_test.dropna(subset=['y'])

        for name, decanter in decanters.items():
            # Re-instantiate
            if name == "DoubleML": decanter = DoubleMLDecanter(random_state=RANDOM_STATE)
            elif name == "GAM": decanter = GamDecanter()
            elif name == "Prophet": decanter = ProphetDecanter()
            elif name == "ML (RandomForest)": decanter = MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))
            elif name == "ARIMA": decanter = ArimaDecanter(order=(1,0,0))
            elif name == "FastLoess": decanter = FastLoessDecanter()
            elif name == "GP": decanter = GPDecanter(random_state=RANDOM_STATE)

            print(f"  Testing {name}...")

            try:
                # 1. Fit on PARTIAL data (where Y is known)
                # Note: We must pass X corresponding to Y.
                decanter.fit(y=df_train['y'], X=df_train[['c1']])

                # 2. Transform on FULL data (X is complete)
                # We want to see if it can predict effect for the gap.
                # BaseDecanter.transform usually requires Y and X to be aligned.
                # But here Y is missing in gap.
                # If we pass Y with NaNs, check if transform handles it.
                # Or we can use `predict_batch` if available via mixin?
                # Or just pass full Y (with NaNs) and see if result has NaNs.
                # Most decanters compute `adjusted = y - effect`. If y is NaN, adjusted is NaN.
                # But `covariate_effect` might be valid!

                result = decanter.transform(y=df_test['y'], X=df_test[['c1']])

                effect_est = result.covariate_effect

                # Check for NaNs in effect
                # Some models (DoubleML) might put NaNs in the beginning/end due to splitting?
                # But here we fit on partial data.
                # If effect has NaNs in the gap, then it failed to interpolate.

                # Extract effect in the gap
                effect_gap = effect_est[missing_mask]
                truth_gap = df_full.loc[missing_mask, 'true_effect']

                if effect_gap.isna().any():
                    # Check how many
                    nan_pct = effect_gap.isna().mean()
                    status = "WARN (Partial Gap)" if nan_pct < 1.0 else "FAIL (No Gap Interp)"
                    rmse_gap = np.nan
                else:
                    rmse_gap = np.sqrt(mean_squared_error(truth_gap, effect_gap))
                    status = "PASS"

                results_data.append({
                    "Scenario": scenario_name,
                    "Model": name,
                    "RMSE (Gap)": rmse_gap,
                    "Status": status
                })

                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(df_full.index, df_full['true_effect'], label='True Effect', color='black', linestyle='--')
                plt.plot(df_test.index, df_test['y'], label='Observed Y (with Gaps)', color='gray', alpha=0.3, marker='.')
                plt.plot(result.covariate_effect.index, result.covariate_effect, label=f'Est Effect ({name})', color='red')

                # Highlight gap
                if scenario_name.startswith("Contiguous"):
                    plt.axvspan(missing_idx[0], missing_idx[-1], color='yellow', alpha=0.2, label='Gap')

                plt.title(f"{scenario_name} - {name}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"plot_C2_{scenario_name.split()[0]}_{name.replace(' ', '_')}.png"))
                plt.close()

            except Exception as e:
                print(f"    ERROR: {e}")
                results_data.append({
                    "Scenario": scenario_name,
                    "Model": name,
                    "RMSE (Gap)": np.nan,
                    "Status": f"ERROR: {str(e)}"
                })

    # Summary Report
    results_df = pd.DataFrame(results_data)
    print("\n--- Validation Results (Scenario C2: Data Sparsity & Gaps) ---")
    print(results_df.to_markdown(index=False))

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_C2.csv"), index=False)

if __name__ == "__main__":
    run_validation()
