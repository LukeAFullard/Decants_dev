import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import sys
import traceback

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

def generate_clean_data(n, seed=42):
    rng = np.random.default_rng(seed)
    # Simple linear trend + signal
    t = np.arange(n)
    trend = 0.1 * t
    c1 = rng.standard_normal(n)
    y = trend + 2.0 * c1 + rng.standard_normal(n) * 0.1

    dates = pd.date_range(start="2020-01-01", periods=n, freq="ME")
    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'c1': c1
    })
    df.set_index('date', inplace=True)
    return df

def inject_adversarial(df, case):
    """
    Inject adversarial data into the dataframe.
    """
    df_mod = df.copy()
    n = len(df)

    if case == "NaNs":
        # Inject NaNs in Covariate
        # Randomly set 10% to NaN
        idx = np.random.choice(df.index, size=int(n*0.1), replace=False)
        df_mod.loc[idx, 'c1'] = np.nan

    elif case == "Infs":
        # Inject Infs in Covariate
        idx = np.random.choice(df.index, size=5, replace=False)
        df_mod.loc[idx, 'c1'] = np.inf

    elif case == "Outliers":
        # Inject Massive Outliers (100 sigma)
        # Standard Normal -> 100 is huge
        idx = np.random.choice(df.index, size=3, replace=False)
        df_mod.loc[idx, 'c1'] = 100.0

    return df_mod

def run_validation():
    print("Generating Clean Data (N=120)...")
    df_clean = generate_clean_data(N_SAMPLES, seed=RANDOM_STATE)

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

    cases = ["NaNs", "Infs", "Outliers"]
    results_data = []

    print("\nRunning Stress Tests...")

    for case in cases:
        print(f"\n--- Case: {case} ---")
        df_test = inject_adversarial(df_clean, case)

        for name, decanter in decanters.items():
            # Re-instantiate to ensure clean state
            if name == "DoubleML": decanter = DoubleMLDecanter(random_state=RANDOM_STATE)
            elif name == "GAM": decanter = GamDecanter()
            elif name == "Prophet": decanter = ProphetDecanter()
            elif name == "ML (RandomForest)": decanter = MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))
            elif name == "ARIMA": decanter = ArimaDecanter(order=(1,0,0))
            elif name == "FastLoess": decanter = FastLoessDecanter()
            elif name == "GP": decanter = GPDecanter(random_state=RANDOM_STATE)

            print(f"  Testing {name}...")

            status = "UNKNOWN"
            outcome_msg = ""

            try:
                # Expectation:
                # NaNs: Some might handle (drop), some might error. Graceful handling preferred.
                # Infs: Should raise ValueError or clean it. Silent corruption is bad.
                # Outliers: Should not crash. Results might be skewed.

                result = decanter.fit_transform(y=df_test['y'], X=df_test[['c1']])

                # If we get here, it didn't crash.
                # Check for NaNs in output
                if result.adjusted_series.isna().all():
                     status = "FAIL (All NaNs)"
                     outcome_msg = "Result is all NaNs"
                else:
                    status = "PASS (Handled)"
                    outcome_msg = "Completed successfully"

                    if case == "Outliers":
                        # Check if outlier skewed the result excessively
                        # We compare RMSE of adjusted series vs clean adjusted series (roughly)
                        # Or just check if adjusted series has massive values.
                        if result.adjusted_series.abs().max() > 1000:
                            status = "WARN (Unstable)"
                            outcome_msg = "Output has massive values (>1000)"

            except ValueError as e:
                status = "PASS (Graceful Error)"
                outcome_msg = f"ValueError: {str(e).splitlines()[0]}" # First line only
            except Exception as e:
                status = "FAIL (Crash)"
                outcome_msg = f"{type(e).__name__}: {str(e)}"
                # traceback.print_exc()

            results_data.append({
                "Case": case,
                "Model": name,
                "Status": status,
                "Outcome": outcome_msg
            })

    # Summary Report
    results_df = pd.DataFrame(results_data)
    print("\n--- Validation Results (Scenario C1: Adversarial Inputs) ---")
    print(results_df.to_markdown(index=False))

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_C1.csv"), index=False)

if __name__ == "__main__":
    run_validation()
