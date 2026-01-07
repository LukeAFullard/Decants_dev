import numpy as np
import pandas as pd
import json
import hashlib
import os
import sys
import tempfile
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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
OUTPUT_DIR = "validation/protocol_D_defensibility"

def generate_data(n, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    c1 = np.sin(t / 5.0)
    y = trend + 2.0 * c1 + rng.standard_normal(n) * 0.1

    dates = pd.date_range(start="2020-01-01", periods=n, freq="ME")
    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'c1': c1
    })
    df.set_index('date', inplace=True)
    return df

def generate_interaction_data(n, seed=42):
    """
    Generates data where the covariate effect depends on time (Interaction).
    Marginalization (Integration) should handle this differently than simple subtraction
    if the model captures the interaction.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Trend
    trend = 0.05 * t

    # Covariate with time-varying effect (Interaction)
    # Effect starts at 0.5 and grows to 2.5
    beta_t = 0.5 + 2.0 * (t / n)
    c1 = np.random.uniform(-1, 1, n)

    # Y = Trend + Beta(t)*C + Noise
    y = trend + beta_t * c1 + rng.standard_normal(n) * 0.1

    dates = pd.date_range(start="2020-01-01", periods=n, freq="ME")
    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'c1': c1,
        'true_trend': trend,
        'true_effect': beta_t * c1
    })
    df.set_index('date', inplace=True)
    return df

def check_determinism():
    print("\n--- Test D1: Determinism ---")
    df = generate_data(N_SAMPLES, seed=RANDOM_STATE)

    # We test DoubleML as it has random components (splitters, nuisance models)
    # MLDecanter also has randomness.

    decanters = {
        "DoubleML": lambda: DoubleMLDecanter(random_state=RANDOM_STATE),
        "ML": lambda: MLDecanter(estimator=RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))
    }

    results = []

    for name, factory in decanters.items():
        # Run 1
        d1 = factory()
        res1 = d1.fit_transform(df['y'], df[['c1']])
        arr1 = res1.covariate_effect.values

        # Run 2
        d2 = factory()
        res2 = d2.fit_transform(df['y'], df[['c1']])
        arr2 = res2.covariate_effect.values

        # Check equality
        # Handle NaNs (nan == nan is False)
        # We fill NaNs with a unique value
        arr1_clean = np.nan_to_num(arr1, nan=-9999.0)
        arr2_clean = np.nan_to_num(arr2, nan=-9999.0)

        is_identical = np.array_equal(arr1_clean, arr2_clean)

        status = "PASS" if is_identical else "FAIL"
        print(f"  {name}: {status}")
        results.append({"Test": "Determinism", "Model": name, "Status": status})

    return results

def check_leakage():
    print("\n--- Test D2: Leakage (Time Travel) ---")
    # Concept: Prediction at T=60 should not change if we change data at T=100.
    # This applies to "Strict" Time Series models.

    df = generate_data(N_SAMPLES, seed=RANDOM_STATE)
    target_idx = 60
    future_idx = 100

    # Models to test
    # DoubleML (timeseries mode) - Should Pass
    # ML (TimeSeriesSplit) - Should Pass
    # ARIMA - Should Pass (filter mode)
    # Prophet - Should Pass? Prophet uses full history to fit parameters, so changing future changes parameters, which changes past fit?
    #   Actually, yes. Global trend fit depends on future data. So Prophet is NOT causal in the strict sense for historical adjustment.
    #   It's a "smoother".

    decanters = {
        "DoubleML (TS)": DoubleMLDecanter(splitter="timeseries", random_state=RANDOM_STATE),
        "ARIMA": ArimaDecanter(order=(1,0,0)),
        "ML (RF)": MLDecanter(estimator=RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))
    }

    results = []

    for name, decanter in decanters.items():
        # Run 1: Original Data
        # Re-instantiate
        if "DoubleML" in name: d = DoubleMLDecanter(splitter="timeseries", random_state=RANDOM_STATE)
        elif "ARIMA" in name: d = ArimaDecanter(order=(1,0,0))
        elif "ML" in name: d = MLDecanter(estimator=RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))

        res1 = d.fit_transform(df['y'], df[['c1']])
        val1 = res1.covariate_effect.iloc[target_idx]

        # Run 2: Altered Future
        df_mod = df.copy()
        df_mod.iloc[future_idx, df_mod.columns.get_loc('y')] += 100.0 # Massive outlier in future

        if "DoubleML" in name: d2 = DoubleMLDecanter(splitter="timeseries", random_state=RANDOM_STATE)
        elif "ARIMA" in name: d2 = ArimaDecanter(order=(1,0,0))
        elif "ML" in name: d2 = MLDecanter(estimator=RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE))

        res2 = d2.fit_transform(df_mod['y'], df_mod[['c1']])
        val2 = res2.covariate_effect.iloc[target_idx]

        # Check
        # Floating point tolerance
        diff = abs(val1 - val2)
        is_leakage = diff > 1e-9

        # DoubleML with TimeSeriesSplit:
        # T=60 is in a test set. The model for this set was trained on indices < 60.
        # Future data (T=100) is not in training set.
        # So it should be identical.

        # MLDecanter:
        # Same logic.

        status = "FAIL (Leakage Detected)" if is_leakage else "PASS"
        print(f"  {name}: {status} (Diff: {diff:.2e})")
        results.append({"Test": "Leakage", "Model": name, "Status": status})

    return results

def check_audit_completeness():
    print("\n--- Test D3: Audit Trail ---")
    df = generate_data(N_SAMPLES, seed=RANDOM_STATE)

    d = DoubleMLDecanter(random_state=RANDOM_STATE)
    d.fit_transform(df['y'], df[['c1']])

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        d.save(path)

        audit_path = path + ".audit.json"
        if not os.path.exists(audit_path):
            print("  FAIL: Audit file not created.")
            return [{"Test": "Audit", "Model": "Base", "Status": "FAIL"}]

        with open(audit_path, 'r') as f:
            log = json.load(f)

        # Verify Keys
        checks = {
            "created_at": "created_at" in log,
            "source_hash": "source_hash" in log.get("library_versions", {}),
            "version": "decants" in log.get("library_versions", {}),
            "history": len(log.get("history", [])) > 0
        }

        all_passed = all(checks.values())
        status = "PASS" if all_passed else f"FAIL ({checks})"
        print(f"  Audit Completeness: {status}")

        # Check Data Hash in history
        # We look for 'fit_transform_start' or 'fit_start'
        found_data_hash = False
        for entry in log['history']:
            if "details" in entry and ("y_hash" in entry["details"] or "X_hash" in entry["details"]):
                found_data_hash = True
                break

        print(f"  Data Hash Found: {found_data_hash}")

        final_status = "PASS" if all_passed and found_data_hash else "FAIL"

        return [{"Test": "Audit", "Model": "Base", "Status": final_status}]

def check_marginalization_effect():
    print("\n--- Test D4: Marginalization vs Forensic Mode ---")
    # Use Interaction Data where Beta varies with time.
    # Standard linear models might fail to capture this unless they are local.
    # Marginalization (integration) averages over the covariate distribution.

    df = generate_interaction_data(N_SAMPLES, seed=RANDOM_STATE)

    models = {
        "DoubleML": DoubleMLDecanter(random_state=RANDOM_STATE),
        "GAM": GamDecanter(),
        "Prophet": ProphetDecanter(),
        "ML (RF)": MLDecanter(estimator=RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE))
    }

    results = []

    # Setup Plot
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4 * len(models)), sharex=True)
    if len(models) == 1: axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        print(f"  Running {name}...")

        # 1. Forensic Mode (Standard fit_transform)
        # This calculates Adjusted = y - Effect(t, C_t)
        res_forensic = model.fit_transform(df['y'], df[['c1']])

        # 2. Strategic Mode (Marginalization)
        # This calculates Adjusted = E[y | t, distribution(C)]
        # For this to be different, C must vary or Effect must vary.
        # In this dataset, C is random uniform.
        try:
            # We pass the full history as the pool
            y_integrated = model.transform_integrated(
                t=df.index,
                C=df[['c1']],
                n_samples=50, # Lower samples for speed in test
                random_state=RANDOM_STATE
            )

            # Create a Series for plotting
            strategic_adjusted = pd.Series(y_integrated, index=df.index)
            strategic_effect = df['y'] - strategic_adjusted

            # Calculate Difference
            diff = np.abs(res_forensic.adjusted_series - strategic_adjusted).mean()
            status = "DIFFERENT" if diff > 0.01 else "SIMILAR"

            print(f"    Difference (MAE): {diff:.4f} -> {status}")

            # Plotting
            ax = axes[i]
            ax.plot(df.index, df['y'], color='gray', alpha=0.3, label='Original')
            ax.plot(df.index, df['true_trend'], color='black', linestyle='--', alpha=0.5, label='True Trend')
            ax.plot(df.index, res_forensic.adjusted_series, color='blue', label='Forensic Adjusted')
            ax.plot(df.index, strategic_adjusted, color='red', linestyle=':', linewidth=2, label='Strategic (Integrated)')

            ax.set_title(f"{name}: Forensic vs Strategic (MAE Diff: {diff:.3f})")
            ax.legend()

            results.append({
                "Test": "Marginalization",
                "Model": name,
                "MAE_Diff": diff,
                "Status": status
            })

        except Exception as e:
            print(f"    Failed to run marginalization for {name}: {e}")
            results.append({
                "Test": "Marginalization",
                "Model": name,
                "MAE_Diff": np.nan,
                "Status": "ERROR"
            })

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "marginalization_comparison.png"))
    print(f"  Saved comparison plot to {os.path.join(OUTPUT_DIR, 'marginalization_comparison.png')}")

    return results

def run_validation():
    results = []
    results.extend(check_determinism())
    results.extend(check_leakage())
    results.extend(check_audit_completeness())
    results.extend(check_marginalization_effect())

    # Summary Report
    results_df = pd.DataFrame(results)
    print("\n--- Validation Results (Protocol D: Defensibility) ---")
    print(results_df.to_markdown(index=False))

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_D.csv"), index=False)

if __name__ == "__main__":
    run_validation()
