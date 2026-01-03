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

def generate_multicollinear_data(n, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t

    # C1 is random
    c1 = rng.standard_normal(n)
    # C2 is perfectly correlated with C1 (plus tiny noise to avoid singular matrix if possible, but keep it high)
    # r > 0.99
    c2 = c1 + rng.standard_normal(n) * 0.01

    # True Effect depends on C1 only (beta=2.0)
    true_effect = 2.0 * c1

    y = trend + true_effect + rng.standard_normal(n) * 0.1

    dates = pd.date_range(start="2020-01-01", periods=n, freq="ME")
    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'c1': c1,
        'c2': c2,
        'true_effect': true_effect
    })
    df.set_index('date', inplace=True)
    return df

def run_validation():
    print("Generating Multi-Collinear Data (N=120, r>0.99)...")
    df = generate_multicollinear_data(N_SAMPLES, seed=RANDOM_STATE)

    # Check correlation
    corr = df['c1'].corr(df['c2'])
    print(f"Correlation(C1, C2): {corr:.6f}")

    # Decanters to test
    # FastLoess supports 1 covariate? If so we skip C2 or it fails?
    # Based on previous tests, FastLoess raises ValueError for >1 covariate.
    # So we will skip FastLoess for this 2-covariate test.

    decanters = {
        "DoubleML": DoubleMLDecanter(random_state=RANDOM_STATE),
        "GAM": GamDecanter(),
        "Prophet": ProphetDecanter(),
        "ML (RandomForest)": MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        "ARIMA": ArimaDecanter(order=(1,0,0)),
        # "FastLoess": FastLoessDecanter(), # Skips as it only supports 1D
        "GP": GPDecanter(random_state=RANDOM_STATE)
    }

    results_data = []

    print("\nRunning Multi-Collinearity Tests...")

    for name, decanter in decanters.items():
        print(f"  Testing {name}...")

        try:
            # Fit/Transform with BOTH C1 and C2
            # Models should figure out that they contain same info.
            # Total effect should match 2*C1.
            # If model splits beta=1 to C1 and beta=1 to C2, that's fine for total effect.
            # If model explodes (beta=1000 to C1, beta=-998 to C2), total effect might still be ok but variance high?

            result = decanter.fit_transform(y=df['y'], X=df[['c1', 'c2']])

            effect_est = result.covariate_effect.fillna(0)
            true_effect = df['true_effect']

            # RMSE of Total Effect
            rmse = np.sqrt(mean_squared_error(true_effect, effect_est))

            # Check for stability/explosion?
            # Hard to inspect internal coefficients generically for all models.
            # But if RMSE is low, then the explosion didn't destroy prediction.

            status = "PASS" if rmse < 0.5 else "WARN"

            results_data.append({
                "Model": name,
                "RMSE (Total Effect)": rmse,
                "Status": status
            })

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['true_effect'], label='True Effect (2*C1)', color='black', linestyle='--')
            plt.plot(df.index, result.covariate_effect, label=f'Est Effect ({name})', color='red')
            plt.title(f"Multi-Collinearity - {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"plot_C3_{name.replace(' ', '_')}.png"))
            plt.close()

        except Exception as e:
            print(f"    ERROR: {e}")
            results_data.append({
                "Model": name,
                "RMSE (Total Effect)": np.nan,
                "Status": f"ERROR: {str(e)}"
            })

    # Summary Report
    results_df = pd.DataFrame(results_data)
    print("\n--- Validation Results (Scenario C3: Multi-Collinearity) ---")
    print(results_df.to_markdown(index=False))

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_C3.csv"), index=False)

if __name__ == "__main__":
    run_validation()
