import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Any, List

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Import all decanters
from decants import (
    DoubleMLDecanter,
    GamDecanter,
    ProphetDecanter,
    ArimaDecanter,
    MLDecanter,
    GPDecanter,
    FastLoessDecanter
)
from decants.utils.diagnostics import check_orthogonality

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
PROTOCOL_NAME = "Protocol A1: Standard Signal Recovery (Orthogonal)"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
N_SAMPLES = 730  # 2 years
BETA_TRUE = 2.0
NOISE_SIGMA = 1.0

# --- 1. Data Generation ---
print(f"[{PROTOCOL_NAME}] Generating Data...")
np.random.seed(RANDOM_STATE)

dates = pd.date_range(start="2023-01-01", periods=N_SAMPLES, freq="D")
t = np.arange(N_SAMPLES)

# Trend: Linear + Seasonal
trend = 0.05 * t + 5.0 * np.sin(2 * np.pi * t / 365.25)
trend_series = pd.Series(trend, index=dates, name="True Trend")

# Covariate: Stationary (White Noise) to ensure orthogonality with Trend
c = np.random.normal(0, 1.0, size=N_SAMPLES)
covariate_series = pd.Series(c, index=dates, name="Covariate")

# Target: Y = Trend + Beta * C + Noise
noise = np.random.normal(0, NOISE_SIGMA, size=N_SAMPLES)
y = trend + BETA_TRUE * c + noise
y_series = pd.Series(y, index=dates, name="Target")

# Save raw data plot
plt.figure(figsize=(10, 5))
plt.plot(dates, y_series, label="Target (Y)", alpha=0.7)
plt.plot(dates, trend_series, label="True Trend", linestyle="--", color="black")
plt.plot(dates, covariate_series, label="Covariate (C)", alpha=0.5)
plt.legend()
plt.title(f"{PROTOCOL_NAME} - Input Data")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "raw_data.png"))
plt.close()

# --- 2. Model Definitions ---
# Dictionary of models to test
models_to_test = {
    "DoubleML": DoubleMLDecanter(random_state=RANDOM_STATE, allow_future=True, strict=True),
    "GAM": GamDecanter(strict=True), # Auto-tunes
    "Prophet": ProphetDecanter(strict=True),
    # Fixed argument name: 'estimator' instead of 'model'
    "ML (RandomForest)": MLDecanter(estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), strict=True),
    "GaussianProcess": GPDecanter(strict=True),
    "FastLoess": FastLoessDecanter(strict=True),
    # ARIMA might need specific order, using default auto/simple if possible, or specifying order
    # Default ArimaDecanter might not auto-tune well without parameters.
    # Let's try a standard (1,1,1) if not auto.
    "ARIMA(1,1,1)": ArimaDecanter(order=(1,1,1), strict=True)
}

results_summary = []

# --- 3. Execution Loop ---
print(f"[{PROTOCOL_NAME}] Testing {len(models_to_test)} models...")

for name, decanter in models_to_test.items():
    print(f"  > Running {name}...")
    try:
        # Fit & Transform
        # For MLDecanter, fit_transform performs CV.
        # For DoubleMLDecanter, fit is naive, transform is CV/CrossFit.
        # For others, fit then transform is standard.
        # However, MLDecanter.transform() is essentially "predict on training data" if we just fit() it.
        # We should use fit_transform() if available for best OOS simulation, but for Signal Recovery
        # we often just want to know if the model structure captures the effect.

        # We will use fit -> transform for uniformity, unless fit_transform is preferred.
        # Actually, fit_transform is preferred for ML-based ones to avoid overfitting.

        if hasattr(decanter, "fit_transform"):
            result = decanter.fit_transform(y_series, covariate_series)
        else:
            decanter.fit(y_series, covariate_series)
            result = decanter.transform(y_series, covariate_series)

        # --- Metrics ---
        # A. Beta Recovery
        valid_idx = result.covariate_effect.dropna().index
        if len(valid_idx) < 10:
            raise ValueError("Insufficient predictions returned.")

        c_valid = covariate_series.loc[valid_idx].values.reshape(-1, 1)
        eff_valid = result.covariate_effect.loc[valid_idx].values

        # Estimate Beta from the effect
        reg = LinearRegression().fit(c_valid, eff_valid)
        beta_est = reg.coef_[0]
        beta_error_pct = abs(beta_est - BETA_TRUE) / BETA_TRUE * 100

        # B. Diagnostics
        ortho = check_orthogonality(result.adjusted_series, covariate_series)
        max_corr = ortho.get('max_abs_corr', np.nan)

        # C. Trend Recovery (RMSE)
        # Note: Adjusted series might be shifted (intercept), so we compare standard deviations or centered RMSE
        # or just report raw RMSE and explain in notes.
        adj_valid = result.adjusted_series.loc[valid_idx]
        trend_valid = trend_series.loc[valid_idx]
        rmse_trend = np.sqrt(mean_squared_error(trend_valid, adj_valid))

        # Store result
        results_summary.append({
            "Model": name,
            "Beta_Est": beta_est,
            "Error_%": beta_error_pct,
            "Max_Corr": max_corr,
            "Trend_RMSE": rmse_trend,
            "Status": "PASS" if beta_error_pct < 15 else "FAIL (Accuracy)" # 15% tolerance
        })

        # --- Visualization ---
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        ax[0].plot(dates, y_series, label="Original", color="gray", alpha=0.5)
        ax[0].plot(dates, result.adjusted_series, label="Adjusted", color="blue")
        ax[0].plot(dates, trend_series, label="True Trend", color="black", linestyle="--")
        ax[0].set_title(f"{name} - Signal Recovery")
        ax[0].legend()

        ax[1].plot(dates, result.covariate_effect, label="Est Effect", color="red")
        ax[1].plot(dates, BETA_TRUE * covariate_series, label="True Effect", color="green", linestyle="--")
        ax[1].set_title("Covariate Effect")

        residuals = result.adjusted_series - trend_series
        ax[2].plot(dates, residuals, label="Residuals", color="purple")
        ax[2].set_title("Residuals")

        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "-")
        plt.savefig(os.path.join(OUTPUT_DIR, f"results_{safe_name}.png"))
        plt.close()

        # Save Audit
        decanter.save(os.path.join(OUTPUT_DIR, f"checkpoint_{safe_name}.pkl"))

    except Exception as e:
        print(f"    ! Failed: {e}")
        results_summary.append({
            "Model": name,
            "Beta_Est": np.nan,
            "Error_%": np.nan,
            "Max_Corr": np.nan,
            "Trend_RMSE": np.nan,
            "Status": f"ERROR: {str(e)[:50]}"
        })

# --- 4. Generate Summary Report ---
print(f"[{PROTOCOL_NAME}] Generating Report...")
df_res = pd.DataFrame(results_summary)
print(df_res)

# Markdown Report
md_report = f"""# Validation Report: Protocol A1 (All Models)

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Protocol:** Standard Signal Recovery (Orthogonal)
**Scenario:** Linear Trend + Seasonal + White Noise Covariate ($\beta=2.0$)

## 1. Summary of Results

| Model | Beta Est | Error % | Orthogonality (Corr) | Status |
| :--- | :--- | :--- | :--- | :--- |
"""

for row in results_summary:
    # Format numbers
    beta = f"{row['Beta_Est']:.3f}" if pd.notnull(row['Beta_Est']) else "NaN"
    err = f"{row['Error_%']:.1f}%" if pd.notnull(row['Error_%']) else "-"
    corr = f"{row['Max_Corr']:.4f}" if pd.notnull(row['Max_Corr']) else "-"

    md_report += f"| **{row['Model']}** | {beta} | {err} | {corr} | {row['Status']} |\n"

md_report += """
## 2. Detailed Analysis

### DoubleML
- **Method:** Uses cross-fitting with Ridge regression to isolate residuals.
- **Performance:** Expected to perform well on linear signals.
- **Notes:** Strict time-series splitting was relaxed (Interpolation Mode) to allow full-dataset recovery for this test.

### GAM (Generalized Additive Models)
- **Method:** Splines for trend + Linear/Spline for covariates.
- **Expectation:** Should capture the linear effect perfectly as $f(x)=x$ is a valid spline.

### Prophet
- **Method:** Bayesian additive model (Trend + Seasonality + Regressors).
- **Expectation:** Prophet is designed exactly for this (Time + Regressors).

### Machine Learning (Random Forest)
- **Method:** Non-parametric regression.
- **Risk:** Might overfit noise or struggle to extrapolate if covariates drift (though here they are stationary).
- **Performance:** Likely noisier than linear methods.

### Gaussian Process
- **Method:** Kernel-based regression.
- **Expectation:** High accuracy but computationally expensive. Handles uncertainty well.

### Fast LOESS
- **Method:** Local regression.
- **Risk:** Might smooth over the covariate effect if the window is too large or data too dense.

### ARIMA
- **Method:** Linear dynamic regression with errors.
- **Expectation:** Should handle stationary covariates well via ARIMAX formulation.

## 3. Artifacts
"""

# Add image links
for row in results_summary:
    safe_name = row['Model'].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "-")
    if "ERROR" not in row['Status']:
        md_report += f"### {row['Model']}\n"
        md_report += f"![{row['Model']}](results_{safe_name}.png)\n\n"

# Write README
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(md_report)

print(f"[{PROTOCOL_NAME}] Validation Suite Complete.")
