
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from sklearn.metrics import root_mean_squared_error

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Use top-level import as decants/__init__.py exposes them
from decants import (
    DoubleMLDecanter,
    GamDecanter,
    ProphetDecanter,
    ArimaDecanter,
    MLDecanter,
    FastLoessDecanter
)
from decants.utils.diagnostics import variance_reduction

# --- Configuration ---
PROTOCOL_NAME = "Protocol A3: Trend-Covariate Confounding"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42
N_SAMPLES = 120  # 10 years monthly
COVARIATE_BETA = 0.5
TARGET_TOTAL_SLOPE = 1.5 # 1.0 (Trend) + 0.5 (Covariate)

def generate_confounded_data(n: int, beta: float, seed: int = 42):
    """
    Generates data where Trend and Covariate are perfectly collinear.
    Trend = t
    Covariate = t
    Y = Trend + beta * Covariate + noise
      = t + beta * t + noise
      = (1 + beta) * t + noise

    If model attributes all to trend: Trend Effect ~ (1+beta)t, Covariate Effect ~ 0
    If model attributes all to covariate: Trend Effect ~ 0, Covariate Effect ~ (1+beta)t
    If model explodes: Trend = 1000t, Covariate = -999t + beta*t (Sum is correct, but individual components are wrong)
    """
    np.random.seed(seed)

    # Time index
    dates = pd.date_range(start='2015-01-01', periods=n, freq='ME')

    # Features
    # Normalized time t from 0 to 1 for stability in some models, but let's use simple range first to stress test
    t = np.arange(n).astype(float)

    # Trend
    trend = t

    # Covariate (perfectly collinear with trend)
    covariate = t

    # Noise
    noise = np.random.normal(0, 1.0, size=n)

    # Target
    # Y = Trend + Effect + Noise
    # Effect = beta * covariate
    true_effect = beta * covariate
    y = trend + true_effect + noise

    df = pd.DataFrame({
        'date': dates,
        'y': y,
        'covariate': covariate,
        'true_trend': trend,
        'true_effect': true_effect
    })
    # Set date index for Decanters that require it
    df.set_index('date', inplace=True)

    return df

def run_validation(decanter_cls, name: str, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    print(f"\n--- Testing {name} ---")

    # Initialize model
    try:
        model = decanter_cls(**kwargs)
    except Exception as e:
        # Some models don't take random_state (e.g. GAM, Prophet)
        # We retry without random_state if that was the error
        if "unexpected keyword argument 'random_state'" in str(e):
             # Try popping random state
             kwargs.pop('random_state', None)
             try:
                 model = decanter_cls(**kwargs)
             except Exception as e2:
                 print(f"Initialization failed (retry): {e2}")
                 return {'status': 'FAIL_INIT', 'error': str(e2)}
        else:
             print(f"Initialization failed: {e}")
             return {'status': 'FAIL_INIT', 'error': str(e)}

    # Fit and Transform
    # BaseDecanter.fit_transform(y, X)
    try:
        # Separate y and X
        y = df['y']
        X = df[['covariate']]

        res = model.fit_transform(y, X)
    except Exception as e:
        print(f"Fit/Transform failed: {e}")
        return {'status': 'FAIL_RUN', 'error': str(e)}

    # Analyze results
    # We want to check if coefficients exploded.
    # Estimated Effect
    est_effect = res.covariate_effect
    est_trend = res.adjusted_series # In Decants, Adjusted = Original - Effect. So Adjusted ~ Trend + Noise

    # Check for NaNs (DoubleML produces NaNs at start)
    valid_mask = ~np.isnan(est_effect)
    if valid_mask.sum() == 0:
         return {'status': 'FAIL_NAN', 'error': "All outputs are NaN"}

    # Metrics

    # 1. Coefficient Stability (Avoid Explosion)
    # We fit a linear regression to the estimated effect to get the implied beta
    # est_effect ~ hat_beta * t
    # If exploded, hat_beta would be huge (positive or negative)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X_cov = df.loc[valid_mask, ['covariate']]
    y_eff = est_effect[valid_mask]
    lr.fit(X_cov, y_eff)
    implied_beta = lr.coef_[0]

    # 2. Implied Trend Slope
    # est_trend ~ hat_alpha * t
    lr_trend = LinearRegression()
    # We use time index t for trend regression
    # t values are in 'covariate' or we recreate them
    t_idx = np.arange(len(df))[valid_mask].reshape(-1, 1)
    y_trend = est_trend[valid_mask]
    lr_trend.fit(t_idx, y_trend)
    implied_trend_slope = lr_trend.coef_[0]

    # Total Slope (should be approx 1 + beta = 1.5)
    total_slope = implied_beta + implied_trend_slope

    # 3. % Error in Total Slope (The ultimate truth metric)
    slope_error_pct = abs(total_slope - TARGET_TOTAL_SLOPE) / TARGET_TOTAL_SLOPE * 100

    # 4. RMSE (Goodness of Fit)
    # Predicted Y = Est Trend + Est Effect
    pred_y = est_trend[valid_mask] + est_effect[valid_mask]
    true_y = df.loc[valid_mask, 'y']
    rmse = root_mean_squared_error(true_y, pred_y)

    print(f"Implied Beta (Covariate Effect): {implied_beta:.4f}")
    print(f"Implied Trend Slope: {implied_trend_slope:.4f}")
    print(f"Total Slope (Target ~ 1.5): {total_slope:.4f}")
    print(f"Slope Error: {slope_error_pct:.2f}%")
    print(f"RMSE: {rmse:.4f}")

    # Criteria:
    # Explosion: |implied_beta| > 10 (arbitrary large threshold, real beta is 0.5)
    # Or |implied_trend_slope| > 10

    explosion = (abs(implied_beta) > 10) or (abs(implied_trend_slope) > 10)

    status = "PASS"
    if explosion:
        status = "FAIL_EXPLOSION"

    # Attribution
    attribution = "MIXED"
    if abs(implied_beta) < 0.1:
        attribution = "TREND"
    elif abs(implied_trend_slope) < 0.1:
        attribution = "COVARIATE"

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y'], label='Original Y', color='grey', alpha=0.5)
    plt.plot(df.index, df['true_trend'], 'k--', label='True Trend (Slope=1)', alpha=0.3)
    plt.plot(df.index, df['true_effect'], 'r--', label='True Effect (Slope=0.5)', alpha=0.3)

    plt.plot(df.index, res.adjusted_series, label='Est. Trend (Adj Series)', color='blue')
    plt.plot(df.index, res.covariate_effect, label='Est. Effect', color='red')

    plt.title(f"{name}: Confounding Test (Status: {status})\nBeta_hat={implied_beta:.2f}, Trend_hat={implied_trend_slope:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"plot_{name.lower().replace(' ', '_')}.png"))
    plt.close()

    return {
        'status': status,
        'implied_beta': implied_beta,
        'implied_trend_slope': implied_trend_slope,
        'total_slope': total_slope,
        'slope_error_pct': slope_error_pct,
        'rmse': rmse,
        'attribution': attribution
    }

def main():
    print(f"Generating data for {PROTOCOL_NAME}...")
    df = generate_confounded_data(N_SAMPLES, COVARIATE_BETA, seed=RANDOM_STATE)

    results = {}

    # List of models to test
    models = [
        (DoubleMLDecanter, "DoubleML", {'random_state': RANDOM_STATE}),
        (GamDecanter, "GAM", {'random_state': RANDOM_STATE}),
        (ProphetDecanter, "Prophet", {}),
        (ArimaDecanter, "ARIMA", {}),
        (MLDecanter, "ML Decanter", {'random_state': RANDOM_STATE}),
        (FastLoessDecanter, "FastLoess", {})
    ]

    summary = []

    for cls, name, kwargs in models:
        res = run_validation(cls, name, df, **kwargs)
        results[name] = res
        summary.append({
            'Model': name,
            'Status': res['status'],
            'Beta (Est)': res.get('implied_beta', np.nan),
            'Trend Slope (Est)': res.get('implied_trend_slope', np.nan),
            'Total Slope Error (%)': res.get('slope_error_pct', np.nan),
            'RMSE': res.get('rmse', np.nan),
            'Attribution': res.get('attribution', 'N/A')
        })

    summary_df = pd.DataFrame(summary)
    # Format float columns
    summary_df['Total Slope Error (%)'] = summary_df['Total Slope Error (%)'].map('{:.2f}%'.format)
    summary_df['RMSE'] = summary_df['RMSE'].map('{:.4f}'.format)

    print("\n--- Summary ---")
    print(summary_df)

    # Save summary to CSV for the report
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

if __name__ == "__main__":
    main()
