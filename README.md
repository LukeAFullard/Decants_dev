# Decants: Robust Time Series Covariate Adjustment

**Decants** ("Decant Time Series") is a professional-grade Python library designed to isolate intrinsic signals from observed time series by removing the effects of exogenous covariates. It implements "Gold Standard" methodologies—from Semi-Parametric Smoothing (GAMs) to Bayesian Decomposition (Prophet) and Double Machine Learning—ensuring results are statistically defensible for high-stakes analysis (e.g., legal, financial, or scientific contexts).

## Features

*   **Generalized Additive Models (GAMs):** Non-linear adjustment using penalized splines (via `pygam`).
*   **Bayesian Decomposition:** Robust handling of holidays and events (via `Prophet`).
*   **ARIMAX:** Parametric state-space modeling for autocorrelated processes (via `statsmodels`).
*   **Double Machine Learning:** Causal residualization for high-dimensional confounding (via `sklearn`).
*   **Gaussian Processes:** Non-parametric Bayesian regression (Kriging) for irregularly sampled time series.
*   **LOESS (Generalized WRTDS):** Empirical, assumption-free adjustment for complex, non-stationary relationships.
*   **Robust Diagnostics:** Variance reduction, orthogonality checks, and correlation analysis.
*   **Unified API:** Consistent `fit`, `transform`, and `DecantResult` interface.

## Installation

```bash
pip install decants
```

*Note: Requires `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `prophet`, `pygam`, and `matplotlib`.*

## Quickstart

### 1. Basic Usage (GAM)

Ideally suited for removing non-linear effects (e.g., temperature) from a series.

```python
import pandas as pd
import numpy as np
from decants import GamDecanter

# Generate Data
dates = pd.date_range("2020-01-01", periods=120, freq="M")
y = pd.Series(np.random.randn(120), index=dates) # Target
X = pd.DataFrame({'temp': np.random.randn(120)}, index=dates) # Covariate

# Initialize & Fit
# Decants automatically handles time trends
model = GamDecanter(n_splines=10, lam=0.5)
result = model.fit_transform(y, X)

# Inspect
print(f"Variance Reduced: {result.stats['pseudo_r2']:.2%}")

# Plot
result.plot()
```

### 2. Handling Irregular Data (Gaussian Processes)

When data has gaps or irregular timestamps, standard lag-based methods fail. The `GPDecanter` uses Gaussian Processes to handle continuous time natively.

```python
from decants import GPDecanter

# Data with irregular gaps
decanter = GPDecanter(kernel_nu=1.5) # Matern 3/2 kernel for robustness
result = decanter.fit_transform(y_irregular, X_irregular)

# The result includes the isolated covariate effect and uncertainty bounds
print(result.stats['uncertainty'].head())
result.plot()
```

### 3. Complex Non-Stationary Relationships (LOESS)

When the relationship between the covariate and the target changes over time (e.g., a "Regime Shift"), standard global models fail. `FastLoessDecanter` builds a local correction surface.

```python
from decants import FastLoessDecanter

# Example: A covariate that has a positive effect in 2020 but negative in 2022
decanter = FastLoessDecanter(span=0.3, grid_resolution=50)
result = decanter.fit_transform(y, X)

# Visualize the locally varying effect
result.plot()
```

### 4. Handling Small Datasets (N=120)

For smaller datasets (e.g., 10 years of monthly data), overly complex models (TVP, Deep Learning) can overfit. **Decants** defaults to robust configurations:

*   **GAM:** Use fewer splines (`n_splines=5` to `10`) to prevent "wiggling" on noise.
*   **Prophet:** Use MCMC sampling or tighter priors if needed.
*   **DoubleML:** Use `interpolation` mode (K-Fold) instead of strict time-series splitting to maximize data usage for training.

### 5. Causal Inference (Double ML)

When you suspect "Ad Spend" drives "Sales", but "Seasonality" confounds both:

```python
from decants import DoubleMLDecanter

# Use 'interpolation' mode for small data efficiency (Cross-Fitting)
dml = DoubleMLDecanter(splitter="kfold", n_splits=5)
result = dml.fit_transform(sales_series, ad_spend_df)

print(f"Orthogonality Check: {result.stats['orthogonality']}")
```

## Methodology & Defensibility

Each method in `decants` adheres to strict statistical principles:
*   **Separation:** Trends are explicitly modeled to prevent covariates from absorbing secular growth.
*   **Validation:** Input indices are strictly aligned; misaligned data raises errors.
*   **Uncertainty:** Confidence intervals are calculated where theoretically appropriate.

## Saving Models (Audit Trail)

To ensure reproducibility in legal or audit contexts:

```python
model.save("audit_model_v1.pkl")
# Later...
loaded_model = GamDecanter.load("audit_model_v1.pkl")
```
