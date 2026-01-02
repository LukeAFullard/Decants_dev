<img src="assets/Decants_logo.png" width="600">

# Decants: Robust Time Series Covariate Adjustment

**Decants** ("Decant Time Series") is a professional-grade Python library designed to isolate intrinsic signals from observed time series by removing the effects of exogenous covariates. It implements "Gold Standard" methodologies—from Semi-Parametric Smoothing (GAMs) to Bayesian Decomposition (Prophet) and Double Machine Learning—ensuring results are statistically defensible for high-stakes analysis (e.g., legal, financial, or scientific contexts).

## Features

*   **Generalized Additive Models (GAMs):** Non-linear adjustment using penalized splines (via `pygam`).
*   **Bayesian Decomposition:** Robust handling of holidays and events (via `Prophet`).
*   **ARIMAX:** Parametric state-space modeling for autocorrelated processes (via `statsmodels`).
*   **Double Machine Learning:** Causal residualization for high-dimensional confounding (via `sklearn`).
*   **Gaussian Processes:** Non-parametric Bayesian regression (Kriging) for irregularly sampled time series.
*   **LOESS (Generalized WRTDS):** Empirical, assumption-free adjustment for complex, non-stationary relationships.
*   **Covariate Integration (Marginalization):** Risk-adjusted trend estimation via Monte Carlo Historical Replay.
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

### 4. Covariate Integration (Marginalization)

Sometimes you don't just want to remove the *value* of a covariate (e.g., "It was hot today"), but its *risk* (e.g., "This location is generally hot").

**Covariate Integration** answers the "Strategic Question": *"What would the trend look like if we faced 'Average' or 'Normal' risk conditions every day?"*

It uses **Monte Carlo Historical Replay** to simulate 100s of historical covariate scenarios for every time point, averaging the predictions to smooth out volatility.

*   **Mode 1: Forensic (Standard `transform`)**: Removes specific daily noise. "Did my factory break down *today*?"
*   **Mode 2: Strategic (`transform_integrated`)**: Removes the entire risk profile. "Is my business growing year-over-year, ignoring weather volatility?"

```python
from decants import FastLoessDecanter

# 1. Setup & Fit
model = FastLoessDecanter(span=0.5)
model.fit(time, sales, weather_data)

# 2. Get the "Forensic" view (Did we fail today?)
clean_sales = model.transform(time, weather_data).adjusted_series

# 3. Get the "Strategic" view (Are we growing long-term?)
# Replays history to normalize for climate risk
normalized_sales = model.transform_integrated(time, weather_data, n_samples=200)
```

*Compatibility Note: This feature is critical for Non-Linear models (LOESS, GAM, GP) where `Average(Input) != Output(Average)`. For Linear models (ARIMAX, Prophet), it is mathematically redundant and raises a warning.*

### 5. Handling Small Datasets (N=120)

For smaller datasets (e.g., 10 years of monthly data), overly complex models (TVP, Deep Learning) can overfit. **Decants** defaults to robust configurations:

*   **GAM:** Use fewer splines (`n_splines=5` to `10`) to prevent "wiggling" on noise.
*   **Prophet:** Use MCMC sampling or tighter priors if needed.
*   **DoubleML:** Use `interpolation` mode (K-Fold) instead of strict time-series splitting to maximize data usage for training.

### 6. Causal Inference (Double ML)

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

## Method Selection Guide

Choosing the right decanting method is critical for defensibility. Use the guide below to select the model that best fits your data assumptions.

| Method | Best Use Case | When to AVOID | Key Assumptions | Short Series (~120 pts)? |
| :--- | :--- | :--- | :--- | :--- |
| **GAM (Generalized Additive Models)** | **General Purpose.** Smooth, non-linear relationships (e.g., Temperature, Humidity). Interpretable. | High-dimensional data (>20 covariates). Sharp discontinuities or step-changes. | Relationships are smooth and additive. | **Caution** (Limit splines) |
| **Prophet** | **Business Data.** Strong seasonality, holidays, and clearly defined events. Robust to outliers. | Scientific data requiring precise physical modeling. Complex covariate interactions. | Trend is piecewise linear/logistic. Seasonality is Fourier-based. | **Caution** (Priors/MCMC) |
| **ARIMAX** | **Autocorrelation.** When residuals are clearly correlated in time (e.g., stock prices). Short-term forecasting. | Irregular sampling (gaps). Complex non-linearities. | Stationarity (after differencing). Linear covariate effects. | **Yes** (Efficient) |
| **Double ML** | **Causal Inference.** High-dimensional confounding. separating "Signal" from "Noise" without a strict functional form. | Small datasets (<100 points) where splitting reduces power. Simple linear problems. | Unconfoundedness (no hidden variables). Overlap. | **No** (Data hungry) |
| **Gaussian Processes (GP)** | **Irregular Data.** Missing data points or unequal timestamps. Small datasets where uncertainty quantification is key. | Large datasets (>2000 points) due to $O(N^3)$ slowness. | Kernel stationarity (unless specifically engineered). Gaussian noise. | **Yes** (Handles uncertainty) |
| **LOESS (Fast LOESS)** | **Regime Shifts.** Complex, changing relationships (e.g., Covariate effect flips over time). "Assumption-Free" audit requirements. | High dimensions (>2 covariates). Sparse data regions (needs neighbors). | Local smoothness. Effect is constant within small neighborhoods. | **Yes** (Adaptable) |
| **ML Decanter** | **High Complexity.** Pure prediction focus where "Black Box" is acceptable. Complex interactions. | Explanation is required (Audit/Legal). Small data (overfitting risk). | IID samples (unless TimeSeriesSplit is strictly used). | **No** (Overfitting risk) |
