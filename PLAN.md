# Project Plan: `decants` (Decant Time Series)

## Phase 0: Architecture & Design Pattern

**Objective:** Define the standard input/output contract so all methods (GAM, Prophet, etc.) behave consistently.

### 0.1. Directory Structure

Instruct the agent to set up the following structure:

```text
decants/
├── __init__.py
├── base.py             # Abstract Base Class (ABC)
├── objects.py          # Data classes for Results
├── methods/
│   ├── __init__.py
│   ├── gam.py          # pygam implementation
│   ├── arima.py        # statsmodels implementation
│   ├── prophet.py      # Prophet implementation
│   └── ml.py           # Scikit-learn residualization
├── utils/
│   ├── plotting.py     # Unified plotting logic
│   └── diagnostics.py  # Statistical checks (ACF, Correlation)
├── tests/
└── pyproject.toml

```

### 0.2. The `DecantResult` Object

We need a standardized object returned by every method.
**Action:** Create a Dataclass in `objects.py` containing:

* `original_series` (pd.Series):
* `adjusted_series` (pd.Series):
* `covariate_effect` (pd.Series):
* `model`: The fitted underlying model object (for introspection).
* `params`: Dictionary of parameters (coefficients, lambdas).
* `conf_int`: DataFrame (lower, upper) for the effect (where applicable).
* `stats`: Dictionary (RMSE, correlation reduction, etc.).
* `plot()`: A method attached to the result that calls the unified plotter.

### 0.3. The Base Class (`BaseDecanter`)

**Action:** Create an ABC in `base.py` that enforces:

* `fit(y, X, **kwargs)`
* `transform(y, X)`
* `fit_transform(y, X)` -> Returns `DecantResult`

---

## Phase 1: The "Gold Standard" – GAM Implementation

**Objective:** Implement Semi-Parametric Smoothing using `pygam`.
**Reference:** Section 2 of background research.

### 1.1. Dependency

* `pip install pygam`

### 1.2. Logic Implementation (`methods/gam.py`)

**Class:** `GamDecanter`
**Steps for Agent:**

1. **Input Handling:** Accept pandas Series/DataFrames. Align indices strictly.
2. **Model Configuration:**
* Allow user to specify `n_splines` and `lam` (smoothing).
* **Crucial:** Automatically include a spline or linear term for the *time index* `l(0)` or `s(0)` to prevent the covariate from absorbing the trend (as per Section 2.3.1).
* Construct the feature matrix: `[Time_Index, Covariate_1, Covariate_n]`.


3. **Fitting:** Use `LinearGAM`. Include a default `gridsearch` method for hyperparameter tuning.
4. **Effect Isolation:**
* Use `model.partial_dependence(term=i, X=X, meshgrid=False)`.
* **Constraint:** Ensure `meshgrid=False` is used to get point-wise effects for subtraction.


5. **Adjustment:** .
6. **Uncertainty:** Use `model.confidence_intervals()` to populate the `DecantResult`.

---

## Phase 2: The "Business" Standard – Prophet Implementation

**Objective:** Implement Bayesian Decomposition for business data/holidays.
**Reference:** Section 4 of background research.

### 2.1. Dependency

* `pip install prophet`

### 2.2. Logic Implementation (`methods/prophet.py`)

**Class:** `ProphetDecanter`
**Steps for Agent:**

1. **Data Prep:** Prophet requires columns named `ds` and `y`. Rename inputs internally, then map back to original indices on output.
2. **Regressor Registration:** Use `m.add_regressor(name)` for every covariate provided.
3. **Fitting:** `m.fit()`.
4. **Decomposition (Predict):** Call `m.predict()`.
5. **Extraction:**
* Identify the column in the forecast dataframe corresponding to the regressor.
* *Edge Case:* If multiple regressors are used, sum their columns or use `extra_regressors_additive`.


6. **Adjustment:**
* .
* *Warning:* Do not subtract from `yhat` (modeled y), subtract from `y` (actuals).



---

## Phase 3: The "Econometric" Standard – ARIMAX

**Objective:** Implement Parametric State-Space modeling.
**Reference:** Section 3 of background research.

### 3.1. Dependency

* `pip install statsmodels`

### 3.2. Logic Implementation (`methods/arima.py`)

**Class:** `ArimaDecanter`
**Steps for Agent:**

1. **Model:** Use `statsmodels.tsa.statespace.sarimax.SARIMAX`.
2. **Exog Handling:** Pass covariates as the `exog` argument.
3. **Fit:** `model.fit()`.
4. **Effect Isolation:**
* Extract parameters (`results.params`).
* Filter params to find only those associated with `exog` columns.
* Compute `effect = exog_matrix @ exog_params`.


5. **Adjustment:** .
6. **Diagnostics:** Include AIC/BIC in the results statistics.

---

## Phase 4: Machine Learning Residualization

**Objective:** Random Forest/Gradient Boosting with strict Time Series Cross-Validation.
**Reference:** Section 5 of background research.

### 4.1. Dependency

* `pip install scikit-learn`

### 4.2. Logic Implementation (`methods/ml.py`)

**Class:** `MLDecanter`
**Steps for Agent:**

1. **Model:** Allow user to pass an estimator (default: `RandomForestRegressor`).
2. **Cross-Validation:**
* **Strict Rule:** Use `sklearn.model_selection.TimeSeriesSplit`.
* *Do not* use standard K-Fold (prevents data leakage).


3. **Prediction:** Use `cross_val_predict` with the time-series splitter to generate out-of-sample predictions for the covariates' effect on .
4. **Adjustment:**
* .
* *Note:* The first fold will be lost due to CV windowing; handle the truncation of indices gracefully (fill with NaN or truncate ).



---

## Phase 5: Unified Visualization & Diagnostics

**Objective:** A beautiful summary of what happened.

### 5.1. Visualization (`utils/plotting.py`)

Create a function `plot_adjustment(result)` using `matplotlib` or `plotly`.
**Layout:**

* **Top Panel:** Overlay of Original (), Adjusted (), and Trend.
* **Middle Panel:** The Isolated Covariate Effect .
* **Bottom Panel:** Residuals of  (to check stationarity).

### 5.2. Diagnostics (`utils/diagnostics.py`)

Create functions to calculate:

* **Pearson Correlation:** Between  and  vs.  and  (Did we remove the correlation?).
* **Variance Reduction:** How much variance did the covariate explain?

---

## Phase 6: API Facade & Testing

**Objective:** Make it easy to import and use.

### 6.1. The `__init__.py`

Expose the classes:

```python
from .methods.gam import GamDecanter
from .methods.prophet import ProphetDecanter
from .methods.arima import ArimaDecanter

```

### 6.2. Testing Strategy

Instruct the agent to write tests for:

1. **Synthetic Data:** Generate data where .
2. **Recovery:** Assert that `Decanter` recovers the coefficient "2" (approx) and that  resembles the Trend + Noise.
3. **Shape Preservation:** Ensure input length equals output length.

---

## Execution Instructions for the Agent

1. **Start with Phase 0**: Set up the folder structure and the `DecantResult` dataclass.
2. **Implement Phase 1 (GAM)**: This is the highest priority method. Write the class and a test script using the synthetic data logic from Section 2.3.2 of the background research.
3. **Implement Phase 5 (Plotting)**: You need to see the results of Phase 1 to verify it works.
4. **Implement Phases 2, 3, and 4**: Sequentially adds the other methods.
5. **Final Polish**: Add docstrings and type hinting.
