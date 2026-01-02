# Decants Validation Plan

**Date:** February 2025
**Scope:** Validation of the `decants` library for high-stakes/legal usage.
**Objective:** Establish a formal protocol to prove that the library produces accurate, defensible, and reproducible covariate adjustments.

## 1. Validation Philosophy
To be admissible in a court of law, the software must demonstrate:
1.  **Correctness:** It accurately removes covariate effects when they exist.
2.  **Conservative Behavior:** It does not "hallucinate" effects where none exist (Type I Error control).
3.  **Traceability:** Every result can be traced back to exact code versions, parameters, and data inputs.

## 2. Core Validation Protocols

### Protocol A: Synthetic Ground Truth Recovery
*Objective: Prove the model can recover a known signal buried in noise.*

1.  **Data Generation:**
    *   Simulate a time series $Y_t = Trend_t + f(C_t) + \epsilon_t$.
    *   $Trend_t$: Known function (e.g., linear + seasonal).
    *   $f(C_t)$: Known covariate effect (e.g., $3 \times Temperature$).
    *   $\epsilon_t$: Gaussian noise.
2.  **Execution:**
    *   Run `Decanter.transform(Y, C)`.
3.  **Success Criteria:**
    *   The estimated `covariate_effect` must correlate with the true $f(C_t)$ ($R^2 > 0.9$).
    *   The `adjusted_series` must match the true $Trend_t$ within a defined error margin (RMSE).

### Protocol B: The "Null" Test (Placebo Verification)
*Objective: Prove the model does not fabricate effects.*

1.  **Data Generation:**
    *   $Y_t$: Random Walk or Pure Trend.
    *   $C_t$: Random noise (uncorrelated with $Y_t$).
2.  **Execution:**
    *   Run `Decanter.transform(Y, C)`.
3.  **Success Criteria:**
    *   The estimated `covariate_effect` should be statistically indistinguishable from zero.
    *   `variance_reduction` metric should be negligible or negative.

### Protocol C: Method-Specific Edge Cases

| Method | Scenario | Success Criteria |
| :--- | :--- | :--- |
| **DoubleML** | High-dimensional noise (100 useless covariates) | LASSO/Ridge correctly shrinks coefficients to zero. |
| **FastLoess** | Singular Matrix (Constant Covariate) | Falls back to weighted mean; does not crash. |
| **GP** | Large gaps in time series | Uncertainty intervals (std) increase in gap regions. |
| **Prophet** | Missing Dates (NaT) | Gracefully handles gaps or raises descriptive error. |

## 3. Defensibility & Integrity Validation

### Audit Trail Verification
*   **Action:** Initialize a model, fit, save, and load.
*   **Check:** Verify `.audit.json` contains:
    *   `source_hash`: Matches the SHA-256 of the library source.
    *   `library_versions`: Matches current environment.
    *   `history`: Contains timestamped 'fit' and 'transform' events.

### Strict Mode Enforcement
*   **Action:** Pass unsorted time-series data with `strict=True`.
*   **Check:** `fit()` must raise `ValueError`. The model refuses to process non-defensible data.

## 4. Operational Validation Checklist
For any specific legal case, the following steps must be performed:

1.  [ ] **Pre-Run:** Run standard unit tests (`pytest`).
2.  [ ] **Data Audit:** Verify input data hash matches the chain of custody.
3.  [ ] **Sensitivity Analysis:** Run the chosen model with slightly perturbed parameters (e.g., changing `span` in Loess by ±10%). Result conclusions must remain stable.
4.  [ ] **Cross-Method Validation:** Compare results of the primary method (e.g., DoubleML) with a secondary method (e.g., GAM). Divergence requires explanation.

## 5. Automated Validation Suite
The repository includes an automated test suite (`tests/`) that implements Protocols A, B, and C.
To execute full validation:

```bash
python -m pytest tests/ --verbose
```
