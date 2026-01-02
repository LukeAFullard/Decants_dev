# Decants Code Audit Report

**Date:** 2025-02-17
**Auditor:** Jules
**Scope:** Full Repository Audit (`decants/`)
**Objective:** Validate methodology, accuracy, completeness, and defensibility for legal use.

## Executive Summary

The `decants` library implements a robust set of methodologies for time-series covariate adjustment ("decanting"). The codebase prioritizes **defensibility** through strict audit logging, prevention of data leakage (especially future leakage), and the use of established statistical and machine learning methods.

**Overall Status:** **Pass / Defensible**
The methodologies implemented (GAM, Prophet, DoubleML, ARIMA, GP, LOESS) are standard and correctly applied. The core infrastructure ensures reproducibility via provenance hashing and sidecar JSON logs.

## Methodology Validation

| Method | Implementation | Defensibility | Comments |
| :--- | :--- | :--- | :--- |
| **GAM** | `pygam` (Splines) | **High** | Correctly isolates trend vs covariates using semi-parametric smoothing. Uses `partial_dependence` for effect isolation. |
| **ARIMA** | `statsmodels` (SARIMAX) | **High** | Uses standard state-space modeling. Correctly disables "Integration" (Monte Carlo) as it is linear. Uses `cov_params` for uncertainty. |
| **Prophet** | `prophet` (Bayesian) | **High** | Correctly handles seasonality/holidays. Warns on non-datetime indices. |
| **DoubleML** | `sklearn` (Cross-Fitting) | **High** | **Critical:** Strictly enforces monotonic time indices and uses `TimeSeriesSplit` to prevent future leakage. This is crucial for causal claims. Naive inference mode is available but clearly distinguished. |
| **Gaussian Process** | `sklearn` (Kriging) | **Medium-High** | Uses a valid "Decomposition Trick" (Total Prediction - Trend Prediction). Relies on kernel assumptions (Matern), which are standard. Computationally heavy but mathematically sound. |
| **Fast LOESS** | `sklearn` (Neighbors + LinearReg) | **High** | Implements WRTDS (Weighted Regressions on Time, Discharge, Seasonality) style logic. Grid-based approximation is efficient and locally valid. |

## detailed File-by-File Audit

### Core Infrastructure

*   **`decants/base.py`**:
    *   **Audit Mode**: Implemented correctly. Every `fit`/`transform` logs inputs (SHA256 hashes), library versions, and parameters. This creates a chain of custody for results.
    *   **Persistence**: Uses `pickle`.
        *   *Observation*: `pickle` is not secure for untrusted data. A warning is present in the code.
        *   *Mitigation*: The sidecar `.audit.json` and `.params.json` files allow inspection of model metadata without unpickling the binary, which is excellent for auditing.
    *   **Alignment**: `_validate_alignment` ensures intersection of indices is not empty.

*   **`decants/integration.py` (`MarginalizationMixin`)**:
    *   **Logic**: Implements Monte Carlo Marginalization ($E[Y|C]$).
    *   **Correctness**: Correctly samples from historical covariate pool.
    *   **Type Safety**: Handles mixed inputs (Timestamp + Float) by creating object arrays. Verified via test.

### Utilities

*   **`decants/utils/crossfit.py`**:
    *   **Safety**: `TimeSeriesSplitter` raises `ValueError` if input is unsorted. This prevents accidental shuffling which would invalidate causal tests.
*   **`decants/utils/diagnostics.py`**:
    *   **Metrics**: Variance Reduction and Orthogonality checks are implemented correctly.
*   **`decants/utils/time.py`**:
    *   **Data Integrity**: Raises `ValueError` for NaT/NaN in time indices.

### Specific Method Findings

*   **`DoubleMLDecanter`**:
    *   Handles split failures gracefully (returns NaNs with warning).
    *   *Correction*: The warning was printed to stdout. It has been updated to also log to the persistent audit trail (`self._log_event`).
*   **`FastLoessDecanter`**:
    *   Correctly restricts to 1 covariate (dimensionality curse mitigation).
    *   Falls back to local mean if linear regression is singular.

## Recommendations & Fixes Applied

1.  **Audit Trail Enhancement**: `DoubleMLDecanter` now explicitly logs split failures to the JSON audit trail, not just stdout.
2.  **Robustness**: Verified `np.hstack` behavior for mixed types in `integration.py` and added comments.

## Conclusion

The code is high-quality and defensible. The explicit "Audit Mode" and strict leakage prevention in `DoubleML` make it suitable for rigorous analysis where the provenance of the adjustment must be proven.
