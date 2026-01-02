# Decants Code Audit Report (Current)

**Date:** February 2025
**Auditor:** Jules (AI Agent)
**Scope:** Full Codebase (`decants/`)
**Objective:** Verify 100% Defensibility for Legal/Forensic Usage.

## Executive Summary
The `decants` library has been audited file-by-file. The codebase implements a robust, defensible framework for covariate adjustment. Key mechanisms for defensibility (Audit Logging, Strict Mode, Time-Series Integrity) are present and functional. The methodology for sophisticated models (DoubleML, GP, Loess) is mathematically sound and handles edge cases (leakage, singular matrices) appropriately.

## Detailed Findings

### 1. Core Framework (`decants/base.py`, `decants/integration.py`)
*   **Audit Logging:** `BaseDecanter` enforces initialization of the audit log. It captures source code hash (`_compute_source_hash`), library versions, and strict mode status. This provides a cryptographic chain of custody for results.
*   **Strict Mode:** The `strict` parameter correctly propagates to `verify_integrity`, which enforces monotonic sorting of indices. This is critical for preventing future-leakage in time-series analysis.
*   **MarginalizationMixin:** The Monte Carlo integration logic correctly handles mixed input types (timestamps + floats) using object arrays and explicit casting. The use of `predict_batch` ensures efficiency and type safety.
*   **Input Validation:** `_validate_alignment` ensures intersection is not empty and warns if indices are not Datetime-like (though allows numeric for flexibility).

### 2. Utilities (`decants/utils/`)
*   **Cross-Fitting (`crossfit.py`):** `TimeSeriesSplitter` includes a guard against unsorted data (raises `ValueError`), effectively preventing accidental leakage. Boundary conditions (insufficient data) are handled with explicit errors.
*   **Time Handling (`time.py`):** `prepare_time_feature` raises `ValueError` for NaT or NaN values, preventing silent propagation of missing time data. Conversion to float days is standard and precise enough for the domain.
*   **Diagnostics:** Variance reduction and orthogonality checks correctly handle alignment and NaNs.

### 3. Method Implementations

| Method | Status | Defensibility Notes |
| :--- | :--- | :--- |
| **ARIMA** | **Pass** | Correctly blocks `transform_integrated` (not applicable). Variance calculation for CIs includes covariance term. |
| **GAM** | **Pass** | `predict_batch` handles mixed types correctly. Gridsearch is optional and logged. |
| **Prophet** | **Pass** | Warns on non-datetime index. Reconstructs DataFrame correctly in `predict_batch`. |
| **DoubleML** | **Pass** | **Critical:** Warns `LEAKAGE_RISK` if used in Interpolation mode. Handles split failures gracefully (returns NaNs with warning). Enforces sorting before splitting. |
| **GaussianProcess** | **Pass** | Uses "Decomposition Trick" (Diff of Predict vs Counterfactual). RobustScaler handles outliers. |
| **MLDecanter** | **Pass** | Uses `TimeSeriesSplit` for `fit_transform`. Leaves initial training window as NaN (correct). |
| **FastLoess** | **Pass** | Implements robust fallback (Weighted Mean) for singular matrices. Handles constant covariates via grid padding. |

### 4. Defensibility Assessment
The library meets the "100% Defensible" standard provided that:
1.  Users initialize models with `strict=True`.
2.  Users inspect the `DecantResult.stats` and `audit_log` for warnings (especially regarding leakage or split failures).

## Recommendations (Minor)
*   **Security:** `BaseDecanter.load` uses `pickle`. This is documented as insecure. For a high-stakes legal context, loading models from untrusted sources must be strictly prohibited.
*   **Type Safety:** While `predict_batch` methods cast to float, ensures that `ValueError` is raised if this fails (e.g. string data passed as covariate). This was verified in the code.

## Conclusion
No critical methodology errors or bugs were found. The remediations from the 2025 Audit appear to be correctly implemented and functional.
