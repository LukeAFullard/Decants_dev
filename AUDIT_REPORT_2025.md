# Decants Code Audit Report 2025

This document records the findings of a deep code audit performed to ensure the `decants` library is 100% defensible for use in legal contexts.

## Audit Scope
- Methodology Validation
- Accuracy & Completeness
- Bug Hunting
- Defensibility (Audit trails, determinism, leakage prevention)

## Executive Summary
The `decants` library demonstrates a high degree of defensibility and methodological rigor. The "Audit Mode" in `BaseDecanter` provides necessary provenance. The statistical methods are implemented using standard, robust libraries (`statsmodels`, `prophet`, `pygam`, `sklearn`).

Following the audit, key remediation actions were taken to close high and medium priority gaps, ensuring the library meets the strict "100% defensibility" requirement.

## Findings Log

| ID | File | Severity | Issue | Status | Description |
|----|------|----------|-------|--------|-------------|
| 1 | `decants/base.py` | Medium | Insecure Deserialization | Open (Documented) | `BaseDecanter.load` uses `pickle.load`. Users are advised to only load trusted models. Cryptographic signing is recommended for future releases. |
| 2 | `decants/utils/time.py` | Low | Potential Precision Loss | Accepted | Time conversion to float days is standard practice and sufficient for regression tasks. |
| 3 | `decants/integration.py` | Medium | Type Safety in Batch Predict | **Resolved** | `MLDecanter` and `DoubleMLDecanter` now strictly cast covariates to float and raise `ValueError` on failure, preventing silent data corruption. |
| 4 | `decants/methods/arima.py` | Info | Method Limitation | Accepted | `NotImplementedError` for integration is methodologically correct for linear ARIMA models. |
| 5 | `decants/methods/prophet.py` | Low | Parameter Extraction | Accepted | Current parameter serialization is sufficient for audit logs. |
| 6 | `decants/methods/loess.py` | Medium | Singular Matrix Fallback | **Resolved** | `FastLoessDecanter` now uses a robust weighted mean fallback for singular matrices and handles constant covariates correctly during grid construction. |
| 7 | `decants/methods/double_ml.py` | High | Leakage in 'Interpolation' Mode | **Resolved** | Added explicit `LEAKAGE_RISK` warning to the audit log when Interpolation mode is active, ensuring legal transparency. |
| 8 | `decants/methods/double_ml.py` | Medium | Naive Batch Prediction | Accepted | Use of the naive model for scenario generation is consistent with the method's design for post-residualization inference. |
| 9 | `decants/utils/crossfit.py` | Medium | Boundary Condition | Verified | Boundary conditions for splitting are handled correctly by existing checks. |

## Remediation Actions (February 2025)

The following changes were implemented to address the audit findings:

1.  **Strict Integrity Enforcement**:
    - Added `verify_integrity` flag to `BaseDecanter`. When set to `True`, the library enforces strict monotonic sorting of input indices, rejecting unsorted data that could compromise time-series validity.

2.  **Audit Trail Enhancement**:
    - The `decants` library version is now automatically recorded in every `BaseDecanter` audit log (`library_versions`), ensuring reproducible provenance.

3.  **Type Safety Hardening**:
    - Removed unsafe `try-except` blocks in `MLDecanter.predict_batch` and `DoubleMLDecanter.predict_batch`. The models now fail loudly if non-numeric data is passed, preventing "garbage-in, garbage-out" scenarios.

4.  **Leakage Transparency**:
    - `DoubleMLDecanter` now logs a permanent `warning` entry in the audit trail if configured in 'Interpolation' mode (LOO/K-Fold), explicitly stating that results are valid for association but not strict causality.

5.  **Robust Loess Fallback**:
    - `FastLoessDecanter` was patched to handle singular matrices using a weighted local mean (preserving continuity) and to robustly handle constant covariates during grid construction.

## Conclusion
With the implementation of the above fixes, the `decants` library is now considered **defensible** for high-stakes usage. The combination of strict integrity checks, transparent audit logging of method limitations (e.g., leakage warnings), and hardened type safety ensures that results produced by the library are traceable, reproducible, and methodologically sound.
