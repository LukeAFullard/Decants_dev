# Decants Code Audit Report 2025

This document records the findings of a deep code audit performed to ensure the `decants` library is 100% defensible for use in legal contexts.

## Audit Scope
- Methodology Validation
- Accuracy & Completeness
- Bug Hunting
- Defensibility (Audit trails, determinism, leakage prevention)

## Executive Summary
The `decants` library demonstrates a high degree of defensibility and methodological rigor. The "Audit Mode" in `BaseDecanter` provides necessary provenance. The statistical methods are implemented using standard, robust libraries (`statsmodels`, `prophet`, `pygam`, `sklearn`).

However, several issues were identified that could affect "100% defensibility" in edge cases or if not used carefully.

## Findings Log

| ID | File | Severity | Issue | Description |
|----|------|----------|-------|-------------|
| 1 | `decants/base.py` | Medium | Insecure Deserialization | `BaseDecanter.load` uses `pickle.load` without cryptographic signature verification. While documented, this is a security risk if models are shared. |
| 2 | `decants/utils/time.py` | Low | Potential Precision Loss | `prepare_time_feature` converts time to float days. For extremely long durations or nanosecond precision, float64 precision might degrade slightly, though unlikely to affect regression. |
| 3 | `decants/integration.py` | Medium | Type Safety in Batch Predict | `MarginalizationMixin` creates object arrays for mixed inputs. While `predict_batch` methods handle this, implicit type casting inside `predict_batch` (e.g. in `MLDecanter` and `DoubleMLDecanter`) ignores ValueError, potentially masking bad data. |
| 4 | `decants/methods/arima.py` | Info | Method Limitation | `ArimaDecanter` explicitly raises `NotImplementedError` for `transform_integrated`. This is methodologically correct (ARIMAX is linear) but limits the unified API usage. |
| 5 | `decants/methods/prophet.py` | Low | Parameter Extraction | `get_model_params` relies on `self.model.params` which might contain numpy arrays that need custom serialization handling (handled in `base.save` but fragile). |
| 6 | `decants/methods/loess.py` | Medium | Singular Matrix Fallback | `FastLoessDecanter` falls back to `np.mean(local_y)` if local regression fails (singular matrix). This creates a discontinuity in the surface. Defensible but potentially visible artifact. |
| 7 | `decants/methods/double_ml.py` | High | Leakage in 'Interpolation' Mode | `DoubleMLDecanter` allows 'interpolation' (LOO/K-Fold) mode. While documented, users might inadvertently use this for time-series where strict leakage prevention is legally required. The "Interpolation" mode is *not* defensible for causality in time-series, only for correlation/smoothing. |
| 8 | `decants/methods/double_ml.py` | Medium | Naive Batch Prediction | `DoubleMLDecanter.predict_batch` uses the naive model fitted on the full dataset, not the cross-fitted models. This is standard for inference but loses the "Double ML" robustness properties for the counterfactual generation step. |
| 9 | `decants/utils/crossfit.py` | Medium | Boundary Condition | `TimeSeriesSplitter` calculates `n_test_points = n_samples - min_train_size`. If `n_test_points < n_splits`, it yields a single split or behaves unexpectedly. (Tested and seems handled, but fragile). |

## Recommendations

1.  **Harden `predict_batch` Type Safety**:
    In `decants/methods/ml.py` and `decants/methods/double_ml.py`, remove the `try-except ValueError: pass` block when casting covariates to float. If non-numeric covariates are passed to a numeric model, it *should* fail loudly rather than silently passing potentially bad data.

2.  **Explicit Warning for Interpolation Mode**:
    In `DoubleMLDecanter`, when `mode='interpolation'` (or `allow_future=True`) is used, add a dedicated audit log entry with a warning flag: `"WARNING: Future Leakage Enabled - Results Valid for Association, Not Strict Causality"`.

3.  **Enhance FastLoess Fallback**:
    Instead of `np.mean(local_y)`, `FastLoessDecanter` could use a weighted mean or the value of the nearest neighbor to preserve continuity better, or simply raise an error if strictness is preferred.

4.  **Audit Trail Versioning**:
    Ensure `DecantResult` includes the version of `decants` used to generate it.

5.  **Data Sorting Enforcement**:
    `BaseDecanter.fit` and `transform` should enforce `sort_index()` on inputs if `verify_integrity=True` flag is added, to prevent user error with unsorted time series (which breaks `TimeSeriesSplitter`). (Note: `DoubleMLDecanter` already does this).

## Conclusion
The codebase is solid. The "defensibility" requirement is largely met by the `BaseDecanter` audit logging and the strict `TimeSeriesSplitter`. The identified issues are primarily edge cases or known methodological trade-offs (like LOO for small data) that are handled explicitly.
