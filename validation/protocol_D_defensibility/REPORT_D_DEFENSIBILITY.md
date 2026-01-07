# Validation Report: Protocol D (Defensibility & Audit)

**Date:** 2025-02-12
**Tester:** Jules
**Decants Version:** 0.1.0
**Audit Hash:** [See Logs]

## 1. Test Description
**What is being tested:**
This protocol validates the **Defensibility** features of the library, crucial for legal/regulatory use.
We test:
1.  **Determinism:** Does `random_state` guarantee identical bit-for-bit results?
2.  **Leakage (Time Travel):** Does a future data change affect past predictions? (Strict Causal Check).
3.  **Audit Integrity:** Is the audit sidecar generated with correct source code hashes and data hashes?
4.  **Marginalization (Integration) vs Forensic (Subtraction):** Verifying that "Strategic Mode" (Integration) produces methodologically distinct results from "Forensic Mode" (Subtraction) when interactions exist.

**Category:**
*Select one:*
- [ ] Accuracy (Ground Truth Recovery)
- [ ] False Positive Control (Null Test)
- [ ] Stress Test / Edge Case
- [x] Defensibility / Audit
- [x] Leakage / Time-Travel
- [x] Methodology Validation

## 2. Rationale
**Why this test is important:**
- **Determinism:** A third party must be able to reproduce findings exactly.
- **Leakage:** In trading or causal inference, using future data to predict the past invalidates the result.
- **Audit:** We must prove *which* code and data produced a result years later.
- **Marginalization:** Users need to know when to use `transform_integrated` (Counterfactual/Strategic) vs standard `fit_transform` (Forensic/Anomaly). We must demonstrate they are not identical.

## 3. Success Criteria
**Expected Outcome:**
- [x] **Determinism:** 100% Identical arrays.
- [x] **Audit:** JSON file contains `source_hash`, `y_hash`, `version`.
- [x] **Leakage:** Zero change for recursive models (DoubleML-TS, ML-TS). Change allowed for global smoothers (ARIMA, Prophet, GP).
- [x] **Marginalization:** Significant difference between Forensic and Strategic results in scenarios with Covariate-Time interaction.

## 4. Validation Implementation

```python
# See validation/protocol_D_defensibility/protocol_D_audit.py
def check_leakage():
    # 1. Predict at T=60
    # 2. Add outlier at T=100
    # 3. Re-fit and Predict at T=60
    # 4. Check diff

def check_marginalization_effect():
    # 1. Generate Data with Time-Covariate Interaction
    # 2. Run fit_transform (Forensic)
    # 3. Run transform_integrated (Strategic)
    # 4. Verify Difference > Threshold
```

## 5. Results

**Summary:**

| Test | Model | Status | Notes |
|:---|:---|:---|:---|
| **Determinism** | DoubleML | **PASS** | Identical output. |
| | ML (RF) | **PASS** | Identical output. |
| **Leakage** | DoubleML (TS) | **PASS** | Diff = 0.0. No future leakage. |
| | ML (RF) | **PASS** | Diff = 0.0. No future leakage. |
| | ARIMA | **FAIL** (Expected) | Diff > 0. Parameters influenced by future. |
| **Audit** | BaseDecanter | **PASS** | Hashes/Metadata correct. |
| **Marginalization** | DoubleML | **DIFFERENT** | MAE Diff ~0.9. Integration captures interaction average. |
| | ML (RF) | **DIFFERENT** | MAE Diff ~1.3. |
| | Prophet | **DIFFERENT** | MAE Diff ~3.2. |
| | GAM | **ERROR** | Known issue: internal `time_trend` missing in external pool. |

### Marginalization vs Forensic Comparison
The following plot demonstrates the divergence between **Forensic Mode** (Blue, Solid) and **Strategic Mode** (Red, Dotted) in a scenario where the covariate effect changes over time (Interaction).
- **Forensic Mode:** Subtracts the specific effect at time $t$ given $C_t$.
- **Strategic Mode:** Estimates the trend by averaging over the historical distribution of covariates ("What if covariates were 'normal'?").

![Marginalization Comparison](marginalization_comparison.png)

**Observations:**
*   **DoubleML** and **ML (RandomForest)** correctly isolate past from future when using `TimeSeriesSplit`. This makes them rigorous for causal claims ("What did we know at time T?").
*   **ARIMA** (and by extension Prophet/GP/Smoothers) fits parameters on the *entire* batch. Changing data at T=100 shifts the global parameters, slightly changing the fit at T=60. This is standard behavior for "Descriptive" or "Smoothing" models but implies they are not strictly "Causal" in a recursive sense unless refitted stepwise.
*   **Audit System** is functioning correctly, capturing cryptographic hashes of inputs and code.
*   **Marginalization Utility:** The divergence confirms that `transform_integrated` provides unique value. In scenarios with changing covariate relationships (Regime Shifts), **Strategic Mode** gives a more stable "Trend" baseline than simple subtraction.

## 6. Defensibility Check
- [x] **Audit Log Present:** Yes
- [x] **Source Hash Verified:** Yes
- [x] **Data Hash Verified:** Yes
- [x] **Methodology Distinction Verified:** Yes

## 7. Conclusion
**Analysis:**
The library meets the Defensibility requirements.
- **Reproducibility** is confirmed.
- **Audit Trails** are robust.
- **Strict Causality** is available via DoubleML/ML in 'timeseries' mode.
- **Methodological Depth** is confirmed: The library offers distinct tools for Forensic (Anomaly) vs Strategic (Trend) analysis.

**Pass/Fail Status:**
- [x] **PASS**
- [ ] **FAIL**
- [ ] **PASS with Caveats**

**Notes:**
- Update docs to clarify that ARIMA/Prophet/GP are global fitters (Full Sample), while DoubleML/ML can be Strict Recursive.
