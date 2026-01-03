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

**Category:**
*Select one:*
- [ ] Accuracy (Ground Truth Recovery)
- [ ] False Positive Control (Null Test)
- [ ] Stress Test / Edge Case
- [x] Defensibility / Audit
- [x] Leakage / Time-Travel

## 2. Rationale
**Why this test is important:**
- **Determinism:** A third party must be able to reproduce findings exactly.
- **Leakage:** In trading or causal inference, using future data to predict the past invalidates the result.
- **Audit:** We must prove *which* code and data produced a result years later.

## 3. Success Criteria
**Expected Outcome:**
- [x] **Determinism:** 100% Identical arrays.
- [x] **Audit:** JSON file contains `source_hash`, `y_hash`, `version`.
- [x] **Leakage:** Zero change for recursive models (DoubleML-TS, ML-TS). Change allowed for global smoothers (ARIMA, Prophet, GP).

## 4. Validation Implementation

```python
# See validation/protocol_D_defensibility/protocol_D_audit.py
def check_leakage():
    # 1. Predict at T=60
    # 2. Add outlier at T=100
    # 3. Re-fit and Predict at T=60
    # 4. Check diff
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

**Observations:**
*   **DoubleML** and **ML (RandomForest)** correctly isolate past from future when using `TimeSeriesSplit`. This makes them rigorous for causal claims ("What did we know at time T?").
*   **ARIMA** (and by extension Prophet/GP/Smoothers) fits parameters on the *entire* batch. Changing data at T=100 shifts the global parameters, slightly changing the fit at T=60. This is standard behavior for "Descriptive" or "Smoothing" models but implies they are not strictly "Causal" in a recursive sense unless refitted stepwise.
*   **Audit System** is functioning correctly, capturing cryptographic hashes of inputs and code.

## 6. Defensibility Check
- [x] **Audit Log Present:** Yes
- [x] **Source Hash Verified:** Yes
- [x] **Data Hash Verified:** Yes

## 7. Conclusion
**Analysis:**
The library meets the Defensibility requirements.
- **Reproducibility** is confirmed.
- **Audit Trails** are robust.
- **Strict Causality** is available via DoubleML/ML in 'timeseries' mode. Users needing strict non-leakage must choose these methods. Users doing retrospective analysis can use ARIMA/Prophet but must acknowledge full-sample fitting.

**Pass/Fail Status:**
- [x] **PASS**
- [ ] **FAIL**
- [ ] **PASS with Caveats**

**Notes:**
- Update docs to clarify that ARIMA/Prophet/GP are global fitters (Full Sample), while DoubleML/ML can be Strict Recursive.
