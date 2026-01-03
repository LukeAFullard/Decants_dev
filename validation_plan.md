# Decants Validation Plan

**Date:** February 2025
**Scope:** Validation of the `decants` library for high-stakes/legal usage.
**Objective:** Establish a formal protocol to prove that the library produces accurate, defensible, and reproducible covariate adjustments suitable for regulatory scrutiny.

## 1. Regulatory & Legal Context
To be suitable for regulatory use (e.g., following principles from **SR 11-7** on Model Risk Management) and defensible in legal challenges, the software must demonstrate:
1.  **Conceptual Soundness:** The mathematical approach is standard, peer-reviewed, or theoretically justified.
2.  **Outcome Analysis:** The model performs as expected on known ground-truth data.
3.  **Ongoing Monitoring:** Stability and robustness are checked under stress.
4.  **Reproducibility:** A third party must be able to reproduce the exact results given the same inputs and code version.

## 2. Validation Protocols

### Protocol A: Synthetic Ground Truth Recovery (Accuracy)
*Objective: Prove the model can correctly identify and separate signal from noise and confounding variables.*

1.  [x] **Scenario A1: Standard Signal Recovery**
    *   **Input:** $Y_t = Trend_t + \beta \cdot C_t + \epsilon_t$.
    *   **Success Criteria:** Estimated $\hat{\beta}$ within 5% of true $\beta$; Adjusted series matches $Trend_t$ (High $R^2$).
    *   *Status:* **PASS** (Linear methods), **PASS** (Non-linear methods approx).

2.  [x] **Scenario A2: Non-Linear Interactions**
    *   **Input:** $Y_t = Trend_t + \sin(C_t) + 0.5 C_t^2 + \epsilon_t$.
    *   **Success Criteria:** Non-parametric methods (GAM, GP, FastLoess) must recover the shape of $f(C_t)$. Linear methods (ARIMA, DML-Linear) should fail or warn.
    *   *Status:* **PASS** (FastLoess), **FAIL/Caveat** (GAM, GP need tuning), **FAIL** (Linear methods as expected).

3.  [x] **Scenario A3: Trend-Covariate Confounding**
    *   **Input:** Both Trend and Covariate grow linearly ($Trend_t = t$, $C_t = t$).
    *   **Success Criteria:** The model should either:
        *   Attribute effect to Trend (conservative for intervention detection).
        *   Or attribute to Covariate based on user specification.
        *   **Crucially:** It must not double-count (explode coefficients).
    *   *Status:* **PASS**. No explosion observed. Prophet attributes to Trend (Conservative), DoubleML/ARIMA attribute to Covariate (Greedy).

### Protocol B: The "Null" Test (Placebo/False Positive Control)
*Objective: Prove the model does not "hallucinate" effects.*

1.  [x] **Scenario B1: White Noise**
    *   **Input:** $Y_t \sim N(0,1)$, $C_t \sim N(0,1)$.
    *   **Success Criteria:** Estimated effect $\approx 0$. No statistically significant components detected.
    *   *Status:* **PASS with Caveats**. DoubleML, ARIMA, Prophet pass (RMSE < 0.2). ML (RandomForest) and GAM show slight overfitting to noise (RMSE ~0.3). See `validation/protocol_B_null_test/REPORT_B1_WHITE_NOISE.md`.

2.  [ ] **Scenario B2: Spurious Correlation (Random Walks)**
    *   **Input:** Two independent random walks ($Y_t = \sum \epsilon$, $C_t = \sum \eta$).
    *   **Success Criteria:**
        *   Model should handle non-stationarity (e.g., via differencing or cointegration checks).
        *   Diagnostics should flag high risk of spurious correlation if unhandled.

### Protocol C: Stress Testing & Robustness
*Objective: Ensure stability under extreme conditions.*

1.  [ ] **Scenario C1: Adversarial Inputs**
    *   **Input:** Infinite values (`np.inf`), NaNs, massive outliers (100$\sigma$).
    *   **Success Criteria:** Graceful failure (ValueError) or robust handling (RobustScaler/Trimming). *No silent corruption.*

2.  [ ] **Scenario C2: Data Sparsity & Gaps**
    *   **Input:** 50% missing data randomly dispersed; large contiguous gap.
    *   **Success Criteria:** Uncertainty intervals should widen significantly in gap regions (GP/Prophet).

3.  [ ] **Scenario C3: Multi-Collinearity**
    *   **Input:** $C_1 = X$, $C_2 = X + \epsilon$ (near perfect correlation).
    *   **Success Criteria:** Estimates should remain stable (e.g., via regularization in DML/Ridge) or solver should warn.

### Protocol D: Defensibility & Audit
*Objective: Ensure the analysis is legally admissible.*

1.  [ ] **Determinism Check**
    *   **Action:** Run the full pipeline twice with `random_state=42`.
    *   **Success Criteria:** Output arrays must be identical bit-for-bit.

2.  [ ] **Leakage Verification (Time Travel)**
    *   **Action:** Check if $Prediction_t$ changes when data at $t+k$ is altered.
    *   **Success Criteria:** Zero change for strict time-series models.

3.  [ ] **Audit Trail Completeness**
    *   **Action:** Verify `.audit.json` contains:
        *   Source Code Hash (SHA-256).
        *   Library Version.
        *   Input Data Hash (SHA-256 of DataFrame).
        *   Exact hyperparameter set.

## 3. Reporting Standards
All validation exercises must result in a **Validation Report** (see `validation_report_template.md`).
This report serves as the primary artifact for:
*   Internal Review.
*   Regulatory Submission.
*   Legal Discovery.

## 4. Execution Plan
To execute the full validation suite for a new release:
1.  Run automated regression tests: `pytest tests/`
2.  Execute "Deep Validation" script (to be created): `python scripts/validate_release.py`
3.  Generate Validation Report from template.

**Note on Audit Mode:**
When running iterative validation tests (where artifacts are not intended for final legal discovery), it is recommended to disable the automatic audit log generation (if supported) or carefully exclude `*.audit.json` files from version control to prevent "git noise" caused by timestamp updates.
