# Validation Report: [Test Case Name]

**Date:** [YYYY-MM-DD]
**Tester:** [Name]
**Decants Version:** [x.y.z]
**Audit Hash:** [SHA-256 of Source]

## 1. Test Description
**What is being tested:**
[Brief description of the specific scenario, feature, or edge case. E.g., "Robustness of DoubleML Decanter to 90% Missing Covariate Data"]

**Category:**
*Select one:*
- [ ] Accuracy (Ground Truth Recovery)
- [ ] False Positive Control (Null Test)
- [ ] Stress Test / Edge Case
- [ ] Defensibility / Audit
- [ ] Leakage / Time-Travel

## 2. Rationale
**Why this test is important:**
[Explanation of why this validation is necessary for regulatory or legal confidence. E.g., "In legal contexts, opposing counsel may argue that missing data invalidates the trend estimate. We must prove the model degrades gracefully rather than hallucinating."]

## 3. Success Criteria
**Expected Outcome:**
[Specific, measurable criteria.]
- [ ] **Statistical:** [E.g., $R^2 > 0.9$, P-value > 0.05 for Null]
- [ ] **Behavioral:** [E.g., Warning logged for missing data]
- [ ] **Stability:** [E.g., Result varies by < 1% across 5 random seeds]
- [ ] **Integrity:** [E.g., Audit log captures the 'data_issue' flag]

## 4. Data Specification
**Characteristics:**
- **N (Samples):** [E.g., 120]
- **Signal-to-Noise Ratio:** [E.g., 2.0]
- **Trend Type:** [E.g., Linear, Structural Break]
- **Covariate Structure:** [E.g., Random Walk, Correlated with Trend]
- **Anomalies:** [E.g., 10% outliers added at random]

## 5. Validation Implementation

```python
import pandas as pd
import numpy as np
from decants.methods import DoubleMLDecanter
from decants.utils.diagnostics import check_orthogonality

# 1. Configuration
random_state = 42
n_samples = 120

# 2. Data Generation
# [Insert simplified generation code here]

# 3. Execution
decanter = DoubleMLDecanter(random_state=random_state)
result = decanter.fit_transform(df, 'target', covariates=['c1', 'c2'])

# 4. Verification
# [Insert assertion or check code here]
```

## 6. Results
**Console Output / Metrics:**
```text
[Paste output here, e.g.]
RMSE: 0.125
Effect Size: 2.1 (True: 2.0)
Warnings: [UserWarning: Input contains NaNs...]
```

## 7. Visual Evidence
**Time Series Decomposition:**
![Decomposition Plot](path/to/plot.png)
*[Caption: Grey=Original, Blue=Adjusted, Red=Covariate Effect. Note how the Red line tracks the synthetic covariate perfectly.]*

**Diagnostics:**
![Diagnostic Plot](path/to/diagnostic.png)
*[Caption: Residuals ACF plot showing no remaining structure.]*

## 8. Defensibility Check
- [ ] **Audit Log Present:** Yes/No
- [ ] **Source Hash Verified:** Yes/No
- [ ] **Data Hash Verified:** Yes/No

## 9. Conclusion
**Analysis:**
[Detailed interpretation. Did the model behave defensibly? If it failed, was the failure safe (e.g., explicit error) or unsafe (silent wrong number)?]

**Pass/Fail Status:**
- [ ] **PASS**
- [ ] **FAIL**
- [ ] **PASS with Caveats** (Explain below)

**Notes:**
[Any follow-up actions, such as "Update documentation to warn about this edge case".]
