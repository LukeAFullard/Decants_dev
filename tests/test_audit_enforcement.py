
import pytest
import pandas as pd
import numpy as np
from decants.base import BaseDecanter
from decants.objects import DecantResult

class BadDecanter(BaseDecanter):
    def __init__(self):
        # INTENTIONALLY SKIPPING super().__init__()
        pass

    def fit(self, y, X, **kwargs):
        # BaseDecanter.fit does "pass", so subclasses override it.
        # But BaseDecanter now has `self._ensure_audit_log()` in the abstract `fit`?
        # Wait, if I override fit, I replace the base method.
        # So I must manually check or rely on `_log_event` inside fit.
        # But if I call super().fit(), it would check.
        # Most implementations don't call super().fit().
        # However, they usually call _log_event immediately.

        self._log_event("fit_start", {})
        return self

    def transform(self, y, X):
        return DecantResult(y, y, y, None)

def test_audit_enforcement():
    """
    Verify that failing to initialize the audit log raises a RuntimeError.
    """
    decanter = BadDecanter()
    y = pd.Series([1, 2, 3])
    X = pd.DataFrame({'a': [1, 2, 3]})

    with pytest.raises(RuntimeError, match="Audit log not initialized"):
        decanter.fit(y, X)
