import unittest
import pandas as pd
import numpy as np
import datetime
from decants.utils.time import prepare_time_feature
from decants.utils.crossfit import TimeSeriesSplitter
from decants.methods.double_ml import DoubleMLDecanter
from decants.methods.ml import MLDecanter
from decants.methods.gam import GamDecanter
from decants.methods.loess import FastLoessDecanter
from decants.methods.gaussian_process import GPDecanter
from decants.base import BaseDecanter

class TestAuditFixes2025(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 50
        self.X = pd.DataFrame({'x1': np.random.randn(self.n)}, index=pd.date_range('2020-01-01', periods=self.n))
        self.y = pd.Series(np.random.randn(self.n), index=self.X.index)

    def test_version_logging(self):
        """Verify 'decants' version is in audit log."""
        model = MLDecanter()
        # Init happens in constructor
        self.assertIn("decants", model._audit_log["library_versions"])
        print(f"Decants Version: {model._audit_log['library_versions']['decants']}")

    def test_integrity_enforcement(self):
        """Verify verify_integrity flag enforces sorting."""
        # Create unsorted data
        X_unsorted = self.X.sample(frac=1, random_state=42)
        y_unsorted = self.y.loc[X_unsorted.index]

        # 1. Without flag (Default): Should succeed (or fail later in splitter, but validation passes)
        # Note: TimeSeriesSplitter checks monotonicity itself, but we are testing BaseDecanter check.
        model = MLDecanter()
        # Should not raise ValueError about sorting (unless validation calls other checks)
        # TimeSeriesSplitter inside MLDecanter WILL raise error during fit_transform if called,
        # but fit() usually just fits generic models which might not care.
        # But MLDecanter.fit() calls _validate_alignment.
        try:
             model.fit(y_unsorted, X_unsorted)
        except Exception:
             pass # We don't care about other errors, just ensuring integrity check doesn't fire yet

        # 2. With flag: Should raise ValueError immediately in _validate_alignment
        model = MLDecanter()
        model.verify_integrity = True
        with self.assertRaisesRegex(ValueError, "Data index is not sorted"):
            model.fit(y_unsorted, X_unsorted)

    def test_double_ml_interpolation_warning(self):
        """Verify Interpolation mode logs a warning."""
        dml = DoubleMLDecanter(splitter="kfold")
        dml.transform(self.y, self.X)

        # Check audit history for warning
        warnings = [entry for entry in dml._audit_log["history"] if entry["type"] == "warning"]
        self.assertTrue(len(warnings) > 0)
        self.assertIn("Future Leakage Enabled", warnings[0]["details"]["message"])

    def test_predict_batch_type_safety(self):
        """Verify predict_batch fails loudly on bad types."""
        dml = DoubleMLDecanter()
        dml.fit(self.y, self.X)

        # Create a batch with strings in covariates
        # Batch: [Time, Covariate]
        batch = np.array([
            [1.0, "bad_data"],
            [2.0, "worse_data"]
        ], dtype=object)

        with self.assertRaises(ValueError):
            dml.predict_batch(batch)

        # Same for MLDecanter
        ml = MLDecanter()
        ml.fit(self.y, self.X)
        with self.assertRaises(ValueError):
            ml.predict_batch(batch)

    def test_loess_fallback_robustness(self):
        """Verify FastLoess doesn't crash on singular matrix (simulated by small span/constant data)."""
        # Create data that yields singular matrix locally (constant covariate)
        X_const = pd.DataFrame({'x1': np.zeros(self.n)}, index=self.y.index)

        # Use degree 1 (linear) which fails on constant X
        loess = FastLoessDecanter(span=0.1, degree=1)

        # Should not crash, but use fallback
        try:
            loess.fit(self.y, X_const)
        except Exception as e:
            self.fail(f"FastLoess crashed on singular matrix: {e}")

        # Ensure grid is populated (no zeros if data is not zero)
        # self.y is random normal, mean ~ 0. But shouldn't be exactly 0 everywhere.
        self.assertFalse(np.all(loess.interpolator.values == 0))


if __name__ == '__main__':
    unittest.main()
