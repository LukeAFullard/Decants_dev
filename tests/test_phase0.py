import unittest
import pandas as pd
import numpy as np
from decants.base import BaseDecanter
from decants.objects import DecantResult

class MockDecanter(BaseDecanter):
    def fit(self, y, X, **kwargs):
        self.fitted = True
        return self

    def transform(self, y, X):
        # Mock implementation: just subtract X from y (assuming single column X for simplicity)
        if isinstance(X, pd.DataFrame):
            effect = X.iloc[:, 0]
        else:
            effect = X

        adjusted = y - effect
        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=effect,
            model="MockModel",
            params={"mock_param": 1.0}
        )

class TestPhase0(unittest.TestCase):
    def test_imports(self):
        try:
            from decants.base import BaseDecanter
            from decants.objects import DecantResult
        except ImportError:
            self.fail("Could not import BaseDecanter or DecantResult")

    def test_mock_decanter(self):
        y = pd.Series([10, 20, 30], index=pd.date_range("2021-01-01", periods=3))
        X = pd.Series([1, 2, 3], index=y.index)

        decanter = MockDecanter()
        result = decanter.fit_transform(y, X)

        self.assertIsInstance(result, DecantResult)
        pd.testing.assert_series_equal(result.original_series, y)
        pd.testing.assert_series_equal(result.covariate_effect, X)
        expected_adjusted = y - X
        pd.testing.assert_series_equal(result.adjusted_series, expected_adjusted)
        self.assertEqual(result.model, "MockModel")
        self.assertEqual(result.params["mock_param"], 1.0)

if __name__ == '__main__':
    unittest.main()
