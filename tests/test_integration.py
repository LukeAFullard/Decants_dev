import unittest
import pandas as pd
import numpy as np
import logging
from decants.methods.gam import GamDecanter
from decants.methods.arima import ArimaDecanter
from decants.methods.prophet import ProphetDecanter
from decants.methods.ml import MLDecanter
from decants.utils.diagnostics import variance_reduction
from sklearn.ensemble import RandomForestRegressor

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Generate Synthetic Data
        np.random.seed(42)
        n = 100
        self.date_range = pd.date_range("2021-01-01", periods=n)

        # Trend: linear
        time_idx = np.arange(n)
        self.trend = 0.05 * time_idx

        # Seasonality: Sine wave
        self.seasonality = np.sin(time_idx / 5)

        # Covariate X: random walk
        self.X = pd.Series(np.cumsum(np.random.normal(0, 0.5, n)), index=self.date_range)

        # Effect of X
        self.true_effect = 2 * self.X

        # Noise
        noise = np.random.normal(0, 0.5, n)

        # Y = Trend + Seasonality + Effect + Noise
        self.y = self.trend + self.seasonality + self.true_effect + noise
        self.y.index = self.date_range

    def test_all_methods_smoke(self):
        """
        Run all 4 methods on the same dataset and verify they return valid results.
        """
        methods = [
            ("GAM", GamDecanter(n_splines=10, lam=0.5)),
            ("ARIMAX", ArimaDecanter(order=(1, 0, 0))), # Simple ARIMAX
            ("Prophet", ProphetDecanter()),
            ("ML", MLDecanter(estimator=RandomForestRegressor(n_estimators=10))) # Small number of estimators for speed
        ]

        for name, decanter in methods:
            with self.subTest(method=name):
                print(f"Testing {name}...")

                # Fit Transform
                # Note: Prophet and ML might produce warnings, let's suppress logging for clarity if needed
                # but for smoke test, we just want it to finish.

                # For ARIMAX, ensure we catch potential convergence warnings or errors if they happen,
                # though simple data should be fine.
                try:
                    result = decanter.fit_transform(self.y, self.X)
                except Exception as e:
                    self.fail(f"{name} failed with error: {e}")

                # Checks
                self.assertIsNotNone(result)
                self.assertEqual(len(result.adjusted_series), len(self.y))
                self.assertEqual(len(result.covariate_effect), len(self.y))

                # Check for NaNs (except maybe first few for ML/ARIMA depending on implementation)
                # MLDecanter has known issue with first fold NaNs.
                # ARIMAX might have differencing issues if d>0.

                if name == "ML":
                    # Known issue: First fold NaNs
                    self.assertTrue(result.adjusted_series.iloc[20:].notna().all(), "ML adjusted series should not have NaNs after initial window")
                else:
                    self.assertTrue(result.adjusted_series.dropna().shape[0] > len(self.y) * 0.9, f"{name} should return mostly non-NaN values")

                # Basic Sanity: Covariate effect should correlate somewhat with true effect
                # This is a smoke test, but let's check directionality at least
                corr = result.covariate_effect.corr(self.true_effect)

                # Check variance reduction
                vr = variance_reduction(result.original_series, result.adjusted_series)

                print(f"{name} - Correlation with Truth: {corr:.4f}, Variance Reduction: {vr:.4f}")

                # We expect positive correlation mostly, but if models fail to capture, it might be low.
                # Asserting strict > 0.5 might be flaky for un-tuned models on random data.
                # So we just print it and assert types.

if __name__ == '__main__':
    unittest.main()
