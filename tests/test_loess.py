import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from decants.methods.loess import FastLoessDecanter

class TestFastLoessDecanter(unittest.TestCase):
    def setUp(self):
        # 1. Generate Complex Data (Regime Shift)
        np.random.seed(99)
        t = np.linspace(0, 100, 200) # Time
        self.t_values = t
        # Covariate: Random fluctuation
        self.C_values = np.random.normal(0, 1, 200)

        # True Signal: A Sine Wave
        true_signal = np.sin(t / 10) * 3
        self.true_signal = true_signal

        # The "Problematic" Covariate Relationship:
        # Early in time (t<50), Covariate adds value.
        # Late in time (t>50), Covariate subtracts value.
        relationship = np.where(t < 50, 2 * self.C_values, -2 * self.C_values)
        self.relationship = relationship

        y_values = true_signal + relationship + np.random.normal(0, 0.5, 200)

        # Create pandas objects
        # We'll use numeric index for simplicity as supported by BaseDecanter/FastLoessDecanter
        # (Though we should also test datetime)
        self.y = pd.Series(y_values, index=t, name="target")
        self.X = pd.DataFrame({"covariate": self.C_values}, index=t)

    def test_fit_transform(self):
        decanter = FastLoessDecanter(span=0.2, grid_resolution=30)
        decanter.fit(self.y, self.X)

        self.assertIsNotNone(decanter.interpolator)

        result = decanter.transform(self.y, self.X)

        # Check result shape
        self.assertEqual(len(result.adjusted_series), 200)
        self.assertEqual(len(result.covariate_effect), 200)

        # Check performance
        # We expect adjusted series to be closer to true_signal
        # Compute RMSE
        rmse_original = np.sqrt(np.mean((self.y.values - self.true_signal)**2))
        rmse_adjusted = np.sqrt(np.mean((result.adjusted_series.values - self.true_signal)**2))

        self.assertLess(rmse_adjusted, rmse_original, "Adjusted series should reduce error vs true signal")

        # Check effect capture
        # The true effect is `relationship`.
        # The estimated effect should correlate with it.
        corr = np.corrcoef(result.covariate_effect.values, self.relationship)[0, 1]
        self.assertGreater(corr, 0.8, "Estimated effect should correlate with true non-stationary effect")

    def test_save_load(self):
        decanter = FastLoessDecanter(span=0.2, grid_resolution=20)
        decanter.fit(self.y, self.X)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "test_loess_model.pkl")
            decanter.save(filepath)

            loaded = FastLoessDecanter.load(filepath)
            self.assertIsNotNone(loaded.interpolator)

            result = loaded.transform(self.y, self.X)
            self.assertEqual(len(result.adjusted_series), 200)

    def test_datetime_index(self):
        # Test with Datetime Index
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        y = pd.Series(np.random.randn(100), index=dates)
        X = pd.DataFrame({'c': np.random.randn(100)}, index=dates)

        decanter = FastLoessDecanter(span=0.3, grid_resolution=10)
        decanter.fit(y, X)
        result = decanter.transform(y, X)

        self.assertEqual(len(result.adjusted_series), 100)

if __name__ == '__main__':
    unittest.main()
