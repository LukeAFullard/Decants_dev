import unittest
import numpy as np
import pandas as pd
from decants.methods.loess import FastLoessDecanter
from decants.methods.gam import GamDecanter
from decants.methods.arima import ArimaDecanter
import warnings
import datetime

class TestMarginalization(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        # Nonlinear relationship: y = t + C^2
        np.random.seed(42)
        self.n_samples = 100
        self.t_dates = pd.date_range(start="2020-01-01", periods=self.n_samples, freq="D")
        self.t_numeric = np.arange(self.n_samples)

        # C is random
        self.C = np.random.uniform(-10, 10, size=self.n_samples)

        # y depends on t and C^2 (nonlinear)
        self.trend = self.t_numeric * 0.1
        self.effect = self.C**2
        self.y = self.trend + self.effect + np.random.normal(0, 0.1, self.n_samples)

        self.y_series = pd.Series(self.y, index=self.t_dates)
        self.C_series = pd.Series(self.C, index=self.t_dates, name="covariate")

    def test_loess_integration(self):
        # Test FastLoessDecanter integration
        model = FastLoessDecanter(span=0.5, grid_resolution=20)
        model.fit(self.y_series, self.C_series)

        # 1. Forensic (Clean)
        res = model.transform(self.y_series, self.C_series)
        clean = res.adjusted_series

        self.assertEqual(len(clean), self.n_samples)

        # 2. Strategic (Integration)
        # Test with datetime index
        integrated = model.transform_integrated(self.t_dates, self.C_series.values, n_samples=50)
        self.assertEqual(len(integrated), self.n_samples)

        print(f"Loess Integrated at t=0: {integrated[0]}")

    def test_gam_integration(self):
        # Test GamDecanter integration
        model = GamDecanter(n_splines=10)
        model.fit(self.y_series, self.C_series)

        # Gam expects t as numeric or whatever was used in fit?
        # GamDecanter uses index 0..N internally.

        t_input = np.arange(self.n_samples)
        integrated = model.transform_integrated(t_input, self.C_series.values, n_samples=50)

        self.assertEqual(len(integrated), self.n_samples)
        print(f"GAM Integrated at t=0: {integrated[0]}")

    def test_linear_warning(self):
        # Test warning for ArimaDecanter
        model = ArimaDecanter()
        # Mock fit to avoid statsmodels optimization time/errors on random data
        model.model_type = 'linear'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # We don't need to fit to trigger the warning in transform_integrated,
            # but we need to pass data.
            try:
                model.transform_integrated([0], [0])
            except:
                pass # expected to fail execution but trigger warning first

            # Check if warning was issued
            has_warning = any("Linear Model" in str(warn.message) for warn in w)
            self.assertTrue(has_warning, "Should warn about using integration on linear model")

if __name__ == '__main__':
    unittest.main()
