import unittest
import pandas as pd
import numpy as np
from decants.methods.prophet import ProphetDecanter
import logging

# Suppress Prophet logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

class TestProphetDecanter(unittest.TestCase):
    def setUp(self):
        # Generate Synthetic Data
        np.random.seed(42)
        n = 100
        self.date_range = pd.date_range("2021-01-01", periods=n)

        # Trend: linear
        time_idx = np.arange(n)
        self.trend = 0.1 * time_idx + 10

        # Covariate X: random binary/sine
        self.X = pd.Series(np.where(time_idx % 2 == 0, 5, 0), index=self.date_range, name="promo")

        # Noise
        noise = np.random.normal(0, 0.1, n)

        # True Effect = 1.0 * X
        self.true_effect = 1.0 * self.X

        # Y
        self.y = self.trend + self.true_effect + noise
        self.y.index = self.date_range

    def test_prophet_recovery(self):
        decanter = ProphetDecanter() # Default settings

        result = decanter.fit_transform(self.y, self.X)

        # Check Shapes
        self.assertEqual(len(result.adjusted_series), len(self.y))

        # Check Recovery
        # Prophet should pick up the regressor effect.
        # Since X is 0 or 5, effect should be 0 or 5.

        # Check correlation
        corr = np.corrcoef(result.covariate_effect, self.true_effect)[0, 1]
        print(f"Correlation: {corr}")
        self.assertGreater(corr, 0.95)

        # Check Magnitude
        # Regress effect vs X
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(self.X, result.covariate_effect)
        print(f"Recovered Coefficient: {slope}")
        self.assertAlmostEqual(slope, 1.0, delta=0.2)

        # Adjusted series should look like Trend
        # We can check if Adjusted is smoother than Original?
        # Or just check residuals?

        # Check if we can add multiple regressors
        X2 = pd.DataFrame({'promo': self.X, 'noise_cov': np.random.randn(len(self.X))}, index=self.date_range)
        result2 = decanter.fit_transform(self.y, X2)
        self.assertEqual(len(result2.covariate_effect), len(self.y))

if __name__ == '__main__':
    unittest.main()
