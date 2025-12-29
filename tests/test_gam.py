import unittest
import pandas as pd
import numpy as np
from decants.methods.gam import GamDecanter

class TestGamDecanter(unittest.TestCase):
    def setUp(self):
        # Generate Synthetic Data
        # y = 2*X + Trend + Noise
        np.random.seed(42)
        n = 100
        self.date_range = pd.date_range("2021-01-01", periods=n)

        # Trend: quadratic
        time_idx = np.arange(n)
        self.trend = 0.01 * time_idx**2

        # Covariate X: random sine wave
        self.X = pd.Series(np.sin(time_idx / 5) + np.random.normal(0, 0.1, n), index=self.date_range)

        # Noise
        noise = np.random.normal(0, 0.5, n)

        # True Effect = 2 * X
        self.true_effect = 2 * self.X

        # Y
        self.y = self.true_effect + self.trend + noise
        self.y.index = self.date_range

    def test_gam_recovery(self):
        # Initialize
        decanter = GamDecanter(n_splines=10, lam=0.5)

        # Fit Transform
        # Test with gridsearch=True (default) to ensure it works
        result = decanter.fit_transform(self.y, self.X, gridsearch=True)

        # Check Shapes
        self.assertEqual(len(result.adjusted_series), len(self.y))
        self.assertEqual(len(result.covariate_effect), len(self.y))

        # Check Recovery
        # Adjusted Series should be close to Trend + Noise
        # Covariate Effect should be close to 2 * X

        # Correlation between estimated effect and true effect
        corr = np.corrcoef(result.covariate_effect, self.true_effect)[0, 1]
        print(f"Correlation between estimated and true effect: {corr}")
        self.assertGreater(corr, 0.9, "Covariate effect should be highly correlated with true effect (2*X)")

        # Check Magnitude (approximate)
        # We can run a simple regression on estimated_effect vs X to see if coef is ~2
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(self.X, result.covariate_effect)
        print(f"Recovered Coefficient: {slope}")
        self.assertAlmostEqual(slope, 2.0, delta=0.5, msg="Should recover coefficient approx 2.0")

    def test_input_alignment(self):
        # Mismatched indices
        y_short = self.y.iloc[10:]
        X_short = self.X.iloc[:-10]

        decanter = GamDecanter()
        # Should align to intersection
        result = decanter.fit_transform(y_short, X_short, gridsearch=False)

        common_len = len(y_short.index.intersection(X_short.index))
        self.assertEqual(len(result.adjusted_series), common_len)

if __name__ == '__main__':
    unittest.main()
