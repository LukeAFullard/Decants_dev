import unittest
import pandas as pd
import numpy as np
from decants.methods.arima import ArimaDecanter

class TestArimaDecanter(unittest.TestCase):
    def setUp(self):
        # Generate Synthetic Data
        np.random.seed(42)
        n = 100
        self.date_range = pd.date_range("2021-01-01", periods=n)

        # AR process for Trend/Noise
        # y_t = 0.8 * y_{t-1} + noise
        ar_component = np.zeros(n)
        for i in range(1, n):
            ar_component[i] = 0.8 * ar_component[i-1] + np.random.normal(0, 0.5)

        self.ar_trend = pd.Series(ar_component, index=self.date_range)

        # Covariate X
        self.X = pd.Series(np.random.normal(0, 1, n), index=self.date_range, name="exog1")

        # True Effect = 3 * X
        self.true_effect = 3.0 * self.X

        # Y
        self.y = self.ar_trend + self.true_effect
        self.y.name = 'y'

    def test_arima_recovery(self):
        # Fit with AR(1)
        decanter = ArimaDecanter(order=(1, 0, 0))

        result = decanter.fit_transform(self.y, self.X)

        # Check Recovery
        # Params should include 'exog1' ~ 3.0 and 'ar.L1' ~ 0.8
        params = result.params

        print("Params:", params)

        self.assertTrue('exog1' in params)
        self.assertAlmostEqual(params['exog1'], 3.0, delta=0.2)

        if 'ar.L1' in params:
             self.assertAlmostEqual(params['ar.L1'], 0.8, delta=0.2)

        # Check Stats
        self.assertIn('AIC', result.stats)

        # Check Adjusted Series
        # Should look like AR component
        corr = np.corrcoef(result.adjusted_series, self.ar_trend)[0, 1]
        print(f"Correlation adjusted vs true AR trend: {corr}")
        self.assertGreater(corr, 0.9)

if __name__ == '__main__':
    unittest.main()
