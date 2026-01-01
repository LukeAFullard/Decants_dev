import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from decants.methods.gaussian_process import GPDecanter

class TestGPDecanter(unittest.TestCase):
    def setUp(self):
        # 1. Generate IRREGULAR Data
        # Randomly sample 100 days from a 365-day year
        np.random.seed(42)
        all_days = np.arange(365)
        self.sample_days = np.sort(np.random.choice(all_days, 100, replace=False))

        t = self.sample_days
        # Covariate: Temperature (Sine wave + random noise)
        temp = 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, 100)
        # True Signal: A slow upward linear trend
        true_trend = 0.05 * t
        # Relationship: Sales = Trend + 2 * Temp + Noise
        y_values = true_trend + (2 * temp) + np.random.normal(0, 1, 100)

        # Create pandas objects
        # Use a real datetime index to test that logic too
        self.start_date = pd.Timestamp("2023-01-01")
        date_index = [self.start_date + pd.Timedelta(days=int(d)) for d in t]

        self.y = pd.Series(y_values, index=date_index, name="sales")
        self.X = pd.DataFrame({"temp": temp}, index=date_index)

    def test_fit_transform(self):
        decanter = GPDecanter(kernel_nu=1.5, random_state=42)
        decanter.fit(self.y, self.X)

        # Check if model is fitted
        self.assertIsNotNone(decanter.model)

        result = decanter.transform(self.y, self.X)

        # Check result structure
        self.assertEqual(len(result.adjusted_series), 100)
        self.assertEqual(len(result.covariate_effect), 100)
        self.assertIn("uncertainty", result.stats)

        # Check that we recovered the trend somewhat
        # The true trend is 0.05 * t.
        # adjusted_series should be close to true_trend (+ noise)

        # We can't be too strict with random data, but let's check correlation
        true_trend_values = 0.05 * self.sample_days
        correlation = np.corrcoef(result.adjusted_series.values, true_trend_values)[0, 1]
        self.assertGreater(correlation, 0.8, "Adjusted series should correlate with true trend")

        # Check covariate effect
        # True effect is 2 * temp
        true_effect = 2 * self.X["temp"].values
        eff_corr = np.corrcoef(result.covariate_effect.values, true_effect)[0, 1]
        self.assertGreater(eff_corr, 0.8, "Estimated effect should correlate with true effect")

    def test_save_load(self):
        decanter = GPDecanter(kernel_nu=1.5, random_state=42)
        decanter.fit(self.y, self.X)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "test_gp_model.pkl")
            decanter.save(filepath)

            loaded = GPDecanter.load(filepath)
            self.assertIsNotNone(loaded.model)

            # Check if loaded model works
            result = loaded.transform(self.y, self.X)
            self.assertEqual(len(result.adjusted_series), 100)

    def test_numeric_index(self):
        # Test with plain numeric index
        y_num = self.y.reset_index(drop=True)
        # recreate index as numeric gaps to simulate irregularity if we want,
        # but reset_index(drop=True) makes it 0..99 regular.
        # Let's use the sample_days as index
        y_num.index = self.sample_days
        X_num = self.X.set_index(pd.Index(self.sample_days))

        decanter = GPDecanter()
        decanter.fit(y_num, X_num)
        result = decanter.transform(y_num, X_num)
        self.assertEqual(len(result.adjusted_series), 100)

if __name__ == '__main__':
    unittest.main()
