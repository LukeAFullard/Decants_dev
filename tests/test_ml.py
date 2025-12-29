import unittest
import pandas as pd
import numpy as np
from decants.methods.ml import MLDecanter
from decants.utils.diagnostics import pearson_correlation, variance_reduction

class TestMLDecanter(unittest.TestCase):
    def setUp(self):
        # Generate Synthetic Data
        np.random.seed(42)
        n = 200
        self.date_range = pd.date_range("2021-01-01", periods=n)

        # Trend
        time_idx = np.arange(n)
        self.trend = 0.05 * time_idx

        # Covariate X
        self.X = pd.Series(np.random.normal(0, 1, n), index=self.date_range, name="feat1")

        # True Effect = 2 * X
        self.true_effect = 2.0 * self.X

        # Y
        self.y = self.trend + self.true_effect + np.random.normal(0, 0.5, n)
        self.y.index = self.date_range

    def test_ml_cv_residualization(self):
        decanter = MLDecanter(cv_splits=5)

        # Fit Transform should use CV
        result = decanter.fit_transform(self.y, self.X)

        # Check that start of series has NaNs due to CV
        # n_splits=5 -> roughly 1/6th of data is initial train?
        # TimeSeriesSplit creates splits of increasing size.
        # First test set starts after first train set.
        self.assertTrue(result.covariate_effect.iloc[0:10].isna().all() or result.covariate_effect.iloc[0:10].isna().any())

        # Check recovery on the later part
        valid_part = result.covariate_effect.dropna()
        self.assertGreater(len(valid_part), 100)

        # Correlation with true effect
        # We align with valid part
        common_idx = valid_part.index
        corr = np.corrcoef(valid_part, self.true_effect.loc[common_idx])[0, 1]
        print(f"ML CV Correlation: {corr}")
        self.assertGreater(corr, 0.7) # RF might be noisy but should correlate

        # Variance Reduction
        vr = variance_reduction(self.y, result.adjusted_series)
        print(f"Variance Reduction: {vr}")
        self.assertGreater(vr, 0.5) # Should reduce variance significantly by removing effect

    def test_diagnostics(self):
        s1 = pd.Series([1, 2, 3, 4, 5])
        s2 = pd.Series([1, 2, 3, 4, 5])
        self.assertAlmostEqual(pearson_correlation(s1, s2), 1.0)

        s3 = pd.Series([1, 1, 1, 1, 1]) # var=0
        # Variance reduction from s1 to s3?
        # 1 - 0/var(s1) = 1.0
        self.assertAlmostEqual(variance_reduction(s1, s3), 1.0)

if __name__ == '__main__':
    unittest.main()
