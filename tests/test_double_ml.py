import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from decants.methods.double_ml import DoubleMLDecanter
from decants.utils.crossfit import TimeSeriesSplitter

class TestDoubleMLDecanter(unittest.TestCase):
    def setUp(self):
        # Create synthetic data: Y = 2*X + trend + noise
        np.random.seed(42)
        n = 100
        self.X = pd.DataFrame({'x1': np.random.randn(n), 'x2': np.random.randn(n)}, index=pd.date_range('2020-01-01', periods=n))

        # Signal we want to keep (trend)
        self.trend = np.linspace(0, 10, n)

        # Signal we want to remove (covariate effect)
        self.cov_effect = 2 * self.X['x1'] - 1.5 * self.X['x2']

        self.y = self.trend + self.cov_effect + np.random.normal(0, 0.1, n)
        self.y = pd.Series(self.y, index=self.X.index)

    def test_mechanics_run(self):
        """Test that it runs without error on standard inputs."""
        decanter = DoubleMLDecanter()
        res = decanter.fit_transform(self.y, self.X)
        self.assertIsNotNone(res.adjusted_series)
        self.assertEqual(len(res.adjusted_series), len(self.y))

    def test_timeseries_mode_nans(self):
        """
        In time series mode, the initial window (min_train_size) cannot be predicted
        out-of-sample, so it should be NaN.
        """
        min_train = 20
        decanter = DoubleMLDecanter(splitter="timeseries", min_train_size=min_train, n_splits=5)
        res = decanter.fit_transform(self.y, self.X)

        # Check first few are NaN
        self.assertTrue(res.covariate_effect.iloc[:min_train].isna().all())
        self.assertTrue(res.adjusted_series.iloc[:min_train].isna().all())

        # Check later ones are filled
        # Note: Depending on splitting logic, we might miss some points between folds?
        # Our TimeSeriesSplitter is intended to be contiguous for the test part.
        self.assertFalse(res.adjusted_series.iloc[min_train:].isna().all())

    def test_loo_mode_full_coverage(self):
        """In LOO mode, we should have predictions for every point."""
        decanter = DoubleMLDecanter(splitter="loo")
        res = decanter.fit_transform(self.y, self.X)

        self.assertFalse(res.adjusted_series.isna().any())
        self.assertEqual(len(res.adjusted_series), len(self.y))

    def test_orthogonality(self):
        """
        Verify that the cleaned signal is orthogonal to covariates.
        Using LinearRegression as nuisance to match generation process perfectly.
        """
        # Y = 2*X1 - 1.5*X2 + Trend
        # Ideally, Adjusted = Trend
        # Adjusted shouldn't correlate with X1 or X2

        decanter = DoubleMLDecanter(nuisance_model=LinearRegression(), splitter="kfold", n_splits=5)
        res = decanter.fit_transform(self.y, self.X)

        # Check stats
        ortho = res.stats['orthogonality']
        # Correlation should be very low
        self.assertLess(ortho['max_abs_corr'], 0.15)

        # Check variance reduction (should be high because we removed a strong signal)
        self.assertGreater(res.stats['variance_reduction'], 0.0)

    def test_small_data_fallback(self):
        """Test with N=120 using LOO as requested."""
        n = 120
        X = pd.DataFrame({'x1': np.random.randn(n)}, index=pd.date_range('2020-01-01', periods=n))
        y = 3 * X['x1'] + np.random.normal(0, 1, n)
        y = pd.Series(y, index=X.index)

        decanter = DoubleMLDecanter(splitter="loo")
        res = decanter.fit_transform(y, X)

        self.assertFalse(res.adjusted_series.isna().any())
        # Check it actually did something (variance reduced)
        self.assertGreater(res.stats['variance_reduction'], 0.5)

    def test_leakage_safeguard(self):
        """
        Verify strict time series splitting doesn't use future data.
        We can't easily probe the internal split loop from here without mocking,
        but we can check if the splitter generates correct indices.
        """
        splitter = TimeSeriesSplitter(n_splits=3, min_train_size=10)
        X = np.arange(30)
        splits = list(splitter.split(X))

        for train_idx, test_idx in splits:
            max_train = train_idx.max()
            min_test = test_idx.min()
            # Train must strictly precede Test
            self.assertLess(max_train, min_test)

if __name__ == '__main__':
    unittest.main()
