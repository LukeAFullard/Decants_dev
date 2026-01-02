import unittest
import pandas as pd
import numpy as np
import datetime
from decants.utils.time import prepare_time_feature
from decants.utils.crossfit import TimeSeriesSplitter
from decants.methods.double_ml import DoubleMLDecanter
from decants.methods.gam import GamDecanter
from decants.methods.loess import FastLoessDecanter
from decants.methods.gaussian_process import GPDecanter

class TestAuditFixes(unittest.TestCase):

    def test_prepare_time_feature(self):
        # 1. Datetime Index
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        t_feat, t_start = prepare_time_feature(dates)

        self.assertIsInstance(t_start, pd.Timestamp)
        self.assertEqual(t_start, dates[0])
        # Day 0 is 0.0, Day 1 is 1.0
        self.assertEqual(t_feat[0], 0.0)
        self.assertEqual(t_feat[1], 1.0)

        # 2. Consistent Transform
        dates_new = pd.date_range("2020-01-05", periods=5, freq="D")
        t_feat_new, _ = prepare_time_feature(dates_new, t_start)
        self.assertEqual(t_feat_new[0], 4.0)

        # 3. Numeric Index
        nums = pd.Index([0, 10, 20])
        t_feat_num, _ = prepare_time_feature(nums)
        self.assertEqual(t_feat_num[0], 0.0)
        self.assertEqual(t_feat_num[1], 10.0)

        # 4. Object Index with Dates
        obj_dates = pd.Index([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)], dtype='object')
        t_feat_obj, t_start_obj = prepare_time_feature(obj_dates)
        self.assertEqual(t_feat_obj[1], 1.0)

    def test_timeseries_splitter_warning(self):
        # Unsorted index
        idx = [0, 2, 1, 3]
        X = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
        tscv = TimeSeriesSplitter(n_splits=2, min_train_size=1)

        with self.assertWarns(UserWarning):
            list(tscv.split(X))

    def test_double_ml_small_data_robustness(self):
        # Very small data, n_splits > n_samples/2
        y = pd.Series([1, 2, 3, 4])
        X = pd.DataFrame({"a": [1, 1, 1, 1]})

        # Should raise ValueError if min_train_size + n_splits > n_samples
        dml = DoubleMLDecanter(n_splits=3, min_train_size=2)

        # fit works (naive)
        dml.fit(y, X)

        # transform calls splitter.split which raises ValueError.
        # DoubleMLDecanter catches ValueError and warns, returning NaNs.
        res = dml.transform(y, X)

        self.assertTrue(res.adjusted_series.isna().all() or res.adjusted_series.isna().any())
        self.assertIn("warning", str(res.stats))

    def test_refactored_methods_run(self):
        # Quick smoke test for GAM, Loess, GP to ensure refactor didn't break init/fit
        y = pd.Series(np.random.randn(20), index=pd.date_range("2020-01-01", periods=20))
        X = pd.DataFrame({"x1": np.random.randn(20)}, index=y.index)

        # GP
        gp = GPDecanter()
        gp.fit(y, X)
        res_gp = gp.transform(y, X)
        self.assertEqual(len(res_gp.adjusted_series), 20)

        # Loess
        loess = FastLoessDecanter()
        loess.fit(y, X)
        res_loess = loess.transform(y, X)
        self.assertEqual(len(res_loess.adjusted_series), 20)

        # GAM
        # Need pygam installed. Assuming it is.
        try:
            gam = GamDecanter()
            gam.fit(y, X)
            res_gam = gam.transform(y, X)
            self.assertEqual(len(res_gam.adjusted_series), 20)
        except Exception as e:
            # If pygam issues, fail test
            self.fail(f"Gam failed: {e}")

if __name__ == '__main__':
    unittest.main()
