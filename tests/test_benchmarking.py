
import pandas as pd
import numpy as np
import pytest
from decants import DecantBenchmarker, GamDecanter, MLDecanter

def test_benchmarker_success():
    # Setup Data
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    # Define Models
    models = {
        "GAM": GamDecanter(n_splines=5),
        "ML": MLDecanter(cv_splits=2)
    }

    # Run Benchmark
    bench = DecantBenchmarker()
    df = bench.benchmark(models, y, X)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "GAM" in df.index
    assert "ML" in df.index
    assert df.loc["GAM", "Status"] == "Success"
    assert "Variance Reduction" in df.columns
    assert "Execution Time (s)" in df.columns

    # Check results storage
    assert "GAM" in bench.results
    assert bench.results["GAM"].model is not None

def test_benchmarker_failure_handling():
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    class BrokenDecanter(GamDecanter):
        def fit(self, y, X, **kwargs):
            raise ValueError("Intentional Failure")

    models = {
        "Broken": BrokenDecanter()
    }

    bench = DecantBenchmarker()
    df = bench.benchmark(models, y, X)

    assert df.loc["Broken", "Status"] == "Failed"
    assert "Intentional Failure" in df.loc["Broken", "Error"]
