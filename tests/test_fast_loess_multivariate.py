import numpy as np
import pandas as pd
import pytest
from decants.methods.loess import FastLoessDecanter

def test_multivariate_loess():
    # Generate data with 2 covariates
    n = 100
    rng = np.random.default_rng(42)

    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    t = np.arange(n)

    # Target: Y = 0.1*t + 2*C1 - 1.5*C2
    c1 = rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    y = 0.1 * t + 2.0 * c1 - 1.5 * c2 + rng.standard_normal(n) * 0.1

    df = pd.DataFrame({'y': y, 'c1': c1, 'c2': c2}, index=dates)

    # Fit FastLoess
    # Reduce grid resolution to avoid slowness in test
    decanter = FastLoessDecanter(grid_resolution=10, span=0.5)

    result = decanter.fit_transform(df['y'], df[['c1', 'c2']])

    # Check shape
    assert len(result.adjusted_series) == n
    assert len(result.covariate_effect) == n

    # Check roughly correct effect
    # Effect should be approx 2*C1 - 1.5*C2
    true_effect = 2.0 * c1 - 1.5 * c2
    est_effect = result.covariate_effect.values

    # Normalize by median diff (FastLoess centers on median)
    # The estimated effect is relative to baseline_c (median of c1, median of c2)
    # So True Effect Baseline = 2*median(c1) - 1.5*median(c2)
    baseline_true = 2.0 * np.median(c1) - 1.5 * np.median(c2)

    # Adjust true effect to match relative scale
    true_effect_rel = true_effect - baseline_true

    rmse = np.sqrt(np.mean((true_effect_rel - est_effect)**2))
    print(f"Multivariate Loess RMSE: {rmse}")

    # Should be reasonably low (e.g. < 0.5)
    assert rmse < 0.5

def test_loess_dimension_explosion_guard():
    # Should raise error if grid points > limit
    # Default limit in code is 2,000,000
    # If grid_res=100 and dim=4 (3 covars + 1 time) => 100^4 = 100,000,000 -> Fail

    n = 10
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        'y': np.random.randn(n),
        'c1': np.random.randn(n),
        'c2': np.random.randn(n),
        'c3': np.random.randn(n)
    }, index=dates)

    decanter = FastLoessDecanter(grid_resolution=100) # 100^4 is huge

    with pytest.raises(ValueError, match="FastLoess dimension explosion"):
        decanter.fit(df['y'], df[['c1', 'c2', 'c3']])

if __name__ == "__main__":
    test_multivariate_loess()
    test_loess_dimension_explosion_guard()
    print("All tests passed!")
