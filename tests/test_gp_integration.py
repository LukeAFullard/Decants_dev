import pytest
import pandas as pd
import numpy as np
from decants.methods.gaussian_process import GPDecanter

def test_gp_transform_integrated():
    """Verify GPDecanter can use MarginalizationMixin logic."""

    # Synthetic Data
    dates = pd.date_range("2023-01-01", periods=20)
    y = pd.Series(np.linspace(0, 10, 20) + np.random.normal(0, 0.1, 20), index=dates)
    X = pd.DataFrame({"temp": np.random.normal(20, 5, 20)}, index=dates)

    # Init GP
    model = GPDecanter(kernel_nu=1.5)
    model.fit(y, X)

    # Check predict_batch directly
    # [t, C]
    # Use internal start time to create correct numeric time
    t_val = (dates[0] - model._t_start).total_seconds() / (24*3600)
    batch_X = np.array([[t_val, 25.0], [t_val, 15.0]])

    preds = model.predict_batch(batch_X)
    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)

    # Check transform_integrated
    # Simulate a scenario: "What if today (last day) had historical weather?"
    last_day = dates[-1]
    hist_weather = X.values # Full history

    # This calls predict_batch internally
    result = model.transform_integrated(
        t=last_day,
        C=hist_weather,
        n_samples=10,
        random_state=42
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 1 # One target day
    assert not np.isnan(result[0])
