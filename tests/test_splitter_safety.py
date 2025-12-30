
import pytest
import numpy as np
import pandas as pd
from decants.utils.crossfit import TimeSeriesSplitter

def test_splitter_boundary_conditions():
    """
    Test the critical boundary condition for TimeSeriesSplitter safety.
    Condition: n_samples >= min_train_size + n_splits
    """
    min_train = 20
    n_splits = 5

    # Boundary Failure Case: N = 24
    # 24 < 20 + 5 (25) -> Should raise ValueError
    X_fail = np.arange(24)
    splitter = TimeSeriesSplitter(n_splits=n_splits, min_train_size=min_train)

    with pytest.raises(ValueError, match="Not enough data"):
        list(splitter.split(X_fail))

    # Boundary Success Case: N = 25
    # 25 >= 20 + 5 -> Should succeed with 1 point per split
    X_pass = np.arange(25)
    splits = list(splitter.split(X_pass))

    assert len(splits) == n_splits

    # Verify split sizes
    # n_test_points = 5, n_splits = 5 -> all test sizes = 1
    for train, test in splits:
        assert len(test) == 1
