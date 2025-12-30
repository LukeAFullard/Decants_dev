
import pandas as pd
import numpy as np
import pytest
from decants.methods.gam import GamDecanter
from decants.methods.ml import MLDecanter
from decants.methods.prophet import ProphetDecanter
from decants.methods.arima import ArimaDecanter
from decants.methods.double_ml import DoubleMLDecanter

def test_empty_intersection_validation():
    y = pd.Series([1, 2, 3], index=[0, 1, 2])
    X = pd.DataFrame({'a': [1, 2, 3]}, index=[3, 4, 5]) # No overlap

    methods = [
        GamDecanter(),
        MLDecanter(),
        ProphetDecanter(),
        ArimaDecanter(),
        DoubleMLDecanter()
    ]

    for method in methods:
        print(f"Testing {method.__class__.__name__}")
        with pytest.raises(ValueError, match="Intersection of y and X indices is empty"):
            method.fit(y, X)

def test_gam_gridsearch_override_logic():
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    # Initialize with specific lambda
    specified_lam = 1000.0
    gam = GamDecanter(lam=specified_lam)
    gam.fit(y, X)

    # Check if lambda was preserved (as a list of arrays)
    # pygam stores lam as a list of arrays or similar structure.
    # We verify the first term's lambda is what we set.
    lam_val = gam.model.lam
    if isinstance(lam_val, list):
         val = lam_val[0]
         if hasattr(val, '__iter__'):
             val = val[0]
         assert val == 1000.0
    else:
        assert lam_val == 1000.0

def test_decant_result_type_compliance():
    # This is a static check really, but we can verify we can put dicts in stats without error
    from decants.objects import DecantResult
    res = DecantResult(
        original_series=pd.Series(),
        adjusted_series=pd.Series(),
        covariate_effect=pd.Series(),
        model=None,
        stats={"nested": {"a": 1}}
    )
    assert res.stats["nested"]["a"] == 1
