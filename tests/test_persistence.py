
import pandas as pd
import numpy as np
import pytest
import os
from decants.methods.gam import GamDecanter
from decants.base import BaseDecanter

def test_save_load_gam():
    # Setup Data
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    # Fit
    gam = GamDecanter(n_splines=5)
    gam.fit(y, X)

    # Save
    filepath = "test_gam_model.pkl"
    gam.save(filepath)

    try:
        # Load
        loaded_gam = GamDecanter.load(filepath)

        # Check type
        assert isinstance(loaded_gam, GamDecanter)

        # Check state (basic check if model exists)
        assert loaded_gam.model is not None

        # Verify prediction match
        res_orig = gam.transform(y, X)
        res_loaded = loaded_gam.transform(y, X)

        pd.testing.assert_series_equal(res_orig.adjusted_series, res_loaded.adjusted_series)

    finally:
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

def test_load_base_method():
    # Setup Data
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    # Fit
    gam = GamDecanter(n_splines=5)
    gam.fit(y, X)

    filepath = "test_base_load.pkl"
    gam.save(filepath)

    try:
        # Load using BaseDecanter class method (polymorphism check)
        loaded = BaseDecanter.load(filepath)
        assert isinstance(loaded, GamDecanter)
    finally:
         if os.path.exists(filepath):
            os.remove(filepath)
