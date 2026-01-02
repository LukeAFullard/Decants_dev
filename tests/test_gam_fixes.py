
import pytest
import pandas as pd
import numpy as np
from decants.methods.gam import GamDecanter
from unittest.mock import MagicMock, patch

def test_gam_defaults_to_gridsearch():
    """
    Verify that GamDecanter defaults to gridsearch=True.
    """
    # We can mock LinearGAM and check if gridsearch is called.
    with patch('decants.methods.gam.LinearGAM') as MockGAM:
        instance = MockGAM.return_value
        instance.gridsearch.return_value = instance # gridsearch returns self usually
        instance.fit.return_value = instance

        gam = GamDecanter()
        X = pd.DataFrame({'A': np.random.randn(20)})
        y = pd.Series(np.random.randn(20))

        gam.fit(y, X)

        # Verify gridsearch was called
        instance.gridsearch.assert_called()
        # Verify fit was NOT called (because gridsearch replaces fit)
        # Wait, the code is:
        # if should_gridsearch: model.gridsearch(...)
        # else: model.fit(...)
        # So fit should NOT be called explicitly if gridsearch is called.
        instance.fit.assert_not_called()

def test_gam_explicit_no_gridsearch():
    """
    Verify that gridsearch can be disabled.
    """
    with patch('decants.methods.gam.LinearGAM') as MockGAM:
        instance = MockGAM.return_value
        instance.fit.return_value = instance

        gam = GamDecanter()
        X = pd.DataFrame({'A': np.random.randn(20)})
        y = pd.Series(np.random.randn(20))

        gam.fit(y, X, gridsearch=False)

        instance.gridsearch.assert_not_called()
        instance.fit.assert_called()
