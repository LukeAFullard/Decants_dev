
import pandas as pd
import numpy as np
import pytest
import os
import json
from decants.methods.gam import GamDecanter

def test_audit_log_creation():
    # Setup
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    # Fit & Transform
    gam = GamDecanter(n_splines=5)
    gam.fit(y, X)
    gam.transform(y, X)

    # Check in-memory audit log
    assert hasattr(gam, '_audit_log')
    assert "history" in gam._audit_log

    history_types = [entry['type'] for entry in gam._audit_log['history']]
    assert "init" in history_types
    assert "fit_start" in history_types
    assert "optimization" in history_types
    assert "transform_start" in history_types
    assert "transform_complete" in history_types

def test_audit_file_export():
    # Setup
    X = pd.DataFrame({'a': np.random.randn(50)})
    y = pd.Series(np.random.randn(50) + X['a'])

    gam = GamDecanter(n_splines=5)
    gam.fit(y, X)

    # Add manual note
    gam.add_interpretation("This model looks solid.", author="Jules")

    filepath = "test_audit_export.pkl"
    audit_path = filepath + ".audit.json"

    try:
        gam.save(filepath)

        assert os.path.exists(filepath)
        assert os.path.exists(audit_path)

        # Verify JSON content
        with open(audit_path, 'r') as f:
            data = json.load(f)

        assert data['library_versions']['python'] is not None
        assert len(data['history']) > 0

        # Check interpretation
        interpretations = [h for h in data['history'] if h['type'] == 'interpretation']
        assert len(interpretations) == 1
        assert interpretations[0]['details']['author'] == "Jules"

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(audit_path):
            os.remove(audit_path)
