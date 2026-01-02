import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
from decants.methods.ml import MLDecanter

def test_source_code_hash_presence():
    """Verify that source hash is computed and stored."""
    model = MLDecanter()
    audit_log = model._audit_log

    assert "library_versions" in audit_log
    assert "source_hash" in audit_log["library_versions"]

    hash_val = audit_log["library_versions"]["source_hash"]
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64 # SHA-256 length
    assert hash_val != "unknown_source_location"

def test_strict_mode_enforcement():
    """Verify that strict=True enforces integrity checks in a real subclass."""

    # Create unsorted data
    dates = pd.date_range("2023-01-01", periods=5)
    y = pd.Series(np.random.randn(5), index=dates)
    X = pd.DataFrame({"feat": np.random.randn(5)}, index=dates)

    # Shuffle
    y_shuffled = y.sample(frac=1, random_state=42)
    X_shuffled = X.loc[y_shuffled.index]

    # 1. Normal mode (strict=False)
    model_lax = MLDecanter(strict=False)
    assert model_lax.verify_integrity is False

    # Note: MLDecanter.transform usually fits then transforms, or we manually call validation.
    # We can check internal state or just call validate_alignment if accessible,
    # but fit() calls it.
    # MLDecanter.fit calls _validate_alignment.
    # However, BaseDecanter._validate_alignment RAISES if integrity check fails.

    # fit() aligns data. If sorting is not enforced, it just aligns common index.
    model_lax.fit(y_shuffled, X_shuffled) # Should succeed (it sorts implicitly or just aligns)

    # 2. Strict mode (strict=True) -> Should raise ValueError because input is unsorted
    # and strict mode enforces input sorting.
    model_strict = MLDecanter(strict=True)
    assert model_strict.verify_integrity is True

    with pytest.raises(ValueError, match="monotonic"):
        model_strict.fit(y_shuffled, X_shuffled)

def test_strict_mode_non_datetime():
    """Verify that strict mode warns/checks for non-datetime."""

    idx = pd.Index([1, 2, 3, 4, 5])
    y = pd.Series(np.random.randn(5), index=idx)
    X = pd.DataFrame({"feat": np.random.randn(5)}, index=idx)

    model_strict = MLDecanter(strict=True)

    with pytest.warns(UserWarning, match="NOT a datetime index"):
        model_strict.fit(y, X)
