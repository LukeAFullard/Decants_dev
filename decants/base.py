from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union, Any, Dict, List
from .objects import DecantResult
import pickle
import hashlib
import json
import sys
import datetime
import platform

class BaseDecanter(ABC):
    """
    Abstract Base Class for all Decanter methods.
    Includes Audit Mode for robust tracking of provenance and decisions.
    """
    def __init__(self):
        # Audit Log
        self._audit_log: Dict[str, Any] = {
            "created_at": datetime.datetime.now().isoformat(),
            "library_versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__
            },
            "history": [],
            "interpretations": []
        }
        # We capture init params in subclasses, or generally?
        # Ideally subclasses call super().__init__() but they often don't if they are dataclasses or simple.
        # We will log the *current* state at save time, but history is important.

    def _ensure_audit_log(self):
        """Helper to ensure audit log is initialized."""
        if not hasattr(self, '_audit_log'):
            raise RuntimeError("Audit log not initialized. Ensure super().__init__() is called in your subclass.")

    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Internal method to append to audit log."""
        self._ensure_audit_log()

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        self._audit_log["history"].append(entry)

    def _hash_data(self, obj: Union[pd.Series, pd.DataFrame]) -> str:
        """Compute SHA256 hash of pandas object for provenance."""
        # hashing pandas is tricky. We use value bytes.
        try:
            return hashlib.sha256(pd.util.hash_pandas_object(obj, index=True).values).hexdigest()
        except Exception:
            return "hash_computation_failed"

    def add_interpretation(self, text: str, author: Optional[str] = None):
        """
        Add a human interpretation or note to the audit log.

        Args:
            text (str): The interpretation note.
            author (str, optional): Name of the analyst.
        """
        self._log_event("interpretation", {"text": text, "author": author})

    def _validate_alignment(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> pd.Index:
        """
        Validate alignment of y and X and return common index.
        Raises ValueError if intersection is empty.
        """
        common_idx = y.index.intersection(X.index)
        if len(common_idx) == 0:
            raise ValueError("Intersection of y and X indices is empty. Cannot fit or transform.")
        return common_idx

    def get_model_params(self) -> Dict[str, Any]:
        """
        Return a dictionary of fitted model parameters/coefficients for serialization.
        Subclasses should override this to provide meaningful exports (e.g. coefficients).
        """
        return {}

    @abstractmethod
    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "BaseDecanter":
        """
        Fit the model to the data.
        """
        self._ensure_audit_log()
        pass

    @abstractmethod
    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Apply the adjustment using the fitted model.
        """
        self._ensure_audit_log()
        pass

    def fit_transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> DecantResult:
        """
        Fit the model and apply the adjustment.
        """
        return self.fit(y, X, **kwargs).transform(y, X)

    def save(self, filepath: str):
        """
        Save the fitted decanter to a file using pickle.
        Also saves a sidecar .audit.json file and a .params.json file.

        Args:
            filepath (str): The path to save the model to (e.g., 'model.pkl').
        """
        # Ensure audit log is up to date with current params if possible
        # (Subclasses might store params differently, but we have history)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        # Helper to serialize non-json types in log
        def default(o):
            if isinstance(o, (np.integer, np.floating)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        # Save Audit JSON
        audit_path = filepath + ".audit.json"
        with open(audit_path, 'w') as f:
            json.dump(self._audit_log, f, indent=2, default=default)

        # Save Params JSON
        params_path = filepath + ".params.json"
        try:
            params = self.get_model_params()
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2, default=default)
        except Exception as e:
            # Fallback if params extraction fails
            with open(params_path, 'w') as f:
                json.dump({"error": f"Failed to export params: {str(e)}"}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BaseDecanter":
        """
        Load a fitted decanter from a file.

        Security Warning:
            This method uses `pickle.load` which is not secure against erroneous or maliciously
            constructed data. Only load files from trusted sources.

        Args:
            filepath (str): The path to load the model from.

        Returns:
            BaseDecanter: The loaded model instance.
        """
        # Security Warning for defensibility
        # We don't raise error, but we document it.

        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        # Ensure it has audit log if it was an old model (migration)
        if not hasattr(obj, '_audit_log'):
            obj._audit_log = {"warning": "Loaded from legacy model without audit log"}

        if not isinstance(obj, BaseDecanter):
             # Warning or strict check?
             if not isinstance(obj, BaseDecanter):
                 raise TypeError(f"Loaded object is not a Decanter, got {type(obj)}")
        return obj
