from typing import Iterator, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, TimeSeriesSplit

class BaseSplitter:
    """Base class for custom splitters."""
    def split(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

class TimeSeriesSplitter(BaseSplitter):
    """
    Strict Time-Series Splitter (Rolling Origin).
    Ensures no future data leakage.

    Args:
        n_splits (int): Number of splits.
        min_train_size (int): Minimum size of the initial training set.
    """
    def __init__(self, n_splits: int = 5, min_train_size: int = 20):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if n_samples < self.min_train_size + self.n_splits:
            raise ValueError(f"Not enough data for TimeSeriesSplitter. Need at least {self.min_train_size + self.n_splits} samples.")

        # Custom implementation for explicit min_train_size control
        indices = np.arange(n_samples)
        # Calculate fold size
        # We reserve min_train_size for the first train
        # The rest (n_samples - min_train_size) is divided into n_splits test sets
        n_test_points = n_samples - self.min_train_size
        if n_test_points <= 0:
             # Fallback: single split if possible or error
             yield (indices[:self.min_train_size], indices[self.min_train_size:])
             return

        # Simple rolling strategy: predict 1 step ahead or block?
        # Standard DML usually predicts a block.
        # Let's try to make blocks roughly equal.
        test_size = n_test_points // self.n_splits
        remainder = n_test_points % self.n_splits

        start = self.min_train_size
        for i in range(self.n_splits):
            end = start + test_size + (1 if i < remainder else 0)
            yield (indices[:start], indices[start:end])
            start = end

class InterpolationSplitter(BaseSplitter):
    """
    Wrapper for K-Fold or LOO splitting.
    Allows future data in training (Interpolation).
    """
    def __init__(self, method: str = "kfold", n_splits: int = 5, random_state: int = 42):
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.method == "loo":
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        return cv.split(X, y)
