from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, Any
from .objects import DecantResult

class BaseDecanter(ABC):
    """
    Abstract Base Class for all Decanter methods.
    """

    def _validate_alignment(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> pd.Index:
        """
        Validate alignment of y and X and return common index.
        Raises ValueError if intersection is empty.
        """
        common_idx = y.index.intersection(X.index)
        if len(common_idx) == 0:
            raise ValueError("Intersection of y and X indices is empty. Cannot fit or transform.")
        return common_idx

    @abstractmethod
    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "BaseDecanter":
        """
        Fit the model to the data.

        Args:
            y (pd.Series): The target time series.
            X (pd.DataFrame or pd.Series): The covariate(s).
            **kwargs: Additional arguments for the underlying model.

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Apply the adjustment using the fitted model.

        Args:
            y (pd.Series): The target time series (can be new data).
            X (pd.DataFrame or pd.Series): The covariate(s) (can be new data).

        Returns:
            DecantResult: The result object containing adjusted series and stats.
        """
        pass

    def fit_transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> DecantResult:
        """
        Fit the model and apply the adjustment.

        Args:
            y (pd.Series): The target time series.
            X (pd.DataFrame or pd.Series): The covariate(s).
            **kwargs: Additional arguments for the underlying model.

        Returns:
            DecantResult: The result object.
        """
        return self.fit(y, X, **kwargs).transform(y, X)
