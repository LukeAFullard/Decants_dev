import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult

class MLDecanter(BaseDecanter):
    """
    ML-based Decanter using scikit-learn.
    Uses Time Series Cross-Validation to generate out-of-sample predictions for residualization.
    """
    def __init__(self, estimator: Optional[BaseEstimator] = None, cv_splits: int = 5):
        """
        Args:
            estimator (BaseEstimator): Scikit-learn regressor. Defaults to RandomForestRegressor.
            cv_splits (int): Number of splits for TimeSeriesSplit.
        """
        super().__init__()
        self.estimator = estimator if estimator is not None else RandomForestRegressor(n_estimators=100, random_state=42)
        self.cv_splits = cv_splits
        self.model = None
        self.feature_names = []

        self._log_event("init", {
            "estimator": self.estimator.__class__.__name__,
            "cv_splits": cv_splits
        })

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "MLDecanter":
        """
        Fit the model on the entire dataset.
        This is used for future `transform` calls on new data.
        """
        self._log_event("fit_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X),
            "kwargs": str(kwargs)
        })

        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        self.feature_names = list(X.columns)

        self.model = clone(self.estimator)
        self.model.fit(X, y)

        self._log_event("fit_complete", {})

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Apply the adjustment using the fitted model (Standard prediction).
        Used for new data or when CV is not desired/possible (e.g. single point).
        """
        self._log_event("transform_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X)
        })

        if self.model is None:
             raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        covariate_effect = self.model.predict(X)
        adjusted = y - covariate_effect

        stats = {}
        self._log_event("transform_complete", {"stats": stats})

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            stats=stats
        )

    def fit_transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> DecantResult:
        """
        Fit and Transform using Time Series Cross-Validation to prevent leakage.
        """
        # Log fit_transform separately as it does special CV logic
        self._log_event("fit_transform_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X),
            "kwargs": str(kwargs)
        })

        # 1. Fit the main model for future use
        self.fit(y, X, **kwargs)

        # 2. Perform CV prediction for the result
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()
        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        covariate_effect = np.full(len(y), np.nan)

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train = y.iloc[train_index]

            # Clone estimator to ensure fresh start
            est = clone(self.estimator)
            est.fit(X_train, y_train)
            pred = est.predict(X_test)

            covariate_effect[test_index] = pred

        # For the initial chunk (indices not in any test set), we have no OOS prediction.
        # We keep NaNs.

        adjusted = y - covariate_effect

        stats = {
            "cv_splits": self.cv_splits
        }

        self._log_event("fit_transform_complete", {"stats": stats})

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            stats=stats
        )
