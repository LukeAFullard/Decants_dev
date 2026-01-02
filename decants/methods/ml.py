import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin

class MLDecanter(BaseDecanter, MarginalizationMixin):
    """
    ML-based Decanter using scikit-learn.
    Uses Time Series Cross-Validation to generate out-of-sample predictions for residualization.
    """
    def __init__(self, estimator: Optional[BaseEstimator] = None, cv_splits: int = 5, strict: bool = False):
        """
        Args:
            estimator (BaseEstimator): Scikit-learn regressor. Defaults to RandomForestRegressor.
            cv_splits (int): Number of splits for TimeSeriesSplit.
            strict (bool): Enforce strict mode for defensibility.
        """
        super().__init__(strict=strict)
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

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Helper for MarginalizationMixin.

        Args:
            X (np.ndarray): Input batch of shape [n_samples, 1 + n_features].
                            Column 0 is Time (ignored by MLDecanter),
                            Columns 1: are Covariates.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        # X is passed from mixin as np.array (possibly object if datetime)
        # X[:, 0] is t (Time), X[:, 1:] is C (Covariates).

        # MLDecanter (e.g. RandomForest) is trained on Covariates ONLY.
        # So we strip the time column.
        X_c = X[:, 1:]

        # Ensure numeric type (handle case where X was object due to Time column)
        # We explicitly cast to float. If this fails, we let the ValueError propagate
        # to ensure strict type safety (Defensibility Recommendation #1).
        X_c = X_c.astype(float)

        return self.model.predict(X_c)

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

        # STRICT DEFENSIBILITY: Enforce Sorting
        if not y.index.is_monotonic_increasing:
             y = y.sort_index()
             X = X.sort_index()

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
