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
        self.estimator = estimator if estimator is not None else RandomForestRegressor(n_estimators=100, random_state=42)
        self.cv_splits = cv_splits
        self.model = None
        self.feature_names = []

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "MLDecanter":
        """
        Fit the model on the entire dataset.
        This is used for future `transform` calls on new data.
        """
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        self.feature_names = list(X.columns)

        self.model = clone(self.estimator)
        self.model.fit(X, y)

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Apply the adjustment using the fitted model (Standard prediction).
        Used for new data or when CV is not desired/possible (e.g. single point).
        """
        if self.model is None:
             raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        covariate_effect = self.model.predict(X)
        adjusted = y - covariate_effect

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model
        )

    def fit_transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> DecantResult:
        """
        Fit and Transform using Time Series Cross-Validation to prevent leakage.
        """
        # 1. Fit the main model for future use
        self.fit(y, X, **kwargs)

        # 2. Perform CV prediction for the result
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()
        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        # cross_val_predict with cv=tscv
        # Note: cross_val_predict with TimeSeriesSplit only returns predictions for the test sets.
        # The initial training set size depends on n_splits.
        # The output size will be smaller than input size?
        # Actually sklearn cross_val_predict documentation says:
        # "For cv=TimeSeriesSplit, the predictions are returned for elements that are in the test set of at least one split."
        # This means the first chunk of data (initial training set) will NOT have predictions.
        # We need to handle this alignment.

        # We can construct the full array with NaNs for the start.

        # Let's perform cross_val_predict
        # It usually returns an array of size matching input IF method is standard k-fold.
        # For TimeSeriesSplit, it might raise error if sizes mismatch or return padded?
        # Checking docs: "If cv is a TimeSeriesSplit instance, the output will contain predictions for all samples that appear in a test set. The indices of samples that are not in any test set will not be present in the output array? No, it returns an array of the same length as X, but with values only for test sets?"
        # Actually, let's verify behavior or handle it manually to be safe.
        # "For partitions where some samples are not in any test set, the result depends on..."
        # Actually in recent sklearn versions, it might fill with default or raise error?
        # Let's do manual loop to be safe and clear about what's happening.

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
        # We can either leave as NaN (safest) or use in-sample fit (risk of leakage).
        # The plan says: "Note: The first fold will be lost due to CV windowing; handle the truncation of indices gracefully (fill with NaN or truncate)."
        # We will keep NaNs.

        adjusted = y - covariate_effect

        stats = {
            "cv_splits": self.cv_splits
        }

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            stats=stats
        )
