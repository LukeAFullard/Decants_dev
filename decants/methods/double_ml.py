from typing import Optional, Union, Any, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.utils.crossfit import TimeSeriesSplitter, InterpolationSplitter
from decants.utils.diagnostics import check_orthogonality, variance_reduction

class DoubleMLDecanter(BaseDecanter):
    """
    Double Machine Learning (DML) Decanter.
    Performs out-of-sample residualization to remove covariate effects without overfitting.

    Can operate in two modes:
    1. 'timeseries' (Default): Strict expanding window. No future leakage.
    2. 'interpolation': Uses future data (LOO/K-Fold) for max information.
    """
    def __init__(
        self,
        nuisance_model: Optional[BaseEstimator] = None,
        splitter: Union[str, Any] = "timeseries",
        n_splits: int = 5,
        min_train_size: int = 20,
        allow_future: bool = False
    ):
        """
        Args:
            nuisance_model: Scikit-learn regressor. Defaults to Ridge(alpha=1.0).
            splitter: 'timeseries', 'kfold', 'loo', or a custom splitter object.
            n_splits: Number of splits for internal splitters.
            min_train_size: Minimum training size for TimeSeriesSplitter.
            allow_future: If True, forces 'interpolation' mode regardless of splitter arg.
        """
        self.nuisance_model = nuisance_model if nuisance_model is not None else Ridge(alpha=1.0)
        self.splitter_arg = splitter
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.allow_future = allow_future
        self.model = None # The last fitted model (conceptually DML doesn't have a single model)

    def _get_splitter(self):
        if self.allow_future:
            # If allow_future is True, we default to LOO if n_splits is roughly sample size or just KFold
            # Actually user guide said "allow_future" creates "interpolation" mode.
            if self.splitter_arg == "loo":
                 return InterpolationSplitter(method="loo")
            return InterpolationSplitter(method="kfold", n_splits=self.n_splits)

        if isinstance(self.splitter_arg, str):
            if self.splitter_arg == "timeseries":
                return TimeSeriesSplitter(n_splits=self.n_splits, min_train_size=self.min_train_size)
            elif self.splitter_arg == "loo":
                 return InterpolationSplitter(method="loo")
            elif self.splitter_arg == "kfold":
                 return InterpolationSplitter(method="kfold", n_splits=self.n_splits)
            else:
                raise ValueError(f"Unknown splitter: {self.splitter_arg}")
        return self.splitter_arg

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "DoubleMLDecanter":
        """
        DoubleML doesn't have a single 'fit' state in the traditional sense,
        but we store the last trained nuisance model on the full dataset for
        potential future 'transform' calls on NEW data (naive application).
        """
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        self.model = clone(self.nuisance_model)
        self.model.fit(X, y)
        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Applies the DoubleML procedure (Cross-Fitting).
        """
        # Data alignment
        y_orig = y.copy() # Keep full original for return
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = y.index.intersection(X.index)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]

        # Prepare storage for OOS predictions
        # Initialize with NaN
        covariate_effect = pd.Series(np.nan, index=y_aligned.index)

        splitter = self._get_splitter()

        # Iterate folds
        # We need to handle the fact that X/y might be non-contiguous if user passed weird data
        # But splitters work on numpy arrays usually.
        X_values = X_aligned.values
        y_values = y_aligned.values
        indices = np.arange(len(y_aligned))

        # To map back to pandas index
        iloc_map = y_aligned.index

        # Track if we did any predictions
        predictions_made = False

        try:
            for train_idx, test_idx in splitter.split(X_values, y_values):
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue

                # Fit nuisance on TRAIN
                est = clone(self.nuisance_model)
                est.fit(X_values[train_idx], y_values[train_idx])

                # Predict on TEST
                preds = est.predict(X_values[test_idx])

                # Store
                # Map integer test_idx back to original index labels
                test_labels = iloc_map[test_idx]
                covariate_effect.loc[test_labels] = preds.flatten() if preds.ndim > 1 else preds
                predictions_made = True

        except ValueError as e:
            # Handle cases like "not enough data for split"
            print(f"Warning: DML Split failed: {e}. Returning NaNs.")

        # If strict time series, early values remain NaN.
        # If LOO, all should be filled.

        adjusted = y_aligned - covariate_effect

        # Run Diagnostics
        ortho_stats = check_orthogonality(adjusted, X_aligned)
        var_red = variance_reduction(y_aligned, adjusted)

        stats = {
            "variance_reduction": var_red,
            "orthogonality": ortho_stats,
            "n_splits": self.n_splits,
            "mode": "interpolation" if self.allow_future or self.splitter_arg in ["loo", "kfold"] else "timeseries"
        }

        # Reindex to original input shape (propagating NaNs where we didn't have data/predictions)
        final_adjusted = adjusted.reindex(y_orig.index)
        final_effect = covariate_effect.reindex(y_orig.index)

        return DecantResult(
            original_series=y_orig,
            adjusted_series=final_adjusted,
            covariate_effect=final_effect,
            model=self.model, # Full model (naive)
            stats=stats
        )
