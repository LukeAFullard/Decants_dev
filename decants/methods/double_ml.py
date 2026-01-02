from typing import Optional, Union, Any, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin
from decants.utils.crossfit import TimeSeriesSplitter, InterpolationSplitter
from decants.utils.diagnostics import check_orthogonality, variance_reduction

class DoubleMLDecanter(BaseDecanter, MarginalizationMixin):
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
        allow_future: bool = False,
        random_state: int = 42
    ):
        """
        Args:
            nuisance_model: Scikit-learn regressor. Defaults to Ridge(alpha=1.0).
            splitter: 'timeseries', 'kfold', 'loo', or a custom splitter object.
            n_splits: Number of splits for internal splitters.
            min_train_size: Minimum training size for TimeSeriesSplitter.
            allow_future: If True, forces 'interpolation' mode regardless of splitter arg.
            random_state: Random state for KFold if used.
        """
        super().__init__()
        self.nuisance_model = nuisance_model if nuisance_model is not None else Ridge(alpha=1.0)
        self.splitter_arg = splitter
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.allow_future = allow_future
        self.random_state = random_state
        self.model = None # The last fitted model (conceptually DML doesn't have a single model)

        self._log_event("init", {
            "nuisance_model": self.nuisance_model.__class__.__name__,
            "splitter_arg": str(splitter),
            "n_splits": n_splits,
            "min_train_size": min_train_size,
            "allow_future": allow_future,
            "random_state": random_state
        })

    def _get_splitter(self):
        if self.allow_future:
            # If allow_future is True, we default to LOO if n_splits is roughly sample size or just KFold
            # Actually user guide said "allow_future" creates "interpolation" mode.
            if self.splitter_arg == "loo":
                 return InterpolationSplitter(method="loo", random_state=self.random_state)
            return InterpolationSplitter(method="kfold", n_splits=self.n_splits, random_state=self.random_state)

        if isinstance(self.splitter_arg, str):
            if self.splitter_arg == "timeseries":
                return TimeSeriesSplitter(n_splits=self.n_splits, min_train_size=self.min_train_size)
            elif self.splitter_arg == "loo":
                 return InterpolationSplitter(method="loo", random_state=self.random_state)
            elif self.splitter_arg == "kfold":
                 return InterpolationSplitter(method="kfold", n_splits=self.n_splits, random_state=self.random_state)
            else:
                raise ValueError(f"Unknown splitter: {self.splitter_arg}")
        return self.splitter_arg

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "DoubleMLDecanter":
        """
        DoubleML doesn't have a single 'fit' state in the traditional sense,
        but we store the last trained nuisance model on the full dataset for
        potential future 'transform' calls on NEW data (naive application).
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

        self.model = clone(self.nuisance_model)
        self.model.fit(X, y)
        self._log_event("fit_complete", {})

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Applies the DoubleML procedure (Cross-Fitting).
        """
        self._log_event("transform_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X)
        })

        # Data alignment
        y_orig = y.copy() # Keep full original for return
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = self._validate_alignment(y, X)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]

        # Defensibility Check: Interpolation Mode Warning
        is_interpolation = self.allow_future or (isinstance(self.splitter_arg, str) and self.splitter_arg in ["loo", "kfold"])
        if is_interpolation:
             self._log_event("warning", {
                 "code": "LEAKAGE_RISK",
                 "message": "WARNING: Future Leakage Enabled (Interpolation Mode). Results Valid for Association/Smoothing, Not Strict Causality."
             })

        # STRICT DEFENSIBILITY: Enforce Sorting
        # DoubleML with TimeSeriesSplit requires strict temporal ordering.
        # If the input was shuffled, we must sort it to avoid leakage.
        if not y_aligned.index.is_monotonic_increasing:
             y_aligned = y_aligned.sort_index()
             X_aligned = X_aligned.sort_index()

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
            # In production/defense, we should log this or warn.
            # Using print is okay for now but maybe logging is better?
            # Keeping print as per original code, but adding check.
            msg = f"Warning: DML Split failed: {e}. Returning NaNs."
            print(msg)
            # Add warning to stats so it's visible in the result object
            # (which is what the test checks and what a user would inspect programmatically)
            # Also explicitly log to audit trail here to ensure timestamped record
            self._log_event("warning", {"message": msg, "error_details": str(e)})

            # Use a mutable stats dict if possible or just append to the final stats
            # But here we build stats later. We'll store the error to include it.
            self._split_error = str(e)

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

        if hasattr(self, '_split_error'):
             stats['warning'] = f"Split failed: {self._split_error}"
             del self._split_error

        # Reindex to original input shape (propagating NaNs where we didn't have data/predictions)
        final_adjusted = adjusted.reindex(y_orig.index)
        final_effect = covariate_effect.reindex(y_orig.index)

        self._log_event("transform_complete", {"stats": stats})

        return DecantResult(
            original_series=y_orig,
            adjusted_series=final_adjusted,
            covariate_effect=final_effect,
            model=self.model, # Full model (naive)
            stats=stats
        )

    def get_model_params(self) -> Dict[str, Any]:
        """Return DoubleML configuration and nuisance model details."""
        params = {
            "nuisance_model": self.nuisance_model.__class__.__name__,
            "splitter_arg": str(self.splitter_arg),
            "n_splits": self.n_splits,
            "min_train_size": self.min_train_size,
            "allow_future": self.allow_future
        }
        # If the nuisance model is linear (e.g. Ridge), try to extract coefficients from the last fitted naive model
        if self.model is not None and hasattr(self.model, "coef_"):
             params["naive_model_coef"] = self.model.coef_
        if self.model is not None and hasattr(self.model, "intercept_"):
             params["naive_model_intercept"] = self.model.intercept_

        return params

    def predict(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Use the naive trained model to predict y from X.
        Note: This does not use cross-fitting/double-ml debiasing,
        it uses the standard model trained in fit().
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        if isinstance(X, pd.Series):
             X = X.to_frame()

        return pd.Series(self.model.predict(X), index=X.index)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Helper for MarginalizationMixin.

        Args:
            X (np.ndarray): Input batch of shape [n_samples, 1 + n_features].
                            Column 0 is Time (ignored by DoubleMLDecanter),
                            Columns 1: are Covariates.
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        # DoubleMLDecanter saves the last fitted naive model on [X -> Y]
        # We use this for integration/inference on scenarios.

        X_c = X[:, 1:]

        # Defensibility: Strict casting
        X_c = X_c.astype(float)

        return self.model.predict(X_c)
