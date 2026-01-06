import pandas as pd
import numpy as np
from pygam import LinearGAM, s, l
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin
from decants.utils.time import prepare_time_feature
import warnings
import datetime

class GamDecanter(BaseDecanter, MarginalizationMixin):
    """
    GAM-based Decanter using pygam.
    Uses Semi-Parametric Smoothing to separate covariate effects from time trend.
    """
    def __init__(self, n_splines: int = 25, lam: Union[float, list] = 0.6,
                 trend_term: str = 'spline', strict: bool = False):
        """
        Args:
            n_splines (int): Number of splines to use for terms.
            lam (float or list): Smoothing parameter(s).
            trend_term (str): 'spline' for s(0) or 'linear' for l(0) for the time index.
            strict: Enforce strict mode.
        """
        super().__init__(strict=strict) # Initialize Audit Log
        self.n_splines = n_splines
        self.lam = lam
        self.trend_term = trend_term
        self.model = None
        self.feature_names = []
        self._t_start = None

        # Log init params
        self._log_event("init", {
            "n_splines": n_splines,
            "lam": str(lam) if isinstance(lam, list) else lam, # lists might be numpy arrays
            "trend_term": trend_term
        })

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "GamDecanter":
        """
        Fit the GAM model.
        """
        # Log fit start
        self._log_event("fit_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X),
            "kwargs": str(kwargs)
        })

        # align indices
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Ensure alignment
        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Initialize start time if needed
        time_feature, self._t_start = prepare_time_feature(common_idx, self._t_start)

        # Construct Feature Matrix
        X_matrix = np.column_stack([time_feature, X.values])

        self.feature_names = ['time_trend'] + list(X.columns)

        # Construct Model Terms
        if self.trend_term == 'linear':
            terms = l(0)
        else:
            terms = s(0, n_splines=self.n_splines)

        # Add terms for covariates
        for i in range(1, X_matrix.shape[1]):
            terms += s(i, n_splines=self.n_splines)

        self.model = LinearGAM(terms, lam=self.lam)

        # Gridsearch Logic
        should_gridsearch = kwargs.get('gridsearch', True)

        if should_gridsearch:
             try:
                 grid_kwargs = {k:v for k,v in kwargs.items() if k!='gridsearch'}
                 self.model = self.model.gridsearch(X_matrix, y.values, **grid_kwargs)
                 self._log_event("optimization", {"method": "gridsearch", "success": True, "lam": str(self.model.lam)})
             except Exception as e:
                 self._log_event("optimization", {"method": "gridsearch", "success": False, "error": str(e)})
                 self.model.fit(X_matrix, y.values)
        else:
            self.model.fit(X_matrix, y.values)
            self._log_event("optimization", {"method": "fit", "lam": str(self.model.lam)})

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Isolate effect and adjust series.
        """
        self._log_event("transform_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X)
        })

        if self.model is None:
            raise ValueError("Model not fitted yet.")

        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Align indices
        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Use consistent time feature
        # If transforming new data, uses self._t_start from fit
        time_feature, _ = prepare_time_feature(common_idx, self._t_start)

        X_matrix = np.column_stack([time_feature, X.values])

        covariate_effect = np.zeros(len(y))
        effect_lower = np.zeros(len(y))
        effect_upper = np.zeros(len(y))

        for i in range(1, X_matrix.shape[1]):
            try:
                pdep, conf = self.model.partial_dependence(term=i, X=X_matrix, meshgrid=False, width=0.95)
                covariate_effect += pdep
                effect_lower += conf[:, 0]
                effect_upper += conf[:, 1]
            except Exception:
                 effect = self.model.partial_dependence(term=i, X=X_matrix, meshgrid=False)
                 covariate_effect += effect

        adjusted = y - covariate_effect

        conf_int_df = pd.DataFrame({
            'lower': effect_lower,
            'upper': effect_upper
        }, index=y.index)

        stats = {
            "score": self.model.statistics_['edof'],
            "AIC": self.model.statistics_['AIC'],
            "pseudo_r2": self.model.statistics_['pseudo_r2']['explained_deviance']
        }

        self._log_event("transform_complete", {"stats": stats})

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            params={"lam": self.model.lam},
            conf_int=conf_int_df,
            stats=stats
        )

    def predict_batch(self, X):
        """
        Helper for MarginalizationMixin.
        Correctly formats [t, C] batch (handling objects/timestamps) for pygam.
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        if len(X) == 0:
            return np.array([])

        # X is passed from mixin as np.array (possibly object if datetime)
        # X[:, 0] is t, X[:, 1:] is C.

        X_t = X[:, 0]
        X_c = X[:, 1:]

        # Ensure Time is numeric relative to t_start

        # Robust check for datetime-like objects in the array
        is_datetime = False
        if X_t.dtype == object and len(X_t) > 0:
             # Check the first element
             first_elem = X_t[0]
             if isinstance(first_elem, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
                  is_datetime = True
             elif isinstance(first_elem, (int, np.integer)) and self._t_start is not None:
                  # Handle case where numpy implicitly converted datetime64[ns] to int64 (nanoseconds)
                  # Heuristic: > 1e16 implies nanoseconds (since ~1970)
                  if first_elem > 1e16:
                       is_datetime = True

        elif np.issubdtype(X_t.dtype, np.datetime64):
             is_datetime = True

        if is_datetime:
             # Convert to DatetimeIndex to handle mix of np.datetime64, datetime.date etc.
             # pd.to_datetime handles nanosecond integers correctly if they are clearly timestamps
             idx = pd.to_datetime(X_t)
             numeric_t, _ = prepare_time_feature(idx, self._t_start)
        else:
             numeric_t = X_t.astype(float)

        # Explicitly cast covariates to float
        numeric_c = X_c.astype(float)

        # Re-stack
        X_final = np.column_stack([numeric_t, numeric_c])

        return self.model.predict(X_final)

    def get_model_params(self) -> Dict[str, Any]:
        """Return fitted GAM parameters."""
        if self.model is None:
            return {}
        return {
            "lam": self.model.lam,
            "n_splines": self.n_splines,
            "trend_term": self.trend_term,
            "statistics": self.model.statistics_
        }
