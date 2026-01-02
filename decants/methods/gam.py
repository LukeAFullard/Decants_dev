import pandas as pd
import numpy as np
from pygam import LinearGAM, s, l
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin
import warnings
import datetime

class GamDecanter(BaseDecanter, MarginalizationMixin):
    """
    GAM-based Decanter using pygam.
    Uses Semi-Parametric Smoothing to separate covariate effects from time trend.
    """
    def __init__(self, n_splines: int = 25, lam: Union[float, list] = 0.6,
                 trend_term: str = 'spline'):
        """
        Args:
            n_splines (int): Number of splines to use for terms.
            lam (float or list): Smoothing parameter(s).
            trend_term (str): 'spline' for s(0) or 'linear' for l(0) for the time index.
        """
        super().__init__() # Initialize Audit Log
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

    def _prepare_time(self, index: pd.Index) -> np.ndarray:
        """Converts index to numeric representation (days since start)."""
        if pd.api.types.is_datetime64_any_dtype(index):
            if self._t_start is None:
                self._t_start = index.min()

            ref_time = self._t_start
            # Convert to days since reference
            # Using seconds / (24*3600) gives fractional days
            return (index - ref_time).total_seconds().to_numpy() / (24 * 3600)
        else:
            # Assume numeric. If integer index 0..N, this preserves it.
            return index.to_numpy(dtype=float)

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
        if pd.api.types.is_datetime64_any_dtype(common_idx):
             self._t_start = common_idx.min()

        # Create Time Feature
        time_feature = self._prepare_time(common_idx)

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
        should_gridsearch = kwargs.get('gridsearch', False)

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
        # If transforming new data, _prepare_time uses self._t_start from fit
        time_feature = self._prepare_time(common_idx)

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

        # X is passed from mixin as np.array (possibly object if datetime)
        # X[:, 0] is t, X[:, 1:] is C.

        X_t = X[:, 0]
        X_c = X[:, 1:]

        # Ensure Time is numeric relative to t_start
        # If the input was constructed with timestamps, X_t elements are Timestamps
        if isinstance(X_t[0], (pd.Timestamp, np.datetime64, datetime.datetime, datetime.date)):
             idx = pd.Index(X_t)
             numeric_t = self._prepare_time(idx)
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
