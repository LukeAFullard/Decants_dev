import pandas as pd
import numpy as np
from pygam import LinearGAM, s, l
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult
import warnings

class GamDecanter(BaseDecanter):
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

        # Create Time Index (0 to N-1)
        time_idx = np.arange(len(y))

        # Construct Feature Matrix
        X_matrix = np.column_stack([time_idx, X.values])

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

        # Warn about time index reset
        warnings.warn("GamDecanter.transform resets time index to 0..N. "
                      "If transforming new/future data, the trend component (term 0) "
                      "may be incorrect relative to training period. "
                      "Ensure data provided is consistent or continuous with training if trend matters.",
                      UserWarning)

        time_idx = np.arange(len(y))
        X_matrix = np.column_stack([time_idx, X.values])

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
