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
        self.n_splines = n_splines
        self.lam = lam
        self.trend_term = trend_term
        self.model = None
        self.feature_names = []

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "GamDecanter":
        """
        Fit the GAM model.

        Constructs a feature matrix: [Time_Index, Covariate_1, ..., Covariate_n]
        """
        # align indices
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Ensure alignment
        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Create Time Index (0 to N-1)
        # Note: This assumes continuity and relative time.
        # If transforming new data, this time index will reset.
        # User is warned to ensure data consistency or use continuous series.
        time_idx = np.arange(len(y))

        # Construct Feature Matrix
        # Column 0: Time Index
        # Columns 1..N: Covariates
        X_matrix = np.column_stack([time_idx, X.values])

        self.feature_names = ['time_trend'] + list(X.columns)

        # Construct Model Terms
        # Term 0 is Time Index.
        if self.trend_term == 'linear':
            terms = l(0)
        else:
            terms = s(0, n_splines=self.n_splines)

        # Add terms for covariates
        for i in range(1, X_matrix.shape[1]):
            terms += s(i, n_splines=self.n_splines)

        self.model = LinearGAM(terms, lam=self.lam)

        # Logic for gridsearch:
        # Default to gridsearch=True ONLY if user did not specify a custom lam in __init__.
        # If kwargs explicitly contains 'gridsearch', respect it.
        # Otherwise, if lam is default (0.6), gridsearch=True. If lam is custom, gridsearch=False.

        explicit_gridsearch = kwargs.get('gridsearch', None)

        should_gridsearch = False
        if explicit_gridsearch is not None:
            should_gridsearch = explicit_gridsearch
        else:
            # Default logic
            if self.lam == 0.6: # 0.6 is the default value in __init__
                should_gridsearch = True
            else:
                should_gridsearch = False

        if should_gridsearch:
             try:
                 # Pass other kwargs to gridsearch
                 grid_kwargs = {k:v for k,v in kwargs.items() if k!='gridsearch'}
                 self.model = self.model.gridsearch(X_matrix, y.values, **grid_kwargs)
             except Exception:
                 # Fallback to fit if gridsearch fails
                 self.model.fit(X_matrix, y.values)
        else:
            self.model.fit(X_matrix, y.values)

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Isolate effect and adjust series.
        """
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

        # Calculate Total Effect of Covariates
        covariate_effect = np.zeros(len(y))

        # Also need to sum confidence intervals (conservative approx).
        effect_lower = np.zeros(len(y))
        effect_upper = np.zeros(len(y))

        for i in range(1, X_matrix.shape[1]):
            # term indices match column indices if we constructed it that way (s(0)+s(1)+...)
            # We request CIs with width=0.95
            try:
                pdep, conf = self.model.partial_dependence(term=i, X=X_matrix, meshgrid=False, width=0.95)
                covariate_effect += pdep

                # Sum the intervals (conservative bounds)
                effect_lower += conf[:, 0]
                effect_upper += conf[:, 1]
            except Exception:
                # Fallback if CI calculation fails
                 effect = self.model.partial_dependence(term=i, X=X_matrix, meshgrid=False)
                 covariate_effect += effect

        adjusted = y - covariate_effect

        conf_int_df = pd.DataFrame({
            'lower': effect_lower,
            'upper': effect_upper
        }, index=y.index)

        # Stats
        stats = {
            "score": self.model.statistics_['edof'],
            "AIC": self.model.statistics_['AIC'],
            "pseudo_r2": self.model.statistics_['pseudo_r2']['explained_deviance']
        }

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            params={"lam": self.model.lam},
            conf_int=conf_int_df,
            stats=stats
        )
