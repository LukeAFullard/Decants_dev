import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Union, Optional, List, Dict, Any, Tuple
from decants.base import BaseDecanter
from decants.objects import DecantResult

class ArimaDecanter(BaseDecanter):
    """
    ARIMAX-based Decanter using statsmodels.
    Uses Parametric State-Space modeling (SARIMAX) to estimate covariate effects.
    """
    def __init__(self, order: Tuple[int, int, int] = (1, 0, 0),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: Optional[str] = None):
        """
        Args:
            order (tuple): The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters.
            seasonal_order (tuple): The (P,D,Q,s) order of the seasonal component.
            trend (str): Parameter controlling the deterministic trend polynomial.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.results = None
        self.exog_names = []

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "ArimaDecanter":
        """
        Fit the SARIMAX model.
        """
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Align indices
        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        self.exog_names = list(X.columns)

        # Initialize SARIMAX
        # SARIMAX takes endog (y) and exog (X)
        self.model = SARIMAX(endog=y, exog=X,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             trend=self.trend,
                             **{k:v for k,v in kwargs.items() if k not in ['order', 'seasonal_order', 'trend']})

        self.results = self.model.fit(disp=False)

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Isolate effect and adjust series.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet.")

        if isinstance(X, pd.Series):
            X = X.to_frame()

        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Effect Isolation
        params = self.results.params

        covariate_effect = np.zeros(len(y))

        # Initialize CIs with NaN to avoid misleading zeros if calculation skipped
        effect_lower = np.full(len(y), np.nan)
        effect_upper = np.full(len(y), np.nan)

        # Extract sub-covariance matrix for exog parameters
        cov_params = self.results.cov_params()

        # Filter params and cov_params for exog
        exog_params_names = [name for name in self.exog_names if name in params.index]

        if not exog_params_names:
            pass
        else:
            beta_exog = params[exog_params_names].values
            X_exog = X[exog_params_names].values

            covariate_effect = X_exog @ beta_exog

            # Confidence Intervals
            # var(effect_i) = x_i @ cov_beta @ x_i.T
            # For each observation i:
            # x_i is (1, K) row vector.
            # var_i = x_i @ cov_sub @ x_i.T

            # Check if all needed params are in cov_params (sometimes optimization fails or some params are fixed)
            missing_cov = [p for p in exog_params_names if p not in cov_params.index]
            if not missing_cov:
                cov_sub = cov_params.loc[exog_params_names, exog_params_names].values

                # Efficient computation of diagonals of X @ cov @ X.T
                variances = (X_exog.dot(cov_sub) * X_exog).sum(axis=1)

                # Ensure non-negative variances (numerical issues)
                variances = np.maximum(variances, 0)

                std_errors = np.sqrt(variances)

                # 95% CI
                effect_lower = covariate_effect - 1.96 * std_errors
                effect_upper = covariate_effect + 1.96 * std_errors

        adjusted = y - covariate_effect

        conf_int_df = pd.DataFrame({
            'lower': effect_lower,
            'upper': effect_upper
        }, index=y.index)

        stats = {
            "AIC": self.results.aic,
            "BIC": self.results.bic,
            "llf": self.results.llf
        }

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.results,
            params=self.results.params.to_dict(), # Restore .to_dict()
            conf_int=conf_int_df,
            stats=stats
        )
