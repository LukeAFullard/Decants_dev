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
        common_idx = y.index.intersection(X.index)
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

        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Effect Isolation
        # Extract parameters for exog variables
        # self.results.params contains all params (ar, ma, variance, exog_params)
        # We need to filter for exog params.

        # exog parameters usually match the column names of X
        # or are prefixed? SARIMAX usually keeps names.

        params = self.results.params

        # Identify exog parameters
        # SARIMAX assigns names to params. For exog, it's usually the column name.
        # But if there are collisions or other things?
        # Let's rely on self.exog_names which we saved.

        # Calculate effect: X @ params_exog

        # Safe way: iterate over exog_names and find corresponding param
        # If param not found (e.g. constant?), skip?

        covariate_effect = np.zeros(len(y))
        effect_lower = np.zeros(len(y))
        effect_upper = np.zeros(len(y))

        # We need variance of the effect for CI.
        # Var(X @ beta) = X @ Var(beta) @ X.T (diagonal elements)
        # X is (N, K), Var(beta) is (K, K) -> result (N,) variance.
        # But we only need beta corresponding to exog.

        # Extract sub-covariance matrix for exog parameters
        cov_params = self.results.cov_params()

        # Filter params and cov_params for exog
        exog_params_names = [name for name in self.exog_names if name in params.index]

        if not exog_params_names:
            # Maybe something went wrong or no exog used?
            # Or names changed?
            # SARIMAX might rename if constant added?
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

            cov_sub = cov_params.loc[exog_params_names, exog_params_names].values

            # Efficient computation of diagonals of X @ cov @ X.T
            # sum((X @ cov) * X, axis=1)
            # variance = (X_exog @ cov_sub * X_exog).sum(axis=1)
            # Wait, element-wise mult?
            # (N, K) @ (K, K) -> (N, K)
            # (N, K) * (N, K) -> (N, K) -> sum -> (N,)

            variances = (X_exog.dot(cov_sub) * X_exog).sum(axis=1)
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
            params=self.results.params.to_dict(),
            conf_int=conf_int_df,
            stats=stats
        )
