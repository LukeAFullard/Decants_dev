import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin

class ProphetDecanter(BaseDecanter, MarginalizationMixin):
    """
    Prophet-based Decanter.
    Uses Bayesian Decomposition (Prophet) to separate covariate effects from trend/seasonality.
    """
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arguments passed to Prophet constructor (e.g. interval_width, daily_seasonality).
        """
        super().__init__()
        self.model_kwargs = kwargs
        self.model = None
        self.regressor_names = []
        self.model_type = 'linear' # Flag for MarginalizationMixin (Prophet is generally additive linear)
        self._log_event("init", {"model_kwargs": str(kwargs)})

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "ProphetDecanter":
        """
        Fit the Prophet model.
        """
        self._log_event("fit_start", {
            "y_hash": self._hash_data(y),
            "X_hash": self._hash_data(X),
            "kwargs": str(kwargs)
        })

        # Align indices
        y = y.dropna()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Ensure column names are strings
        X.columns = X.columns.astype(str)

        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Prepare Data for Prophet
        df = pd.DataFrame({'ds': y.index, 'y': y.values})

        self.regressor_names = list(X.columns)

        # Add regressors to df
        for col in self.regressor_names:
            df[col] = X[col].values

        # Initialize Model
        self.model = Prophet(**self.model_kwargs)

        # Add regressors to model
        for col in self.regressor_names:
            self.model.add_regressor(col)

        # Fit
        self.model.fit(df, **kwargs)

        self._log_event("fit_complete", {"regressors": self.regressor_names})

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

        # Ensure column names are strings to match fit
        X.columns = X.columns.astype(str)

        # Align indices
        common_idx = self._validate_alignment(y, X)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Prepare Data for Prediction (Prophet needs 'ds' and regressors)
        future = pd.DataFrame({'ds': y.index})
        for col in self.regressor_names:
            if col not in X.columns:
                 raise ValueError(f"Covariate '{col}' missing from X.")
            future[col] = X[col].values

        # Predict
        forecast = self.model.predict(future)

        # Extract Effects
        covariate_effect = np.zeros(len(y))

        if 'extra_regressors_additive' in forecast.columns:
             covariate_effect += forecast['extra_regressors_additive'].values

        # Fallback manual sum if 'extra_regressors_additive' is missing (older versions?)
        elif self.regressor_names:
             for col in self.regressor_names:
                if col in forecast.columns:
                    covariate_effect += forecast[col].values

        if 'extra_regressors_multiplicative' in forecast.columns:
             multi = forecast['extra_regressors_multiplicative'].values
             trend = forecast['trend'].values
             covariate_effect += trend * multi
             self._log_event("warning", {"message": "Multiplicative regressors detected. Effect isolation is approximate."})

        # Adjusted Series = y - Covariate Effect
        adjusted = y - covariate_effect

        stats = {}

        self._log_event("transform_complete", {"stats": stats})

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            params=self.model.params,
            stats=stats
        )

    def get_model_params(self) -> Dict[str, Any]:
        """Return fitted Prophet parameters."""
        if self.model is None:
            return {}
        # Prophet params are numpy arrays, base.save handles serialization
        return self.model.params
