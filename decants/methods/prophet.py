import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Union, Optional, List, Dict, Any
import warnings
from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin

class ProphetDecanter(BaseDecanter, MarginalizationMixin):
    """
    Prophet-based Decanter.
    Uses Bayesian Decomposition (Prophet) to separate covariate effects from trend/seasonality.
    """
    def __init__(self, strict: bool = False, **kwargs):
        """
        Args:
            strict: Enforce strict mode.
            **kwargs: Arguments passed to Prophet constructor (e.g. interval_width, daily_seasonality).
        """
        super().__init__(strict=strict)
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

        # Validation for Prophet index requirement
        if not pd.api.types.is_datetime64_any_dtype(y.index):
             warnings.warn(
                 "Prophet requires a Datetime Index. "
                 "The provided index is not strictly datetime64. "
                 "This may cause Prophet to crash or behave unexpectedly.",
                 UserWarning
             )

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

        # Do NOT include full params in result to avoid serialization bloat
        # Params are accessible via self.model.params if really needed
        # Safely extract 'k' if present
        params_summary = {}
        try:
             if 'k' in self.model.params:
                k_val = self.model.params['k']
                if isinstance(k_val, np.ndarray) and k_val.size > 0:
                     params_summary['k'] = float(k_val.flat[0])
                else:
                     params_summary['k'] = float(k_val)
        except Exception:
             pass # Ignore if extraction fails

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            params=params_summary, # Lightweight summary
            stats=stats
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Helper for MarginalizationMixin.
        Converts the [t, C] numpy array batch into a Prophet-compatible DataFrame.

        Args:
            X (np.ndarray): Input batch of shape [n_samples, 1 + n_features].
                            Column 0 is Time ('ds').
                            Columns 1: are Covariates (in order of self.regressor_names).
        """
        if self.model is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        # X[:, 0] is t, X[:, 1:] is C.
        X_t = X[:, 0]
        X_c = X[:, 1:]

        # Prophet expects a DataFrame with 'ds' and regressor columns
        data = {'ds': X_t}

        # Add regressors
        # We assume X_c columns correspond exactly to self.regressor_names in order
        if X_c.shape[1] != len(self.regressor_names):
             raise ValueError(f"Batch covariate dimension {X_c.shape[1]} does not match model regressors {len(self.regressor_names)}")

        for i, name in enumerate(self.regressor_names):
            data[name] = X_c[:, i]

        df_batch = pd.DataFrame(data)

        # Predict
        forecast = self.model.predict(df_batch)

        # We return the TOTAL prediction (yhat) because MarginalizationMixin
        # calculates the expectation of Y given C, then (usually) we subtract the baseline.
        # Wait, MarginalizationMixin calculates E[Y | t, C].
        # If we return yhat, that is exactly E[Y | t, C].
        return forecast['yhat'].values

    def get_model_params(self) -> Dict[str, Any]:
        """Return fitted Prophet parameters."""
        if self.model is None:
            return {}
        # Prophet params are numpy arrays, base.save handles serialization
        return self.model.params
