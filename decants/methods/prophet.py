import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Union, Optional, List, Dict, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult

class ProphetDecanter(BaseDecanter):
    """
    Prophet-based Decanter.
    Uses Bayesian Decomposition (Prophet) to separate covariate effects from trend/seasonality.
    """
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arguments passed to Prophet constructor (e.g. interval_width, daily_seasonality).
        """
        self.model_kwargs = kwargs
        self.model = None
        self.regressor_names = []

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "ProphetDecanter":
        """
        Fit the Prophet model.
        """
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
        # Prophet expects 'ds' and 'y' columns
        # Note: y.index should be Datetime or convertible to it, or numeric.
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

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Isolate effect and adjust series.
        """
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
        # Note: Prophet usually wants 'y' too if we are checking accuracy, but for predict we don't strictly need 'y'.
        # However, we are "transforming" y, so we use y.index.

        future = pd.DataFrame({'ds': y.index})
        for col in self.regressor_names:
            if col not in X.columns:
                 raise ValueError(f"Covariate '{col}' missing from X.")
            future[col] = X[col].values

        # Predict
        forecast = self.model.predict(future)

        # Extract Effects
        # Prophet forecast dataframe has columns for each regressor.
        # We need to sum them up to get total covariate effect.
        # Alternatively, 'extra_regressors_additive' gives the sum of additive regressors.
        # If user used multiplicative regressors, they would be in 'extra_regressors_multiplicative'.
        # For this implementation, we assume additive as per standard residualization logic.

        covariate_effect = np.zeros(len(y))

        if 'extra_regressors_additive' in forecast.columns:
             covariate_effect += forecast['extra_regressors_additive'].values

        # Fallback manual sum if 'extra_regressors_additive' is missing (older versions?)
        elif self.regressor_names:
             for col in self.regressor_names:
                if col in forecast.columns:
                    covariate_effect += forecast[col].values

        if 'extra_regressors_multiplicative' in forecast.columns:
             # Just a warning or log? We don't support multiplicative decanting easily.
             # If present, it multiplies the trend.
             # Effect = Trend * (1 + multi) - Trend = Trend * multi
             # But we don't know the Trend easily without pulling it.
             # Let's try to extract it if it exists.
             multi = forecast['extra_regressors_multiplicative'].values
             trend = forecast['trend'].values
             covariate_effect += trend * multi

        # Adjusted Series = y - Covariate Effect
        adjusted = y - covariate_effect

        stats = {}

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            params=self.model.params,
            stats=stats
        )
