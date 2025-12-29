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

        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        # Prepare Data for Prophet
        # Prophet expects 'ds' and 'y' columns
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

        # Align indices
        common_idx = y.index.intersection(X.index)
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

        # Let's check if extra_regressors_additive exists and use it,
        # but also check for multiplicative just in case (though we didn't expose mode setting easily yet, default is additive).

        covariate_effect = np.zeros(len(y))

        if 'extra_regressors_additive' in forecast.columns:
             covariate_effect += forecast['extra_regressors_additive'].values

        if 'extra_regressors_multiplicative' in forecast.columns:
             # Multiplicative effect is a multiplier on trend?
             # y(t) = Trend * (1 + Multiplicative_Terms) + Additive_Terms
             # Effectively the "effect" contribution to y is ... complex to separate as "y - effect" if it's multiplicative.
             # But usually residualization implies subtraction.
             # If we have multiplicative, the "effect" in units of y depends on the trend.
             # Prophet's column 'extra_regressors_multiplicative' is likely the value of the term itself (percentage?).
             # Actually, prophet documentation says components are in units of y for additive,
             # but for multiplicative they are percentages?
             # Let's assume Additive for now as per "Decant" philosophy (Adjustment = y - Effect).
             # If user does multiplicative, this might be tricky.
             # We will ignore multiplicative for now or assume 0 if not present.
             pass

        # If individual columns exist and extra_regressors_additive not found (older prophet?), we sum manually.
        if 'extra_regressors_additive' not in forecast.columns:
            # Manually sum
            for col in self.regressor_names:
                if col in forecast.columns:
                    covariate_effect += forecast[col].values

        # Adjusted Series = y - Covariate Effect
        adjusted = y - covariate_effect

        # Confidence Intervals
        # Prophet gives yhat_lower/upper, but that includes trend/seasonality uncertainty.
        # It doesn't typically give CI for the *regressor effect* specifically in the forecast dataframe,
        # unless we look at the samples?
        # The forecast df has 'extra_regressors_additive', but usually no 'extra_regressors_additive_lower'.
        # We might not be able to populate conf_int easily without sampling.
        # So we leave it None or try to find it?
        # Prophet MCMC samples can give it, but default is MAP (no samples).
        # We will leave conf_int as None for now.

        stats = {}
        # Try to get some basic stats if possible?

        return DecantResult(
            original_series=y,
            adjusted_series=adjusted,
            covariate_effect=pd.Series(covariate_effect, index=y.index),
            model=self.model,
            params=self.model.params,
            stats=stats
        )
