import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted
import datetime

from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin
from decants.utils.time import prepare_time_feature

class GPDecanter(BaseDecanter, MarginalizationMixin):
    """
    Gaussian Process Decanter for robust removal of covariate effects
    in irregularly sampled time series.

    Methodology: Non-Parametric Bayesian Regression (Kriging)
    Primary Use Case: High-precision adjustment of irregularly sampled time series
    with complex, unknown, non-linear covariate relationships.
    """
    def __init__(self, kernel_nu: float = 1.5, normalize_y: bool = True, random_state: int = 42, strict: bool = False):
        super().__init__(strict=strict)
        self.kernel_nu = kernel_nu
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.model: Optional[GaussianProcessRegressor] = None
        self.X_scaler = RobustScaler()
        self.alpha_ = None
        self._t_start = None

    def _build_kernel(self, n_covariates: int):
        """
        Constructs an Additive Kernel: Trend + Covariates + Noise
        Using Matern kernel for roughness handling.
        """
        # 1. Time Trend Kernel (Matern 3/2 for flexibility)
        # Length scale bounds allow it to find long-term trends
        # We increase constant_value_bounds to allow for high variance if data is not perfectly scaled
        k_time = ConstantKernel(constant_value_bounds=(1e-5, 1e6)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e4), nu=self.kernel_nu)

        # 2. Covariate Kernel (One shared kernel for simplicity, or separate if dims are high)
        # We assume covariates interact smoothly.
        k_cov = ConstantKernel(constant_value_bounds=(1e-5, 1e6)) * Matern(length_scale=np.ones(n_covariates), length_scale_bounds=(1e-1, 1e4), nu=self.kernel_nu)

        # 3. Noise Kernel (Handles measurement error)
        k_noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

        return k_time + k_cov + k_noise

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "GPDecanter":
        """
        Fit the Gaussian Process model.

        Args:
            y: Target series with time index.
            X: Covariate(s) with time index (aligned with y).
        """
        self._ensure_audit_log()

        common_idx = self._validate_alignment(y, X)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]

        self._log_event("fit_start", {"n_samples": len(y_aligned)})

        t, self._t_start = prepare_time_feature(common_idx, self._t_start)

        # Ensure X is 2D
        if isinstance(X_aligned, pd.Series):
            X_aligned = X_aligned.to_frame()

        C = X_aligned.values

        # Data Prep
        self.train_t = t.reshape(-1, 1)
        self.train_C = self.X_scaler.fit_transform(C) # Robust scaling for outliers
        self.train_y = y_aligned.values

        # Combine Time and Covariates for fitting: X_full = [t, C]
        X_full = np.hstack([self.train_t, self.train_C])

        # Initialize Kernel & Model
        kernel = self._build_kernel(n_covariates=self.train_C.shape[1])

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0, # We use WhiteKernel for noise instead of alpha parameter
            normalize_y=self.normalize_y, # Handles mean-centering
            n_restarts_optimizer=3, # Avoid local minima
            random_state=self.random_state
        )

        self.model.fit(X_full, self.train_y)

        # Extract the learned weights (alpha) for manual decomposition
        self.alpha_ = self.model.alpha_

        self._log_event("fit_complete", {"kernel": str(self.model.kernel_)})

        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Decants the covariate effect.
        """
        self._ensure_audit_log()
        check_is_fitted(self.model)

        common_idx = self._validate_alignment(y, X)
        # Note: In standard decants usage, we often decant the same series we fitted on,
        # or a new series. Ideally we should use the index from the input.

        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]

        if isinstance(X_aligned, pd.Series):
            X_aligned = X_aligned.to_frame()

        t, _ = prepare_time_feature(common_idx, self._t_start)
        C = X_aligned.values

        # Prep inputs
        t_in = t.reshape(-1, 1)
        C_in = self.X_scaler.transform(C)

        # --- THE DECOMPOSITION TRICK ---

        # 1. Predict Total (Time + Covariates)
        X_full = np.hstack([t_in, C_in])
        pred_total, std_total = self.model.predict(X_full, return_std=True)

        # 2. Predict Counterfactual (Time + Zero Covariates)
        # Since we use RobustScaler, 0.0 is the median (center).
        C_neutral = np.zeros_like(C_in)
        X_neutral = np.hstack([t_in, C_neutral])

        pred_trend_only = self.model.predict(X_neutral)

        # 3. Isolate Effect
        # Effect = Total_Prediction - Trend_Prediction
        covariate_effect = pred_total - pred_trend_only

        # 4. Decant
        # We subtract the *modeled* effect from the *original* data
        y_adjusted = y_aligned.values - covariate_effect

        result = DecantResult(
            original_series=y_aligned,
            adjusted_series=pd.Series(y_adjusted, index=common_idx, name=y.name),
            covariate_effect=pd.Series(covariate_effect, index=common_idx, name="covariate_effect"),
            model=self.model, # Added model
            stats={
                "uncertainty": pd.Series(std_total, index=common_idx, name="uncertainty"),
                "kernel_params": self.model.kernel_.get_params()
            }
        )

        return result

    def get_model_params(self) -> Dict[str, Any]:
        """Return fitted parameters for the audit trail."""
        if self.model is None:
            return {}

        # Convert kernel params to something serializable if needed,
        # but get_params() usually returns simple types or objects.
        # We'll rely on json default serializer in base.
        return {
            "kernel": str(self.model.kernel_),
            "kernel_params": self.model.kernel_.get_params(),
            "kernel_nu": self.kernel_nu,
            "normalize_y": self.normalize_y,
            "log_marginal_likelihood": self.model.log_marginal_likelihood_value_
                if hasattr(self.model, "log_marginal_likelihood_value_") else None
        }

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Helper for MarginalizationMixin to enable Monte Carlo integration.

        Args:
            X (np.ndarray): Input batch of shape [n_samples, 1 + n_features].
                            Column 0 is Time (will be processed relative to _t_start).
                            Columns 1: are Covariates (will be scaled).
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # X is passed from mixin as np.array (possibly object if datetime)
        # X[:, 0] is t, X[:, 1:] is C.

        X_t = X[:, 0]
        X_c = X[:, 1:]

        # 1. Prepare Time
        # Robust check for datetime-like objects in the array
        is_datetime = False
        if X_t.dtype == object and len(X_t) > 0:
             # Check the first element
             first_elem = X_t[0]
             if isinstance(first_elem, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
                  is_datetime = True
             elif isinstance(first_elem, (int, np.integer)) and self._t_start is not None:
                  if first_elem > 1e16:
                       is_datetime = True
        elif np.issubdtype(X_t.dtype, np.datetime64):
             is_datetime = True

        if is_datetime:
             idx = pd.to_datetime(X_t)
             # Reuse prepare_time_feature but handle the fact it expects an Index
             numeric_t, _ = prepare_time_feature(idx, self._t_start)
        else:
             numeric_t = X_t.astype(float)

        # 2. Prepare Covariates
        # Explicitly cast to float and Scale using the stored scaler
        C_raw = X_c.astype(float)
        C_scaled = self.X_scaler.transform(C_raw)

        # 3. Combine
        t_in = numeric_t.reshape(-1, 1)
        X_full = np.hstack([t_in, C_scaled])

        # 4. Predict
        return self.model.predict(X_full)
