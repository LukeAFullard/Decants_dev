import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

from decants.base import BaseDecanter
from decants.objects import DecantResult

class GPDecanter(BaseDecanter):
    """
    Gaussian Process Decanter for robust removal of covariate effects
    in irregularly sampled time series.

    Methodology: Non-Parametric Bayesian Regression (Kriging)
    Primary Use Case: High-precision adjustment of irregularly sampled time series
    with complex, unknown, non-linear covariate relationships.
    """
    def __init__(self, kernel_nu: float = 1.5, normalize_y: bool = True, random_state: int = 42):
        super().__init__()
        self.kernel_nu = kernel_nu
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.model: Optional[GaussianProcessRegressor] = None
        self.X_scaler = RobustScaler()
        self.alpha_ = None
        self._t_start = None

    def _prepare_time(self, index: pd.Index) -> np.ndarray:
        """Converts index to numeric representation."""
        if pd.api.types.is_datetime64_any_dtype(index):
            # For transform, we need to respect the training start time if we want consistency
            # However, for simple regression on time distance, we just need a consistent reference.
            # If self._t_start is set (during fit), use it.
            # But wait, 'fit' sets it. 'transform' uses it.
            if self._t_start is None:
                 # Should only happen during fit if called first, or if we are lenient.
                 # But actually _prepare_time is called in fit first.
                 pass

            ref_time = self._t_start if self._t_start is not None else index.min()

            # Convert to days since reference
            return (index - ref_time).total_seconds().to_numpy() / (24 * 3600)
        else:
            # Assume numeric
            return index.to_numpy(dtype=float)

    def _build_kernel(self, n_covariates: int):
        """
        Constructs an Additive Kernel: Trend + Covariates + Noise
        Using Matern kernel for roughness handling.
        """
        # 1. Time Trend Kernel (Matern 3/2 for flexibility)
        # Length scale bounds allow it to find long-term trends
        k_time = ConstantKernel() * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e4), nu=self.kernel_nu)

        # 2. Covariate Kernel (One shared kernel for simplicity, or separate if dims are high)
        # We assume covariates interact smoothly.
        k_cov = ConstantKernel() * Matern(length_scale=np.ones(n_covariates), length_scale_bounds=(1e-1, 1e4), nu=self.kernel_nu)

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

        if pd.api.types.is_datetime64_any_dtype(common_idx):
            self._t_start = common_idx.min()

        t = self._prepare_time(common_idx)

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

        t = self._prepare_time(common_idx)
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
