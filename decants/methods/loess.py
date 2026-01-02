import numpy as np
import pandas as pd
import datetime
from typing import Optional, Union, Dict, Any, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RegularGridInterpolator

from decants.base import BaseDecanter
from decants.objects import DecantResult
from decants.integration import MarginalizationMixin
from decants.utils.time import prepare_time_feature

class FastLoessDecanter(BaseDecanter, MarginalizationMixin):
    """
    FastLoessDecanter: Implements WRTDS-style covariate adjustment using
    grid-based local regression (Multivariate LOESS).

    Methodology: Multivariate Weighted Local Regression (Generalized WRTDS)
    Primary Use Case: Empirical, "Assumption-Free" adjustment of data where
    relationships are complex, non-monotonic, or prone to regime shifts.
    """
    def __init__(self, span: float = 0.3, grid_resolution: int = 50, degree: int = 1):
        super().__init__()
        self.span = span  # Percentage of data to use as neighbors (0.1 to 1.0)
        self.grid_res = grid_resolution # Size of the interpolation grid (50x50)
        self.degree = degree # 1 = Linear local fit
        self.scaler = StandardScaler()
        self.interpolator = None
        self.baseline_c = 0.0 # Effect is relative to this baseline
        self._t_start = None

    def _tricube_weights(self, distances: np.ndarray) -> np.ndarray:
        """Standard LOESS weighting function: (1 - u^3)^3"""
        # Normalize distances to 0-1 range within the neighborhood
        max_dist = np.max(distances)
        if max_dist == 0: return np.ones_like(distances)
        u = distances / max_dist
        return (1 - u**3)**3

    def fit(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series], **kwargs) -> "FastLoessDecanter":
        """
        Builds the correction surface.
        """
        self._ensure_audit_log()

        common_idx = self._validate_alignment(y, X)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]

        self._log_event("fit_start", {"n_samples": len(y_aligned)})

        t, self._t_start = prepare_time_feature(common_idx, self._t_start)

        if isinstance(X_aligned, pd.Series):
            X_aligned = X_aligned.to_frame()

        C = X_aligned.values

        # 1 covariate restriction
        if C.shape[1] > 1:
             raise ValueError("FastLoessDecanter currently supports only 1 covariate column.")

        self.t_train = np.array(t).reshape(-1, 1) # Assign self.t_train
        self.c_train = np.array(C)
        self.c_train = self.c_train.reshape(-1, 1)
        self.y_train = np.array(y_aligned.values)
        self.baseline_c = np.median(self.c_train) # Default baseline is median

        # Combine and Scale for distance calculations
        self.X_train = np.hstack([self.t_train, self.c_train])
        self.X_scaled = self.scaler.fit_transform(self.X_train)

        # 2. Define the Grid (The "Skeleton")
        # We create a mesh covering the min/max of time and covariates
        t_min, t_max = self.t_train.min(), self.t_train.max()
        c_min, c_max = self.c_train.min(), self.c_train.max()

        # Add small buffer to grid to cover edges
        t_pad = (t_max - t_min) * 0.05
        if t_pad == 0: t_pad = 1.0 # Handle single time point

        c_pad = (c_max - c_min) * 0.05
        if c_pad == 0: c_pad = 1.0 # Handle constant covariate

        t_grid = np.linspace(t_min - t_pad, t_max + t_pad, self.grid_res)
        c_grid = np.linspace(c_min - c_pad, c_max + c_pad, self.grid_res)

        # 3. Fit Local Models at Grid Nodes
        n_neighbors = int(self.span * len(self.y_train))
        n_neighbors = max(n_neighbors, self.degree + 2) # Ensure enough points for regression

        if n_neighbors > len(self.y_train):
             raise ValueError(f"Not enough data for FastLoess with degree={self.degree}. Need at least {n_neighbors} samples, got {len(self.y_train)}.")

        nbrs_engine = NearestNeighbors(n_neighbors=n_neighbors).fit(self.X_scaled)

        # Changed: Store Predicted Y instead of Effect to enable Integration
        grid_preds = np.zeros((self.grid_res, self.grid_res))

        # Loop through grid nodes (This is the heavy lifting, done only once)
        for i, t_val in enumerate(t_grid):
            for j, c_val in enumerate(c_grid):

                # Scale the query point
                query_point = np.array([[t_val, c_val]])
                query_scaled = self.scaler.transform(query_point)

                # Find neighbors
                dists, indices = nbrs_engine.kneighbors(query_scaled)
                indices = indices[0]
                dists = dists[0]

                # Get Local Data
                X_local = self.X_train[indices] # Unscaled for regression
                y_local = self.y_train[indices]
                weights = self._tricube_weights(dists)

                # Fit Weighted Linear Regression
                # Model: y = b0 + b1*t + b2*c
                reg = LinearRegression()
                try:
                    reg.fit(X_local, y_local, sample_weight=weights)

                    # Calculate Predicted Y at this Grid Node
                    prediction = reg.predict(query_point)[0]
                    grid_preds[i, j] = prediction
                except Exception:
                    # Fallback if fit fails (e.g. singular matrix): Use Weighted Mean of local neighborhood.
                    # This preserves surface continuity better than global mean or zero.
                    if len(y_local) > 0:
                         # Normalize weights for average
                         w_sum = np.sum(weights)
                         if w_sum > 0:
                            grid_preds[i, j] = np.average(y_local, weights=weights)
                         else:
                            grid_preds[i, j] = np.mean(y_local)
                    else:
                         # Should not happen if n_neighbors > 0, but as last resort
                         grid_preds[i, j] = np.mean(self.y_train)

        # 4. Create Interpolator
        # We use nearest extrapolation to avoid crashing at edges if we are slightly outside grid,
        # but the local regression handles the trend logic.
        # RegularGridInterpolator requires grid points to be strictly ascending.
        self.interpolator = RegularGridInterpolator(
            (t_grid, c_grid), grid_preds,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        self._log_event("fit_complete", {"grid_res": self.grid_res, "span": self.span})
        return self

    def transform(self, y: pd.Series, X: Union[pd.DataFrame, pd.Series]) -> DecantResult:
        """
        Applies the correction surface to the data.
        """
        self._ensure_audit_log()

        if self.interpolator is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        common_idx = self._validate_alignment(y, X)
        y_aligned = y.loc[common_idx]
        X_aligned = X.loc[common_idx]

        if isinstance(X_aligned, pd.Series):
            X_aligned = X_aligned.to_frame()

        t, _ = prepare_time_feature(common_idx, self._t_start)
        C = X_aligned.values

        # 1 covariate restriction
        if C.shape[1] > 1:
             raise ValueError("FastLoessDecanter currently supports only 1 covariate column.")

        # Look up the predicted Y for every actual data point
        points = np.column_stack((t, C.ravel()))
        estimated_preds = self.interpolator(points)

        # Look up the predicted Y for Baseline Scenario
        baseline_points = np.column_stack((t, np.full(len(t), self.baseline_c)))
        baseline_preds = self.interpolator(baseline_points)

        # Handle NaNs
        if np.isnan(estimated_preds).any():
             estimated_preds = np.nan_to_num(estimated_preds)
        if np.isnan(baseline_preds).any():
             baseline_preds = np.nan_to_num(baseline_preds)

        # Effect = Pred(Actual) - Pred(Baseline)
        estimated_effects = estimated_preds - baseline_preds

        # Decant
        y_adjusted = y_aligned.values - estimated_effects

        result = DecantResult(
            original_series=y_aligned,
            adjusted_series=pd.Series(y_adjusted, index=common_idx, name=y.name),
            covariate_effect=pd.Series(estimated_effects, index=common_idx, name="covariate_effect"),
            model=self.interpolator,
            stats={
                "baseline_c": self.baseline_c,
                "span": self.span,
                "grid_res": self.grid_res
            }
        )

        return result

    def get_model_params(self) -> Dict[str, Any]:
        """Return fitted parameters for the audit trail."""
        return {
            "span": self.span,
            "grid_resolution": self.grid_res,
            "degree": self.degree,
            "baseline_c": self.baseline_c,
            # We could serialize grid_values if we wanted, but it might be large.
            # "grid_values": self.interpolator.values.tolist() if self.interpolator else None
        }

    def predict_batch(self, X):
        """
        Helper for MarginalizationMixin.
        Returns the full predicted Y from the interpolated surface.
        Args:
            X: [Time, Covariates]
        """
        if self.interpolator is None:
             raise RuntimeError("Model is not fitted. Call fit() first.")

        # X is passed from mixin as np.array (possibly object if datetime)
        # X[:, 0] is t, X[:, 1:] is C.

        X_t = X[:, 0]
        X_c = X[:, 1:]

        # Ensure Time is numeric relative to t_start
        if isinstance(X_t[0], (pd.Timestamp, np.datetime64, datetime.datetime, datetime.date)):
             idx = pd.Index(X_t)
             numeric_t, _ = prepare_time_feature(idx, self._t_start)
        else:
             numeric_t = X_t.astype(float) # Ensure float

        # Explicitly cast covariates to float to avoid object-dtype errors in interpolator
        # This handles cases where mixed types in stack caused object array
        numeric_c = X_c.astype(float)

        # Re-stack
        X_final = np.column_stack([numeric_t, numeric_c])

        return self.interpolator(X_final)
