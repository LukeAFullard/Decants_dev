import numpy as np
import pandas as pd
import warnings
import datetime

class MarginalizationMixin:
    """
    A plug-in module that adds 'Integration/Normalization' powers
    to any time-series model.
    """

    def transform_integrated(self, t, C, history_mask=None, n_samples=200, random_state=None):
        """
        Args:
            t (array): The list of timestamps we want to adjust (e.g., [Today]).
            C (array): The covariates for those times (e.g., [Today's Weather]).
            history_mask (bool array): Optional filter. E.g., "Only use history from 2010-2020".
            n_samples (int): How many historical days to replay? (More = smoother, slower).
            random_state (int, RandomState instance or None): Controls the randomness of the subsampling.

        Returns:
            y_integrated: The 'Climate Corrected' trend.
        """
        # Sanity Check for Linear Models
        if getattr(self, 'model_type', 'nonlinear') == 'linear':
            warnings.warn(
                "You are using Integration on a Linear Model (ARIMAX/Kalman/Prophet). "
                "This is computationally expensive and mathematically identical to standard subtraction. "
                "We recommend using method='clean' (subtraction) instead.",
                UserWarning
            )

        # --- 1. SETUP THE "CLIMATE" POOL ---

        # 1. Apply history mask (if any) to narrow down to "allowed history"
        # Handle list input safely
        if isinstance(C, list):
             C = np.asarray(C)

        if history_mask is not None:
            if isinstance(C, (pd.DataFrame, pd.Series)):
                 pool_C_raw = C.iloc[history_mask]
            else:
                 pool_C_raw = C[history_mask]
        else:
            pool_C_raw = C

        # 2. Convert to numpy (handling reordering if DF)
        # DEFENSIBILITY: Robustly handle DataFrame columns to prevent misalignment.
        if isinstance(pool_C_raw, pd.DataFrame):
             expected_names = getattr(self, "feature_names", None) or \
                              getattr(self, "regressor_names", None) or \
                              getattr(self, "exog_names", None)
             if expected_names:
                  missing = [c for c in expected_names if c not in pool_C_raw.columns]
                  if missing:
                       raise ValueError(f"Covariate pool missing columns: {missing}")
                  pool_C_raw = pool_C_raw[expected_names]

        pool_C = np.asarray(pool_C_raw)
        if pool_C.ndim == 1: pool_C = pool_C.reshape(-1, 1)

        # 3. Drop NaNs
        # DEFENSIBILITY: Remove NaNs from the pool.
        # Including NaNs in the historical pool causes crashes (RF) or garbage results.
        mask_valid = ~np.isnan(pool_C).any(axis=1)
        if not np.all(mask_valid):
             n_dropped = np.sum(~mask_valid)
             warnings.warn(f"Covariate pool contains {n_dropped} rows with NaN. Dropping them for integration safety.", UserWarning)
             pool_C = pool_C[mask_valid]

        if len(pool_C) == 0:
             raise ValueError("Covariate pool is empty (after masking and NaN dropping). Cannot proceed with integration.")

        # Optimization: We don't need to replay 10,000 days. 200 is usually enough.
        # Randomly pick 200 days from history to represent the "Climate".
        rng = np.random.default_rng(random_state)

        if len(pool_C) > n_samples:
            indices = rng.choice(len(pool_C), n_samples, replace=False)
            active_pool = pool_C[indices]
        else:
            active_pool = pool_C

        n_history_samples = len(active_pool)

        # Ensure t is iterable and has length
        # Handle pandas Timestamp specifically as it behaves like a scalar but sometimes wraps oddly
        if isinstance(t, (pd.Timestamp, datetime.datetime, datetime.date)) or np.isscalar(t):
             t = np.array([t])
        else:
            t = np.asarray(t)
            # Check for 0-d array (scalar wrapped in array)
            if t.ndim == 0:
                t = t.reshape(1)

        n_targets = len(t)
        y_integrated = np.zeros(n_targets)

        # --- 2. THE REPLAY LOOP ---
        # For every single day we want to adjust...
        for i in range(n_targets):
            current_t = t[i]

            # A. CREATE THE SCENARIOS
            # We want to ask the model: "What if it was Today's Time, but History's Weather?"

            # Create a column of "Today's Time" repeated n_history_samples times
            # [Today, Today, Today....]
            batch_t = np.full((n_history_samples, 1), current_t)

            # Stack it next to the Historical Weather scenarios
            # [Today | Weather_from_2010]
            # [Today | Weather_from_2011]
            # ...
            # Handle mixed types if t is datetime and C is float
            if isinstance(current_t, (pd.Timestamp, np.datetime64, datetime.date)):
                 # Force Object dtype for mixed stack
                 # We create an object array manually to avoid numpy type coercion issues
                 # Verified robust for mixed Time/Float inputs.
                 batch_X = np.empty((n_history_samples, 1 + active_pool.shape[1]), dtype=object)
                 batch_X[:, 0] = batch_t[:, 0]
                 batch_X[:, 1:] = active_pool
            else:
                 batch_X = np.hstack([batch_t, active_pool])

            # B. ASK THE MODEL
            # "Predict the outcome for all scenarios."
            # (This assumes 'self' is the Decanter model (GP/Loess) with a predict method)
            if hasattr(self, 'predict_batch'):
                preds = self.predict_batch(batch_X) # Specialized fast path
            elif hasattr(self, 'model') and hasattr(self.model, 'predict'):
                # Standard Sklearn path
                # Note: Some sklearn models expect specific shapes or types.
                # Assuming the model can handle numpy array [t, C]
                preds = self.model.predict(batch_X)
            else:
                 raise NotImplementedError("Model must implement 'predict_batch' or have a 'model.predict' method.")

            # C. AVERAGE THE RESULTS
            # This is the "Integration" step.
            y_integrated[i] = np.mean(preds)

        return y_integrated
