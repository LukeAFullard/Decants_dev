import pandas as pd
import numpy as np
import datetime
from typing import Optional, Union, Tuple

def prepare_time_feature(index: pd.Index, t_start: Optional[pd.Timestamp] = None) -> Tuple[np.ndarray, Optional[pd.Timestamp]]:
    """
    Converts a pandas Index (Datetime or Numeric) into a numeric time feature array.
    If the index is Datetime, it converts to fractional days since t_start (or min(index)).

    Args:
        index (pd.Index): The index to convert.
        t_start (pd.Timestamp, optional): The reference start time. If None, uses index.min().

    Returns:
        Tuple[np.ndarray, Optional[pd.Timestamp]]:
            - The numeric time feature array (float).
            - The t_start used (useful for storing in the model for future transforms).
    """
    # Check for Datetime (pandas standard or object-dtype containing dates)
    is_dt = pd.api.types.is_datetime64_any_dtype(index)

    # If object type, sample check for date objects
    if not is_dt and index.dtype == 'object' and len(index) > 0:
        # Check first non-null element
        first_valid = index.dropna()[0] if not index.dropna().empty else None
        if isinstance(first_valid, (datetime.date, datetime.datetime, pd.Timestamp)):
            try:
                index = pd.to_datetime(index)
                is_dt = True
            except:
                pass

    if is_dt:
        if t_start is None:
            t_start = index.min()
            # Ensure t_start is compatible with index type for subtraction
            if not isinstance(t_start, pd.Timestamp):
                 t_start = pd.to_datetime(t_start)

        # Convert to days since reference
        # Using seconds / (24*3600) gives fractional days
        # We ensure index is datetime64[ns]
        if not isinstance(index, pd.DatetimeIndex):
             index = pd.to_datetime(index)

        delta = (index - t_start)

        # STRICT DEFENSIBILITY: Do not silently pass NaNs.
        # If the time index has NaTs (missing time), we cannot compute a trend.
        if delta.isna().any():
             raise ValueError("Input Index contains NaT (Missing Time). Cannot proceed safely.")

        return delta.total_seconds().to_numpy() / (24 * 3600), t_start
    else:
        # Assume numeric.
        vals = index.to_numpy(dtype=float)
        if np.isnan(vals).any():
             raise ValueError("Input Index contains NaN. Cannot proceed safely.")
        return vals, t_start
