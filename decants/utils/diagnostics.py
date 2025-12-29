import numpy as np
import pandas as pd
from typing import Tuple, Dict

def pearson_correlation(s1: pd.Series, s2: pd.Series) -> float:
    """
    Calculate Pearson correlation between two series.
    """
    # Align and drop NaNs
    common_idx = s1.dropna().index.intersection(s2.dropna().index)
    if len(common_idx) < 2:
        return np.nan
    return s1.loc[common_idx].corr(s2.loc[common_idx])

def variance_reduction(original: pd.Series, adjusted: pd.Series) -> float:
    """
    Calculate percentage variance reduction.
    1 - Var(Adjusted) / Var(Original)
    """
    common_idx = original.dropna().index.intersection(adjusted.dropna().index)
    if len(common_idx) < 2:
        return np.nan

    var_orig = original.loc[common_idx].var()
    var_adj = adjusted.loc[common_idx].var()

    if var_orig == 0:
        return 0.0

    return 1.0 - (var_adj / var_orig)
