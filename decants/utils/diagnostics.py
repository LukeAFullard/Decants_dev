import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

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

def check_autocorrelation(series: pd.Series, lags: int = 20) -> Dict[str, Any]:
    """
    Check for autocorrelation in a time series using ACF and Ljung-Box test.
    Returns a dictionary containing ACF values and Ljung-Box test statistics.
    """
    series_clean = series.dropna()

    # Calculate ACF
    # fft=True is standard for performance
    acf_values = acf(series_clean, nlags=lags, fft=True)

    # Ljung-Box Test
    # Returns a dataframe
    lb_test = acorr_ljungbox(series_clean, lags=[lags], return_df=True)
    lb_stat = lb_test['lb_stat'].iloc[0]
    lb_pvalue = lb_test['lb_pvalue'].iloc[0]

    return {
        'acf': acf_values,
        'lb_stat': lb_stat,
        'lb_pvalue': lb_pvalue
    }
