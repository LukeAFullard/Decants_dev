import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union
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
    if len(series_clean) < 2:
        return {
            'acf': np.nan,
            'lb_stat': np.nan,
            'lb_pvalue': np.nan
        }

    # Calculate ACF
    # fft=True is standard for performance
    try:
        acf_values = acf(series_clean, nlags=min(lags, len(series_clean)-1), fft=True)

        # Ljung-Box Test
        # Returns a dataframe
        lb_test = acorr_ljungbox(series_clean, lags=[min(lags, len(series_clean)-1)], return_df=True)
        lb_stat = lb_test['lb_stat'].iloc[0]
        lb_pvalue = lb_test['lb_pvalue'].iloc[0]
    except Exception:
        # Fallback for very short series
        return {
            'acf': np.nan,
            'lb_stat': np.nan,
            'lb_pvalue': np.nan
        }

    return {
        'acf': acf_values,
        'lb_stat': lb_stat,
        'lb_pvalue': lb_pvalue
    }

def check_orthogonality(residuals: pd.Series, covariates: Union[pd.DataFrame, pd.Series]) -> Dict[str, float]:
    """
    Check orthogonality between the cleaned residuals and the covariates.
    Calculates the maximum absolute correlation between residuals and any covariate.

    Args:
        residuals (pd.Series): The cleaned signal (s_t).
        covariates (pd.DataFrame or pd.Series): The covariates (X_t).

    Returns:
        Dict: {'max_abs_corr': float, 'mean_abs_corr': float}
    """
    if isinstance(covariates, pd.Series):
        covariates = covariates.to_frame()

    # Align
    common_idx = residuals.dropna().index.intersection(covariates.dropna().index)
    if len(common_idx) < 2:
         return {'max_abs_corr': np.nan, 'mean_abs_corr': np.nan}

    res_aligned = residuals.loc[common_idx]
    cov_aligned = covariates.loc[common_idx]

    correlations = []
    for col in cov_aligned.columns:
        corr = res_aligned.corr(cov_aligned[col])
        correlations.append(abs(corr))

    if not correlations:
        return {'max_abs_corr': 0.0, 'mean_abs_corr': 0.0}

    return {
        'max_abs_corr': np.nanmax(correlations),
        'mean_abs_corr': np.nanmean(correlations)
    }
