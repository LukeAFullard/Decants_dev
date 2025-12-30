import pandas as pd
import numpy as np
import time
from typing import List, Dict, Union, Type, Any
from decants.base import BaseDecanter
from decants.objects import DecantResult

class DecantBenchmarker:
    """
    Utility to run multiple Decanter models on the same dataset and compare their performance.
    """

    def __init__(self):
        self.results = {}
        self.summary = []

    def benchmark(self,
                  models: Dict[str, BaseDecanter],
                  y: pd.Series,
                  X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Run the provided models on y and X and return a summary table.

        Args:
            models (Dict[str, BaseDecanter]): Dictionary mapping names to instantiated Decanter objects.
            y (pd.Series): Target time series.
            X (pd.DataFrame): Covariates.

        Returns:
            pd.DataFrame: Summary statistics for each model.
        """
        self.results = {}
        self.summary = []

        for name, model in models.items():
            start_time = time.time()
            error = None
            result = None

            try:
                # We assume the user passes fresh instances or we clone them?
                # BaseDecanter doesn't strictly support sklearn.clone perfectly if it has complex init.
                # We assume the user instantiates them.

                # Fit and Transform
                result = model.fit_transform(y, X)
                self.results[name] = result

            except Exception as e:
                error = str(e)
                # Still log the failure

            elapsed = time.time() - start_time

            # Collect Stats
            row = {
                "Model": name,
                "Status": "Success" if error is None else "Failed",
                "Execution Time (s)": round(elapsed, 4),
                "Error": error
            }

            if result:
                # Variance Reduction
                if "variance_reduction" in result.stats:
                     row["Variance Reduction"] = result.stats["variance_reduction"]
                else:
                    # Calculate if not present (should be in stats usually for DoubleML, but we can calc manually for others)
                    # Use diagnostics utils if needed, or simple calc
                    try:
                        var_orig = result.original_series.var()
                        var_adj = result.adjusted_series.var()
                        row["Variance Reduction"] = 1 - (var_adj / var_orig)
                    except:
                        row["Variance Reduction"] = np.nan

                # Orthogonality (if available)
                if "orthogonality" in result.stats:
                    ortho = result.stats["orthogonality"]
                    if isinstance(ortho, dict):
                        row["Max Abs Corr (Residuals vs X)"] = ortho.get("max_abs_corr", np.nan)
                    else:
                        row["Max Abs Corr (Residuals vs X)"] = np.nan

                # Model Specifics
                row["AIC"] = result.stats.get("AIC", np.nan)
                row["BIC"] = result.stats.get("BIC", np.nan)

            self.summary.append(row)

        return pd.DataFrame(self.summary).set_index("Model")
