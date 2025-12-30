from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import pandas as pd

@dataclass
class DecantResult:
    """
    Standardized object returned by every decant method.
    """
    original_series: pd.Series
    adjusted_series: pd.Series
    covariate_effect: pd.Series
    model: Any
    params: Dict[str, Any] = field(default_factory=dict)
    conf_int: Optional[pd.DataFrame] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def plot(self):
        """
        Calls the unified plotter.
        """
        try:
            from decants.utils.plotting import plot_adjustment
            return plot_adjustment(self)
        except ImportError as e:
            # Check if it's a missing dependency
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("Matplotlib is not installed. Please install 'matplotlib' to use plotting features.")
                return None

            # If matplotlib is installed but the import failed for another reason, re-raise
            raise e
