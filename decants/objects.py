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
    stats: Dict[str, float] = field(default_factory=dict)

    def plot(self):
        """
        Calls the unified plotter.
        Note: The unified plotter logic will be implemented in decants/utils/plotting.py.
        For now, this is a placeholder or can import the plotter if it exists.
        """
        try:
            from .utils.plotting import plot_adjustment
            return plot_adjustment(self)
        except ImportError:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("Matplotlib is not installed. Please install 'matplotlib' to use plotting features.")
                return None

            print("Plotting utility not yet implemented.")
            # Basic fallback plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

            # Top Panel: Original, Adjusted
            axes[0].plot(self.original_series, label='Original', color='black', alpha=0.5)
            axes[0].plot(self.adjusted_series, label='Adjusted', color='blue')
            axes[0].set_title("Original vs Adjusted Series")
            axes[0].legend()

            # Middle Panel: Covariate Effect
            axes[1].plot(self.covariate_effect, label='Covariate Effect', color='green')
            axes[1].set_title("Isolated Covariate Effect")
            axes[1].legend()

            # Bottom Panel: Residuals (Adjusted Series seems to be the residualized series?)
            # Usually Adjusted = Original - Covariate Effect.
            # If we want to check stationarity of Adjusted Series.
            axes[2].plot(self.adjusted_series, label='Adjusted (Residuals)', color='red', alpha=0.7)
            axes[2].set_title("Adjusted Series (Residuals)")
            axes[2].legend()

            plt.tight_layout()
            return fig
