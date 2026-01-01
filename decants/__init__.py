from .base import BaseDecanter
from .objects import DecantResult
from .methods.gam import GamDecanter
from .methods.prophet import ProphetDecanter
from .methods.arima import ArimaDecanter
from .methods.ml import MLDecanter
from .methods.double_ml import DoubleMLDecanter
from .methods.gaussian_process import GPDecanter
from .utils.benchmarking import DecantBenchmarker

__version__ = "0.1.0"

__all__ = [
    "BaseDecanter",
    "DecantResult",
    "GamDecanter",
    "ProphetDecanter",
    "ArimaDecanter",
    "MLDecanter",
    "DoubleMLDecanter",
    "GPDecanter",
    "DecantBenchmarker"
]
