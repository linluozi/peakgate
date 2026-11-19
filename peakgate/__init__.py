"""peakgate - per-call peak allocation, measured and gated."""

from .core import Measurement
from .core import Stats
from .core import compare
from .core import measure
from .core import required_reps

__all__ = ["Measurement", "Stats", "compare", "measure", "required_reps"]
__version__ = "0.1.0"