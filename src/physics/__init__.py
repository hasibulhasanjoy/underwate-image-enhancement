"""
src/physics — Physics-based prior estimation for P-UWDM conditioning.

Exports
-------
AmbientLightEstimator    — estimates global ambient light A ∈ ℝ³
TransmissionEstimator    — estimates per-pixel transmission t(x) ∈ [0,1]
DegradationEstimator     — computes 6-dim degradation feature vector
DegradationFeatures      — output dataclass from DegradationEstimator
AmbientConfig
TransmissionConfig
DegradationConfig
"""

from .ambient import (
    AmbientConfig,
    AmbientLightEstimator,
    estimate_ambient_batch,
)
from .transmission import (
    TransmissionConfig,
    TransmissionEstimator,
    estimate_transmission_batch,
)
from .degradation import (
    DegradationConfig,
    DegradationEstimator,
    DegradationFeatures,
    estimate_degradation_batch,
)

__all__ = [
    "AmbientConfig",
    "AmbientLightEstimator",
    "estimate_ambient_batch",
    "TransmissionConfig",
    "TransmissionEstimator",
    "estimate_transmission_batch",
    "DegradationConfig",
    "DegradationEstimator",
    "DegradationFeatures",
    "estimate_degradation_batch",
]
