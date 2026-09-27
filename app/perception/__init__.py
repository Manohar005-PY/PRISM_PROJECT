"""PRISM Perception Subsystem - Visual detection, ground-plane projection, and observation uncertainty."""

from app.perception.adapters.synthetic import (
    SyntheticDetector,
    SyntheticFrameSource,
)
from app.perception.adapters.yolo import YOLODetector
from app.perception.detector import Detector, FrameSource
from app.perception.metrics import (
    contact_pixel_error,
    normalized_position_error,
    position_rmse,
    valid_projection_rate,
    x_rmse,
    z_rmse,
)
from app.perception.model import (
    Detection,
    DetectionRejection,
    DetectionRejectionReason,
    PhysicalObservation,
    TrackedObservation,
)
from app.perception.observation import ObservationExtractor
from app.perception.pipeline import PerceptionPipeline
from app.perception.uncertainty import (
    MeasurementUncertaintyConfig,
    MeasurementUncertaintyModel,
)

__all__ = [
    "Detection",
    "DetectionRejection",
    "DetectionRejectionReason",
    "Detector",
    "FrameSource",
    "MeasurementUncertaintyConfig",
    "MeasurementUncertaintyModel",
    "ObservationExtractor",
    "PerceptionPipeline",
    "PhysicalObservation",
    "SyntheticDetector",
    "SyntheticFrameSource",
    "TrackedObservation",
    "YOLODetector",
    "contact_pixel_error",
    "normalized_position_error",
    "position_rmse",
    "valid_projection_rate",
    "x_rmse",
    "z_rmse",
]
