"""Perception pipeline orchestrating frame detection, observation extraction, and rejection logging."""

from __future__ import annotations

from typing import Sequence

from app.perception.detector import Detector
from app.perception.model import (
    Detection,
    DetectionRejection,
    PhysicalObservation,
)
from app.perception.observation import ObservationExtractor
from app.perception.uncertainty import MeasurementUncertaintyModel
from app.state.projection import GroundPlaneProjector


class PerceptionPipeline:
    """End-to-end perception orchestrator bridging 2D visual detectors and physical ground-plane observations."""

    def __init__(
        self,
        detector: Detector,
        projector: GroundPlaneProjector,
        uncertainty_model: MeasurementUncertaintyModel | None = None,
        min_confidence: float = 0.20,
    ) -> None:
        self.detector = detector
        self.extractor = ObservationExtractor(
            projector=projector,
            uncertainty_model=uncertainty_model,
            min_confidence=min_confidence,
        )

    def process_frame(
        self,
        frame: object,
        timestamp: float,
    ) -> tuple[list[PhysicalObservation], list[DetectionRejection]]:
        """Run detector on a frame and convert valid detections into physical observations.

        Args:
            frame: Camera frame or synthetic scene representation.
            timestamp: Monotonic frame capture time in seconds.

        Returns:
            (valid_observations, rejections) tuple.
        """
        raw_detections = self.detector.detect(frame, timestamp)
        valid_obs: list[PhysicalObservation] = []
        rejections: list[DetectionRejection] = []

        for det in raw_detections:
            result = self.extractor.extract_observation(det)
            if isinstance(result, PhysicalObservation):
                valid_obs.append(result)
            elif isinstance(result, DetectionRejection):
                rejections.append(result)

        return valid_obs, rejections

    def process_frame_observations(
        self,
        frame: object,
        timestamp: float,
    ) -> list[PhysicalObservation]:
        """Convenience method returning only valid PhysicalObservation objects."""
        obs, _ = self.process_frame(frame, timestamp)
        return obs
