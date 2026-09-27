"""Observation extraction: bottom-center contact point extraction, ground-plane projection, and diagnostic rejection."""

from __future__ import annotations

from typing import Sequence

from app.core.types import BoundingBox, Vector2
from app.perception.model import (
    Detection,
    DetectionRejection,
    DetectionRejectionReason,
    PhysicalObservation,
)
from app.perception.uncertainty import MeasurementUncertaintyModel
from app.state.projection import GroundPlaneProjector


class ObservationExtractor:
    """Extracts physical ground-plane observations from 2D image detections using bottom-center contact projection."""

    def __init__(
        self,
        projector: GroundPlaneProjector,
        uncertainty_model: MeasurementUncertaintyModel | None = None,
        min_confidence: float = 0.20,
    ) -> None:
        if not isinstance(projector, GroundPlaneProjector):
            raise TypeError(f"projector must be an explicit GroundPlaneProjector instance, got {type(projector)}")
        self.projector = projector
        self.uncertainty_model = uncertainty_model if uncertainty_model is not None else MeasurementUncertaintyModel()
        self.min_confidence = min_confidence

    def extract_observation(
        self,
        detection: Detection,
    ) -> PhysicalObservation | DetectionRejection:
        """Project a 2D bounding box detection onto the ground plane and assign measurement covariance R.

        Args:
            detection: 2D image detection proposal.

        Returns:
            PhysicalObservation on success, or DetectionRejection on failure.
        """
        # 1. Confidence validation
        if detection.confidence < self.min_confidence:
            return DetectionRejection(
                detection=detection,
                reason=DetectionRejectionReason.LOW_CONFIDENCE,
                detail=f"Confidence {detection.confidence:.2f} below threshold {self.min_confidence:.2f}",
            )

        # 2. Extract bottom-center contact point
        contact_u, contact_v = detection.bbox.bottom_center

        # 3. Image boundary check
        w_img = float(self.projector.config.intrinsics.image_width)
        h_img = float(self.projector.config.intrinsics.image_height)
        if contact_u < 0.0 or contact_u > w_img or contact_v < 0.0 or contact_v > h_img:
            return DetectionRejection(
                detection=detection,
                reason=DetectionRejectionReason.OUT_OF_BOUNDS,
                detail=f"Contact point ({contact_u:.1f}, {contact_v:.1f}) outside image sensor [0..{w_img:.0f}, 0..{h_img:.0f}]",
            )

        # 4. Horizon check
        if contact_v <= self.projector.horizon_v:
            return DetectionRejection(
                detection=detection,
                reason=DetectionRejectionReason.ABOVE_HORIZON,
                detail=f"Contact point v={contact_v:.1f} lies at or above horizon v_horiz={self.projector.horizon_v:.1f}",
            )

        # 5. Ground-plane projection via Part 3 Projector
        pos_ground = self.projector.project((contact_u, contact_v))
        if pos_ground is None:
            return DetectionRejection(
                detection=detection,
                reason=DetectionRejectionReason.PROJECTION_FAILED,
                detail=f"Projection failed for bottom-center pixel ({contact_u:.1f}, {contact_v:.1f})",
            )

        # 6. Depth validity validation
        if pos_ground.z <= 0.0:
            return DetectionRejection(
                detection=detection,
                reason=DetectionRejectionReason.INVALID_DEPTH,
                detail=f"Projected ground depth Z={pos_ground.z:.2f}m is non-positive",
            )

        # 7. Measurement covariance R calculation
        cov_r = self.uncertainty_model.compute_covariance(
            depth_z=pos_ground.z,
            confidence=detection.confidence,
        )

        return PhysicalObservation(
            position=pos_ground,
            measurement_covariance=cov_r,
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            detection_id=detection.detection_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
        )

    def extract_observations(
        self,
        detections: Sequence[Detection],
    ) -> tuple[list[PhysicalObservation], list[DetectionRejection]]:
        """Batch extract observations from multiple detections."""
        observations: list[PhysicalObservation] = []
        rejections: list[DetectionRejection] = []

        for det in detections:
            result = self.extract_observation(det)
            if isinstance(result, PhysicalObservation):
                observations.append(result)
            else:
                rejections.append(result)

        return observations, rejections
