"""Structural protocol interfaces for PRISM subsystems.

These protocols establish explicit contracts between pipeline stages:
Detector -> Tracker -> GroundProjector -> StateEstimator -> Predictor -> RiskEvaluator -> DecisionEngine.
All protocols depend solely on PRISM core domain types, enforcing complete decoupling from
hardware, ML frameworks (YOLO, PyTorch), and computer-vision libraries (OpenCV).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.types import (
    Decision,
    ObjectObservation,
    ObjectState,
    RiskAssessment,
    Trajectory,
    Vector2,
    WorldState,
)


@runtime_checkable
class Detector(Protocol):
    """Perception detector converting raw sensory frames into image-space observations.

    Implementations may wrap YOLO, MobileNet, TFLite, or synthetic generators.
    """

    def detect(self, frame: object) -> list[ObjectObservation]:
        """Detect objects within an image frame.

        Args:
            frame: Raw image or sensor frame object (opaque to domain core).

        Returns:
            List of detected object observations in image space.
        """
        ...


@runtime_checkable
class Tracker(Protocol):
    """Multi-object tracker assigning consistent track IDs to observations over time.

    Implementations may wrap ByteTrack, SORT, or ground-truth track assignment.
    """

    def update(
        self,
        observations: list[ObjectObservation],
        timestamp: float,
    ) -> list[ObjectObservation]:
        """Update tracks with new detections from a frame.

        Args:
            observations: Detections from the current frame.
            timestamp: Monotonic timestamp of the frame in seconds.

        Returns:
            Observations with associated object_id track identifiers.
        """
        ...


@runtime_checkable
class GroundProjector(Protocol):
    """Geometric projector transforming image-space pixel coordinates to ground-plane coordinates.

    Implementations may use planar homography, camera intrinsics/extrinsics, or depth map projection.
    """

    def project(self, point: tuple[float, float]) -> Vector2 | None:
        """Project a 2D image pixel coordinate onto the 2D physical ground plane.

        Args:
            point: Image coordinate (x, y) in pixels (typically bottom-center of bbox).

        Returns:
            Vector2 on the ground plane (meters: x lateral, z forward/depth),
            or None if the point lies above the horizon or outside the valid projection field.
        """
        ...


@runtime_checkable
class EgoMotionProvider(Protocol):
    """Estimates user / camera motion velocity on the ground plane."""

    def estimate_motion(self, timestamp: float) -> Vector2 | None:
        """Estimate user ground-plane motion at the given timestamp.

        Args:
            timestamp: Monotonic timestamp in seconds.

        Returns:
            Vector2 velocity of ego motion in m/s, or None if stationary / unavailable.
        """
        ...


@runtime_checkable
class StateEstimator(Protocol):
    """Maintains and filters spatial state dynamics (position, velocity) for tracked objects.

    Implementations may use Extended Kalman Filters (EKF), Unscented Kalman Filters (UKF),
    or constant-velocity kinematics.
    """

    def update(
        self,
        observations: list[ObjectObservation],
        timestamp: float,
    ) -> WorldState:
        """Process incoming observations and return the updated world state.

        Args:
            observations: Tracked observations from the current frame.
            timestamp: Current timestamp in seconds.

        Returns:
            Updated immutable WorldState containing tracked ObjectState instances.
        """
        ...


@runtime_checkable
class Predictor(Protocol):
    """Extrapolates physical object state into future ground-plane trajectories."""

    def predict(
        self,
        state: ObjectState,
        horizon: float,
    ) -> Trajectory:
        """Predict the future trajectory for a given object state.

        Args:
            state: Current estimated physical state of the object.
            horizon: Future time window to predict across in seconds.

        Returns:
            Trajectory containing future TrajectoryPoint samples.
        """
        ...


@runtime_checkable
class RiskEvaluator(Protocol):
    """Evaluates collision threat metrics (TTC, CPA distance) and semantic risk levels."""

    def evaluate(
        self,
        state: ObjectState,
        trajectory: Trajectory,
        timestamp: float,
    ) -> RiskAssessment:
        """Assess the collision risk for an object given its state and predicted trajectory.

        Args:
            state: Current physical state of the object.
            trajectory: Projected trajectory of the object.
            timestamp: Current evaluation timestamp in seconds.

        Returns:
            RiskAssessment detailing risk level, score, TTC, and CPA distance.
        """
        ...


@runtime_checkable
class DecisionEngine(Protocol):
    """Arbitrates risk assessments into actionable user alerts and system decisions."""

    def decide(self, risk_assessment: RiskAssessment) -> Decision:
        """Determine what alert or action should be taken based on risk assessment.

        Args:
            risk_assessment: Evaluated risk for a candidate hazard.

        Returns:
            Decision specifying the alert action (NO_ACTION, CAUTION, WARNING, STOP).
        """
        ...
