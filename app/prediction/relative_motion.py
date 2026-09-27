"""Relative motion formulation between user (ego) and obstacles.

PRISM collision risk depends fundamentally on relative motion (p_rel, v_rel)
rather than absolute obstacle velocity alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.types import ObjectState, Vector2


@dataclass(frozen=True)
class EgoMotionState:
    """Ground-plane kinematic state of the user/ego agent.

    Attributes:
        position: 2D ground-plane position Vector2(X, Z) in meters.
        velocity: 2D ground-plane velocity Vector2(Vx, Vz) in m/s.
        timestamp: Timestamp of the state in seconds.
    """

    position: Vector2
    velocity: Vector2
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.position, Vector2):
            raise TypeError(f"position must be Vector2, got {type(self.position)}")
        if not isinstance(self.velocity, Vector2):
            raise TypeError(f"velocity must be Vector2, got {type(self.velocity)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class RelativeMotion:
    """Relative kinematics of an obstacle with respect to the user.

    Attributes:
        relative_position: p_rel = p_object - p_ego (meters).
        relative_velocity: v_rel = v_object - v_ego (m/s).
    """

    relative_position: Vector2
    relative_velocity: Vector2

    def __post_init__(self) -> None:
        if not isinstance(self.relative_position, Vector2):
            raise TypeError(f"relative_position must be Vector2, got {type(self.relative_position)}")
        if not isinstance(self.relative_velocity, Vector2):
            raise TypeError(f"relative_velocity must be Vector2, got {type(self.relative_velocity)}")

    @property
    def distance(self) -> float:
        """Instantaneous Euclidean distance between ego and object in meters."""
        return self.relative_position.magnitude

    @property
    def relative_speed(self) -> float:
        """Magnitude of relative velocity in m/s."""
        return self.relative_velocity.magnitude

    @property
    def is_approaching(self) -> bool:
        """Check whether the relative distance is currently decreasing.

        Condition: p_rel · v_rel < 0.
        """
        dot_product = self.relative_position.x * self.relative_velocity.x + self.relative_position.z * self.relative_velocity.z
        return dot_product < 0.0

    @property
    def closing_speed(self) -> float:
        """Rate at which distance is decreasing in m/s (positive when approaching)."""
        dist = self.distance
        if dist <= 0.0:
            return 0.0
        dot_product = self.relative_position.x * self.relative_velocity.x + self.relative_position.z * self.relative_velocity.z
        return -dot_product / dist


def compute_relative_motion(
    ego_position: Vector2,
    ego_velocity: Vector2,
    object_position: Vector2,
    object_velocity: Vector2,
) -> RelativeMotion:
    """Compute relative motion vectors between ego and object positions/velocities."""
    p_rel = object_position - ego_position
    v_rel = object_velocity - ego_velocity
    return RelativeMotion(
        relative_position=p_rel,
        relative_velocity=v_rel,
    )


def compute_relative_motion_from_states(
    ego_state: EgoMotionState | ObjectState,
    object_state: ObjectState,
) -> RelativeMotion:
    """Compute relative motion from state instances."""
    return compute_relative_motion(
        ego_position=ego_state.position,
        ego_velocity=ego_state.velocity,
        object_position=object_state.position,
        object_velocity=object_state.velocity,
    )
