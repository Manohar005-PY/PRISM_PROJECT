"""Analytical collision geometry, Closest Point of Approach (CPA), and Time-to-Collision (TTC).

Computes exact geometric trajectory interactions between user and obstacles
without heuristic clamping or artificial approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.types import Vector2


@dataclass(frozen=True)
class CPAResult:
    """Analytical Closest Point of Approach (CPA) calculation result.

    Attributes:
        time_to_cpa: Future time offset in seconds when closest approach occurs (>= 0.0).
        distance_at_cpa: Minimum Euclidean separation distance at CPA in meters.
        relative_position_at_cpa: Relative position Vector2(X, Z) at CPA.
        is_approaching: True iff the object is currently closing distance toward the user.
    """

    time_to_cpa: float
    distance_at_cpa: float
    relative_position_at_cpa: Vector2
    is_approaching: bool

    def __post_init__(self) -> None:
        if not isinstance(self.time_to_cpa, (int, float)) or not math.isfinite(self.time_to_cpa) or self.time_to_cpa < 0.0:
            raise ValueError(f"time_to_cpa must be a non-negative finite float, got {self.time_to_cpa!r}")
        if not isinstance(self.distance_at_cpa, (int, float)) or not math.isfinite(self.distance_at_cpa) or self.distance_at_cpa < 0.0:
            raise ValueError(f"distance_at_cpa must be a non-negative finite float, got {self.distance_at_cpa!r}")
        if not isinstance(self.relative_position_at_cpa, Vector2):
            raise TypeError(f"relative_position_at_cpa must be a Vector2, got {type(self.relative_position_at_cpa)}")
        object.__setattr__(self, "time_to_cpa", float(self.time_to_cpa))
        object.__setattr__(self, "distance_at_cpa", float(self.distance_at_cpa))


@dataclass(frozen=True)
class CollisionModel:
    """Configurable geometric circular safety boundary model.

    Effective Collision Radius R = object_radius + ego_radius + additional_margin.

    Attributes:
        object_radius: Bounding radius of the physical obstacle (meters).
        ego_radius: Safety bounding radius of the user (meters).
        additional_margin: Buffer margin for navigation comfort (meters).
    """

    object_radius: float = 0.35
    ego_radius: float = 0.30
    additional_margin: float = 0.35

    def __post_init__(self) -> None:
        for name, val in [("object_radius", self.object_radius), ("ego_radius", self.ego_radius), ("additional_margin", self.additional_margin)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got {val!r}")
        object.__setattr__(self, "object_radius", float(self.object_radius))
        object.__setattr__(self, "ego_radius", float(self.ego_radius))
        object.__setattr__(self, "additional_margin", float(self.additional_margin))

    @property
    def effective_radius(self) -> float:
        """Combined circular collision threshold distance in meters."""
        return self.object_radius + self.ego_radius + self.additional_margin


@dataclass(frozen=True)
class CollisionPrediction:
    """Structured collision prediction evaluation.

    Attributes:
        collision_predicted: True iff a physical collision is predicted within the horizon.
        ttc: Time-to-collision in seconds (None if no collision occurs within the model).
        closest_distance: Minimum separation distance at CPA in meters.
        time_to_cpa: Time offset to CPA in seconds.
        collision_radius: Effective collision boundary radius in meters.
    """

    collision_predicted: bool
    ttc: float | None
    closest_distance: float
    time_to_cpa: float
    collision_radius: float

    def __post_init__(self) -> None:
        if self.ttc is not None:
            if not isinstance(self.ttc, (int, float)) or not math.isfinite(self.ttc) or self.ttc < 0.0:
                raise ValueError(f"ttc must be a non-negative finite float or None, got {self.ttc!r}")
            object.__setattr__(self, "ttc", float(self.ttc))
        if not isinstance(self.closest_distance, (int, float)) or not math.isfinite(self.closest_distance) or self.closest_distance < 0.0:
            raise ValueError(f"closest_distance must be non-negative, got {self.closest_distance!r}")
        if not isinstance(self.time_to_cpa, (int, float)) or not math.isfinite(self.time_to_cpa) or self.time_to_cpa < 0.0:
            raise ValueError(f"time_to_cpa must be non-negative, got {self.time_to_cpa!r}")
        if not isinstance(self.collision_radius, (int, float)) or not math.isfinite(self.collision_radius) or self.collision_radius < 0.0:
            raise ValueError(f"collision_radius must be non-negative, got {self.collision_radius!r}")
        object.__setattr__(self, "closest_distance", float(self.closest_distance))
        object.__setattr__(self, "time_to_cpa", float(self.time_to_cpa))
        object.__setattr__(self, "collision_radius", float(self.collision_radius))


def closest_point_of_approach(
    relative_position: Vector2,
    relative_velocity: Vector2,
    epsilon: float = 1e-9,
) -> CPAResult:
    """Calculate the Closest Point of Approach (CPA) between user and object.

    Solves d/dt ||p_rel + v_rel * t||^2 = 0 => t_cpa = -(p_rel · v_rel) / ||v_rel||^2.

    Args:
        relative_position: Relative position Vector2 (p_object - p_ego) in meters.
        relative_velocity: Relative velocity Vector2 (v_object - v_ego) in m/s.
        epsilon: Threshold for zero relative velocity.

    Returns:
        CPAResult with time_to_cpa, distance_at_cpa, relative_position_at_cpa, and is_approaching.
    """
    if not isinstance(relative_position, Vector2):
        raise TypeError(f"relative_position must be Vector2, got {type(relative_position)}")
    if not isinstance(relative_velocity, Vector2):
        raise TypeError(f"relative_velocity must be Vector2, got {type(relative_velocity)}")

    px, pz = relative_position.x, relative_position.z
    vx, vz = relative_velocity.x, relative_velocity.z

    v_sq = vx * vx + vz * vz
    dot_pv = px * vx + pz * vz
    is_approaching = dot_pv < 0.0

    # If relative velocity is effectively zero, distance does not change
    if v_sq <= epsilon:
        return CPAResult(
            time_to_cpa=0.0,
            distance_at_cpa=relative_position.magnitude,
            relative_position_at_cpa=relative_position,
            is_approaching=False,
        )

    t_raw = -dot_pv / v_sq

    # Only future CPA matters; if t_raw < 0, CPA was in the past and distance is currently increasing
    if t_raw <= 0.0:
        return CPAResult(
            time_to_cpa=0.0,
            distance_at_cpa=relative_position.magnitude,
            relative_position_at_cpa=relative_position,
            is_approaching=is_approaching,
        )

    p_cpa = Vector2(
        x=px + vx * t_raw,
        z=pz + vz * t_raw,
    )

    return CPAResult(
        time_to_cpa=t_raw,
        distance_at_cpa=p_cpa.magnitude,
        relative_position_at_cpa=p_cpa,
        is_approaching=is_approaching,
    )


def time_to_collision(
    relative_position: Vector2,
    relative_velocity: Vector2,
    collision_radius: float,
    epsilon: float = 1e-9,
) -> float | None:
    """Analytically calculate Time-to-Collision (TTC) with a circular safety boundary.

    Solves the quadratic equation:
        ||p_rel + v_rel * t||^2 = R^2
        a * t^2 + b * t + c = 0
    where:
        a = ||v_rel||^2
        b = 2 * (p_rel · v_rel)
        c = ||p_rel||^2 - R^2

    Args:
        relative_position: Relative position Vector2 in meters.
        relative_velocity: Relative velocity Vector2 in m/s.
        collision_radius: Effective collision boundary radius R in meters.
        epsilon: Numerical threshold for zero relative velocity.

    Returns:
        Smallest non-negative time t (seconds) when boundary intersection occurs,
        0.0 if already inside the safety radius, or None if no collision path exists.
    """
    if not isinstance(relative_position, Vector2):
        raise TypeError(f"relative_position must be Vector2, got {type(relative_position)}")
    if not isinstance(relative_velocity, Vector2):
        raise TypeError(f"relative_velocity must be Vector2, got {type(relative_velocity)}")
    if not isinstance(collision_radius, (int, float)) or not math.isfinite(collision_radius) or collision_radius < 0.0:
        raise ValueError(f"collision_radius must be a non-negative finite float, got {collision_radius!r}")

    px, pz = relative_position.x, relative_position.z
    vx, vz = relative_velocity.x, relative_velocity.z

    r_sq = collision_radius * collision_radius
    p_sq = px * px + pz * pz
    c = p_sq - r_sq

    # Case 1: Already inside or on collision boundary
    if c <= 0.0:
        return 0.0

    a = vx * vx + vz * vz
    # Case 2: Zero relative velocity while outside collision boundary
    if a <= epsilon:
        return None

    b = 2.0 * (px * vx + pz * vz)
    discriminant = b * b - 4.0 * a * c

    # Case 3: No real geometric intersection (trajectory misses circle)
    if discriminant < 0.0:
        return None

    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2.0 * a)
    t2 = (-b + sqrt_d) / (2.0 * a)

    # Find earliest non-negative root
    non_negative_roots = [t for t in (t1, t2) if t >= 0.0]
    if not non_negative_roots:
        return None

    return min(non_negative_roots)


def analyze_collision(
    relative_position: Vector2,
    relative_velocity: Vector2,
    collision_model: CollisionModel | None = None,
    horizon: float = 5.0,
) -> CollisionPrediction:
    """Perform full collision analysis computing CPA, TTC, and horizon intersection.

    Args:
        relative_position: Relative position Vector2 in meters.
        relative_velocity: Relative velocity Vector2 in m/s.
        collision_model: Circular collision parameters (defaults to 1.0m effective radius).
        horizon: Evaluation time horizon in seconds.

    Returns:
        CollisionPrediction struct with collision_predicted, ttc, closest_distance, time_to_cpa.
    """
    model = collision_model if collision_model is not None else CollisionModel()
    r_eff = model.effective_radius

    cpa = closest_point_of_approach(relative_position, relative_velocity)
    ttc = time_to_collision(relative_position, relative_velocity, r_eff)

    collision_predicted = (ttc is not None) and (0.0 <= ttc <= horizon)

    return CollisionPrediction(
        collision_predicted=collision_predicted,
        ttc=ttc,
        closest_distance=cpa.distance_at_cpa,
        time_to_cpa=cpa.time_to_cpa,
        collision_radius=r_eff,
    )
