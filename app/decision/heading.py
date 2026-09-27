"""Deterministic local safe-heading analysis and clearance calculation.

Coordinate Convention:
    +Z = forward  (angle = 0 radians)
    +X = right    (positive angle)
    -X = left     (negative angle)
    Ray direction for angle theta: (sin(theta), cos(theta))
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from app.core.types import ObjectState, Vector2
from app.decision.context import DecisionContext


@dataclass(frozen=True)
class SafeHeading:
    """Evaluated candidate travel direction and clearance.

    Attributes:
        angle: Evasion heading in radians (0 = forward, positive = right, negative = left).
        clearance: Unobstructed forward distance in meters along the corridor.
        deviation_from_current: Angular difference |angle - current_heading| in radians.
        is_valid: True iff clearance >= context.minimum_clearance.
    """

    angle: float
    clearance: float
    deviation_from_current: float
    is_valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.angle, (int, float)) or not math.isfinite(self.angle):
            raise ValueError(f"angle must be a finite float, got {self.angle!r}")
        if not isinstance(self.clearance, (int, float)) or not math.isfinite(self.clearance) or self.clearance < 0.0:
            raise ValueError(f"clearance must be non-negative, got {self.clearance!r}")
        if not isinstance(self.deviation_from_current, (int, float)) or not math.isfinite(self.deviation_from_current) or self.deviation_from_current < 0.0:
            raise ValueError(f"deviation_from_current must be non-negative, got {self.deviation_from_current!r}")

        object.__setattr__(self, "angle", float(self.angle))
        object.__setattr__(self, "clearance", float(self.clearance))
        object.__setattr__(self, "deviation_from_current", float(self.deviation_from_current))


def compute_safe_headings(
    context: DecisionContext,
    num_samples: int = 37,
    max_lookahead: float = 6.0,
    user_radius: float = 0.35,
    obstacle_radius: float = 0.35,
) -> list[SafeHeading]:
    """Sample candidate headings across [-max_turn_angle, +max_turn_angle] and evaluate clearance.

    Args:
        context: DecisionContext containing ego kinematics and predicted obstacles.
        num_samples: Number of discrete angular rays to evaluate (odd integer for center ray).
        max_lookahead: Maximum corridor search distance in meters.
        user_radius: User personal body radius in meters.
        obstacle_radius: Nominal obstacle clearance radius in meters.

    Returns:
        List of SafeHeading instances ordered from left (-max_turn_angle) to right (+max_turn_angle).
    """
    if num_samples < 3:
        num_samples = 3

    half_angle = context.max_turn_angle
    angular_step = (2.0 * half_angle) / (num_samples - 1)
    corridor_half_width = user_radius + obstacle_radius

    ego_x, ego_z = context.ego_position.x, context.ego_position.z
    results: list[SafeHeading] = []

    for i in range(num_samples):
        theta = -half_angle + i * angular_step
        # Unit direction vector along ray theta
        dir_x = math.sin(theta)
        dir_z = math.cos(theta)

        ray_clearance = max_lookahead

        for obs in context.predicted_obstacles:
            # Relative vector from user to obstacle
            rel_x = obs.position.x - ego_x
            rel_z = obs.position.z - ego_z

            # Projection of obstacle along the ray (longitudinal distance)
            s_long = rel_x * dir_x + rel_z * dir_z

            # Perpendicular distance from obstacle center to the ray
            cross_x = rel_x - s_long * dir_x
            cross_z = rel_z - s_long * dir_z
            d_perp = math.sqrt(cross_x * cross_x + cross_z * cross_z)

            # Check if obstacle falls within the safety corridor
            if d_perp < corridor_half_width:
                if s_long > 0.0:
                    # Penetration entry distance along ray
                    penetration_offset = math.sqrt(max(0.0, corridor_half_width * corridor_half_width - d_perp * d_perp))
                    blocked_distance = max(0.0, s_long - penetration_offset)
                    if blocked_distance < ray_clearance:
                        ray_clearance = blocked_distance
                elif math.sqrt(rel_x * rel_x + rel_z * rel_z) < corridor_half_width:
                    # Obstacle is currently overlapping the user
                    ray_clearance = 0.0

        deviation = abs(theta - context.current_heading)
        is_valid = ray_clearance >= context.minimum_clearance

        results.append(
            SafeHeading(
                angle=theta,
                clearance=ray_clearance,
                deviation_from_current=deviation,
                is_valid=is_valid,
            )
        )

    return results


def select_best_heading(
    safe_headings: Sequence[SafeHeading],
    context: DecisionContext,
    w_heading: float = 0.60,
    w_clearance: float = 0.40,
    max_lookahead: float = 6.0,
) -> SafeHeading | None:
    """Select the optimal safe heading minimizing deviation from preferred direction while maximizing clearance.

    Returns:
        Best SafeHeading, or None if no heading satisfies minimum clearance constraints.
    """
    valid_candidates = [h for h in safe_headings if h.is_valid]
    if not valid_candidates:
        return None

    def heading_cost(h: SafeHeading) -> tuple[float, float, float]:
        # Component 1: Normalized deviation from preferred goal heading
        dev_preferred = abs(h.angle - context.preferred_heading) / max(1e-4, context.max_turn_angle)
        # Component 2: Clearance penalty (lower clearance = higher cost)
        norm_clearance = min(1.0, h.clearance / max_lookahead)
        cost_clearance = 1.0 - norm_clearance

        composite_cost = w_heading * dev_preferred + w_clearance * cost_clearance
        # Tie-breakers: lower cost, smaller deviation from current, prefer smaller absolute angle
        return (composite_cost, h.deviation_from_current, abs(h.angle))

    return min(valid_candidates, key=heading_cost)
