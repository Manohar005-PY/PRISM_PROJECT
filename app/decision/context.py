"""Decision context container bundling user kinematics, constraints, and geometric obstacle state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from app.core.types import ObjectState, RiskAssessment, Vector2


@dataclass(frozen=True)
class DecisionContext:
    """Bounded, immutable navigation context consumed by the DecisionEngine.

    Contains the exact spatial and risk inputs required to determine local avoidance actions.

    Attributes:
        ego_position: User ground-plane position in meters (Vector2(X, Z)).
        current_heading: Current travel angle in radians (0 = +Z forward, positive = right, negative = left).
        preferred_heading: Goal/nominal path angle in radians.
        max_turn_angle: Maximum allowable evasive turn angle in radians (e.g. pi/2).
        minimum_clearance: Minimum acceptable spatial clearance from obstacles in meters.
        current_speed: Current walking speed in m/s.
        maximum_speed: Maximum nominal walking speed in m/s.
        risk_assessment: Evaluated RiskAssessment from Part 6 (or None).
        predicted_obstacles: Tuple of active ObjectState instances from state estimation / prediction.
        timestamp: Evaluation timestamp in seconds.
    """

    ego_position: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    current_heading: float = 0.0
    preferred_heading: float = 0.0
    max_turn_angle: float = math.pi / 2.0
    minimum_clearance: float = 1.0
    current_speed: float = 1.0
    maximum_speed: float = 1.5
    risk_assessment: RiskAssessment | None = None
    predicted_obstacles: tuple[ObjectState, ...] = ()
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.ego_position, Vector2):
            raise TypeError(f"ego_position must be Vector2, got {type(self.ego_position)}")
        for name, val in [
            ("current_heading", self.current_heading),
            ("preferred_heading", self.preferred_heading),
            ("max_turn_angle", self.max_turn_angle),
            ("minimum_clearance", self.minimum_clearance),
            ("current_speed", self.current_speed),
            ("maximum_speed", self.maximum_speed),
            ("timestamp", self.timestamp),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")

        if self.max_turn_angle <= 0.0:
            raise ValueError(f"max_turn_angle must be positive, got {self.max_turn_angle}")
        if self.minimum_clearance <= 0.0:
            raise ValueError(f"minimum_clearance must be positive, got {self.minimum_clearance}")
        if self.current_speed < 0.0:
            raise ValueError(f"current_speed must be non-negative, got {self.current_speed}")
        if self.maximum_speed < self.current_speed:
            raise ValueError(f"maximum_speed ({self.maximum_speed}) cannot be less than current_speed ({self.current_speed})")

        if self.risk_assessment is not None and not isinstance(self.risk_assessment, RiskAssessment):
            raise TypeError(f"risk_assessment must be RiskAssessment or None, got {type(self.risk_assessment)}")

        if not isinstance(self.predicted_obstacles, (tuple, list)):
            raise TypeError(f"predicted_obstacles must be a Sequence of ObjectState, got {type(self.predicted_obstacles)}")
        tuple_obs = tuple(self.predicted_obstacles)
        for i, obs in enumerate(tuple_obs):
            if not isinstance(obs, ObjectState):
                raise TypeError(f"predicted_obstacles[{i}] must be an ObjectState, got {type(obs)}")

        object.__setattr__(self, "current_heading", float(self.current_heading))
        object.__setattr__(self, "preferred_heading", float(self.preferred_heading))
        object.__setattr__(self, "max_turn_angle", float(self.max_turn_angle))
        object.__setattr__(self, "minimum_clearance", float(self.minimum_clearance))
        object.__setattr__(self, "current_speed", float(self.current_speed))
        object.__setattr__(self, "maximum_speed", float(self.maximum_speed))
        object.__setattr__(self, "predicted_obstacles", tuple_obs)
        object.__setattr__(self, "timestamp", float(self.timestamp))
