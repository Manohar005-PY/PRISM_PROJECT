"""Ground-truth entities for synthetic simulation in PRISM.

Represents true physical state on the 2D ground plane (X: lateral, Z: forward/depth).
These entities represent exact simulated reality and are strictly separated from
noisy observations and estimated states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.types import Vector2


@dataclass(frozen=True)
class SimulatedEntity:
    """Ground-truth representation of an obstacle or dynamic object in the simulated world.

    Attributes:
        object_id: Unique identifier for the entity across simulation frames.
        class_id: Semantic category identifier (e.g. 'person', 'chair', 0).
        position: Exact 2D physical position on the ground plane (meters).
        velocity: Exact 2D physical velocity on the ground plane (m/s).
        timestamp: Monotonic simulation time in seconds.
    """

    object_id: int
    class_id: int | str
    position: Vector2
    velocity: Vector2
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, int):
            raise TypeError(f"SimulatedEntity.object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.class_id, (int, str)):
            raise TypeError(f"SimulatedEntity.class_id must be int or str, got {type(self.class_id)}")
        if not isinstance(self.position, Vector2):
            raise TypeError(f"SimulatedEntity.position must be Vector2, got {type(self.position)}")
        if not isinstance(self.velocity, Vector2):
            raise TypeError(f"SimulatedEntity.velocity must be Vector2, got {type(self.velocity)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"SimulatedEntity.timestamp must be a finite float, got {self.timestamp!r}")
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class EgoEntity:
    """Ground-truth representation of the user / ego agent in the simulated world.

    Attributes:
        position: Exact 2D physical position on the ground plane (meters).
        velocity: Exact 2D physical velocity on the ground plane (m/s).
        timestamp: Monotonic simulation time in seconds.
    """

    position: Vector2
    velocity: Vector2
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.position, Vector2):
            raise TypeError(f"EgoEntity.position must be Vector2, got {type(self.position)}")
        if not isinstance(self.velocity, Vector2):
            raise TypeError(f"EgoEntity.velocity must be Vector2, got {type(self.velocity)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"EgoEntity.timestamp must be a finite float, got {self.timestamp!r}")
        object.__setattr__(self, "timestamp", float(self.timestamp))
