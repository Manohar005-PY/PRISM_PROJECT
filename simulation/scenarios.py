"""Deterministic predefined simulation scenarios for PRISM.

Defines reproducible benchmark scenarios representing canonical indoor interaction patterns:
1. HEAD_ON: User and obstacle approach each other along the same path.
2. CROSSING: User and obstacle travel on perpendicular paths intersecting at t=5.0s, (0, 0).
3. PARALLEL: User and obstacle travel in the same direction at equal speed with constant lateral separation.
4. DIVERGING: User and obstacle travel on diverging trajectories with expanding distance.
5. STATIONARY_OBSTACLE: User approaches an immobile obstacle located directly ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.types import Vector2
from simulation.entities import EgoEntity, SimulatedEntity


@dataclass(frozen=True)
class Scenario:
    """Specification for a deterministic simulation run.

    Attributes:
        name: Semantic name identifying the scenario.
        ego_initial: Initial state of the user.
        objects_initial: Initial states of all obstacles/objects.
        duration: Total simulation run time in seconds.
        timestep: Time delta between consecutive frames in seconds.
    """

    name: str
    ego_initial: EgoEntity
    objects_initial: tuple[SimulatedEntity, ...]
    duration: float
    timestep: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Scenario.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.ego_initial, EgoEntity):
            raise TypeError(f"Scenario.ego_initial must be an EgoEntity, got {type(self.ego_initial)}")
        if not isinstance(self.objects_initial, (tuple, list)):
            raise TypeError(f"Scenario.objects_initial must be a sequence, got {type(self.objects_initial)}")
        if not isinstance(self.duration, (int, float)) or not math.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError(f"Scenario.duration must be a positive finite float, got {self.duration!r}")
        if not isinstance(self.timestep, (int, float)) or not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ValueError(f"Scenario.timestep must be a positive finite float, got {self.timestep!r}")
        if self.timestep > self.duration:
            raise ValueError(f"Scenario.timestep ({self.timestep}) cannot exceed duration ({self.duration})")

        tuple_objects = tuple(self.objects_initial)
        for i, obj in enumerate(tuple_objects):
            if not isinstance(obj, SimulatedEntity):
                raise TypeError(f"Initial object at index {i} must be SimulatedEntity, got {type(obj)}")

        object.__setattr__(self, "objects_initial", tuple_objects)
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "timestep", float(self.timestep))


def create_head_on_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Create a head-on approach scenario.

    User moves forward along +Z at 1 m/s from (0, 0).
    Object #1 moves backward along -Z at 1 m/s from (0, 5).
    They approach along the same centerline and converge toward a collision at t=2.5s at (0, 2.5).
    """
    ego = EgoEntity(
        position=Vector2(x=0.0, z=0.0),
        velocity=Vector2(x=0.0, z=1.0),
        timestamp=0.0,
    )
    obj = SimulatedEntity(
        object_id=1,
        class_id="person",
        position=Vector2(x=0.0, z=5.0),
        velocity=Vector2(x=0.0, z=-1.0),
        timestamp=0.0,
    )
    return Scenario(
        name="HEAD_ON",
        ego_initial=ego,
        objects_initial=(obj,),
        duration=duration,
        timestep=timestep,
    )


def create_crossing_scenario(duration: float = 6.0, timestep: float = 0.1) -> Scenario:
    """Create a perpendicular crossing scenario with a known intersection point.

    User moves east along +X from (-5, 0) at 1 m/s.
    Object #1 moves south along -Z from (0, 5) at 1 m/s.
    Theoretical intersection point: (0, 0) at t=5.0s.
    """
    ego = EgoEntity(
        position=Vector2(x=-5.0, z=0.0),
        velocity=Vector2(x=1.0, z=0.0),
        timestamp=0.0,
    )
    obj = SimulatedEntity(
        object_id=1,
        class_id="person",
        position=Vector2(x=0.0, z=5.0),
        velocity=Vector2(x=0.0, z=-1.0),
        timestamp=0.0,
    )
    return Scenario(
        name="CROSSING",
        ego_initial=ego,
        objects_initial=(obj,),
        duration=duration,
        timestep=timestep,
    )


def create_parallel_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Create a parallel motion scenario.

    User moves forward along +Z from (0, 0) at 1 m/s.
    Object #1 moves forward along +Z from (2, 5) at 1 m/s.
    Both travel at the same velocity, maintaining a constant lateral separation of 2 meters.
    No collision occurs.
    """
    ego = EgoEntity(
        position=Vector2(x=0.0, z=0.0),
        velocity=Vector2(x=0.0, z=1.0),
        timestamp=0.0,
    )
    obj = SimulatedEntity(
        object_id=1,
        class_id="person",
        position=Vector2(x=2.0, z=5.0),
        velocity=Vector2(x=0.0, z=1.0),
        timestamp=0.0,
    )
    return Scenario(
        name="PARALLEL",
        ego_initial=ego,
        objects_initial=(obj,),
        duration=duration,
        timestep=timestep,
    )


def create_diverging_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Create a diverging trajectories scenario.

    User moves forward along +Z from (0, 0) at 1 m/s.
    Object #1 moves laterally right along +X from (2, 5) at 1 m/s.
    Their relative separation increases continuously over time.
    """
    ego = EgoEntity(
        position=Vector2(x=0.0, z=0.0),
        velocity=Vector2(x=0.0, z=1.0),
        timestamp=0.0,
    )
    obj = SimulatedEntity(
        object_id=1,
        class_id="person",
        position=Vector2(x=2.0, z=5.0),
        velocity=Vector2(x=1.0, z=0.0),
        timestamp=0.0,
    )
    return Scenario(
        name="DIVERGING",
        ego_initial=ego,
        objects_initial=(obj,),
        duration=duration,
        timestep=timestep,
    )


def create_stationary_obstacle_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Create a stationary obstacle scenario.

    User moves forward along +Z from (0, 0) at 1 m/s.
    Object #1 remains immobile at (0, 5) with zero velocity.
    Used to benchmark reactive distance thresholding against predictive trajectory estimation.
    """
    ego = EgoEntity(
        position=Vector2(x=0.0, z=0.0),
        velocity=Vector2(x=0.0, z=1.0),
        timestamp=0.0,
    )
    obj = SimulatedEntity(
        object_id=1,
        class_id="chair",
        position=Vector2(x=0.0, z=5.0),
        velocity=Vector2(x=0.0, z=0.0),
        timestamp=0.0,
    )
    return Scenario(
        name="STATIONARY_OBSTACLE",
        ego_initial=ego,
        objects_initial=(obj,),
        duration=duration,
        timestep=timestep,
    )
