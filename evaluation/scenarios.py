"""Evaluation benchmark scenario suites and controlled perturbation variants."""

from __future__ import annotations

from app.core.types import Vector2
from simulation.entities import EgoEntity, SimulatedEntity
from simulation.scenarios import (
    Scenario,
    create_crossing_scenario,
    create_diverging_scenario,
    create_head_on_scenario,
    create_parallel_scenario,
    create_stationary_obstacle_scenario,
)


def create_head_on_fast_scenario(duration: float = 3.0, timestep: float = 0.1) -> Scenario:
    """Fast head-on approach (relative velocity 3 m/s)."""
    ego = EgoEntity(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 1.5), timestamp=0.0)
    obj = SimulatedEntity(1, "person", Vector2(0.0, 6.0), Vector2(0.0, -1.5), timestamp=0.0)
    return Scenario("HEAD_ON_FAST", ego, (obj,), duration, timestep)


def create_head_on_slow_scenario(duration: float = 6.0, timestep: float = 0.1) -> Scenario:
    """Slow head-on approach (relative velocity 1 m/s)."""
    ego = EgoEntity(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.5), timestamp=0.0)
    obj = SimulatedEntity(1, "person", Vector2(0.0, 5.0), Vector2(0.0, -0.5), timestamp=0.0)
    return Scenario("HEAD_ON_SLOW", ego, (obj,), duration, timestep)


def create_near_miss_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Near miss: Obstacle passes close laterally (d_CPA = 0.8m) without colliding."""
    ego = EgoEntity(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 1.0), timestamp=0.0)
    obj = SimulatedEntity(1, "person", Vector2(0.8, 5.0), Vector2(0.0, -1.0), timestamp=0.0)
    return Scenario("NEAR_MISS", ego, (obj,), duration, timestep)


def create_departing_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Departing obstacle: Starts close at 2.0m and moves away in front of user."""
    ego = EgoEntity(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.8), timestamp=0.0)
    obj = SimulatedEntity(1, "person", Vector2(0.0, 2.0), Vector2(0.0, 1.5), timestamp=0.0)
    return Scenario("DEPARTING_OBJECT", ego, (obj,), duration, timestep)


def create_multi_object_scenario(duration: float = 5.0, timestep: float = 0.1) -> Scenario:
    """Multi-obstacle scenario with static chair, approaching threat, and stationary table."""
    ego = EgoEntity(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 1.0), timestamp=0.0)
    obj1 = SimulatedEntity(1, "chair", Vector2(2.5, 4.0), Vector2(0.0, 0.0), timestamp=0.0)
    obj2 = SimulatedEntity(2, "person", Vector2(0.0, 6.0), Vector2(0.0, -1.0), timestamp=0.0)
    obj3 = SimulatedEntity(3, "table", Vector2(-2.0, 5.0), Vector2(0.0, 0.0), timestamp=0.0)
    return Scenario("MULTI_OBJECT", ego, (obj1, obj2, obj3), duration, timestep)


def get_canonical_benchmark_suite() -> dict[str, Scenario]:
    """Return dictionary of standard evaluation scenarios."""
    return {
        "HEAD_ON": create_head_on_scenario(),
        "HEAD_ON_FAST": create_head_on_fast_scenario(),
        "HEAD_ON_SLOW": create_head_on_slow_scenario(),
        "CROSSING": create_crossing_scenario(),
        "PARALLEL": create_parallel_scenario(),
        "DIVERGING": create_diverging_scenario(),
        "STATIONARY_OBSTACLE": create_stationary_obstacle_scenario(),
        "NEAR_MISS": create_near_miss_scenario(),
        "DEPARTING_OBJECT": create_departing_scenario(),
        "MULTI_OBJECT": create_multi_object_scenario(),
    }
