"""S01 — Stationary Obstacle Scenario Family.

Tests basic collision geometry, closing distance, and increasing predictive risk toward
an immobile obstacle (e.g. chair, bin, column) as the user approaches.
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s01_stationary_scenario(
    initial_distance: float = 6.0,
    lateral_offset: float = 0.0,
    user_speed: float = 1.0,
    obstacle_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 6.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a parameterized S01 Stationary Obstacle scenario."""
    ego = AgentSpecification(
        agent_id="ego_user",
        class_id="user",
        initial_position=Vector2(x=0.0, z=0.0),
        initial_velocity=Vector2(x=0.0, z=user_speed),
        physical_radius=user_radius,
    )

    obstacle = AgentSpecification(
        agent_id=1,
        class_id="furniture",
        initial_position=Vector2(x=lateral_offset, z=initial_distance),
        initial_velocity=Vector2(x=0.0, z=0.0),
        physical_radius=obstacle_radius,
        behavior="stationary",
    )

    params = {
        "initial_distance": initial_distance,
        "lateral_offset": lateral_offset,
        "user_speed": user_speed,
        "obstacle_radius": obstacle_radius,
    }

    return ScenarioDefinition(
        scenario_id=f"S01_stationary_run_{run_idx:04d}",
        scenario_family="S01_stationary",
        name=f"Stationary Obstacle (dist={initial_distance}m, offset={lateral_offset}m)",
        description="User moves forward at constant speed toward an immobile obstacle located ahead.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(obstacle,),
        environment=EnvironmentSpecification(description="open_corridor"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
