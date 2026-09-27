"""S04 — Moving Away Scenario Family.

Tests false-positive suppression and departing target identification.
Target moves in the same direction ahead of the user at equal or higher speed,
resulting in non-positive closing velocities, infinite/null TTC, and low risk scores.
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s04_moving_away_scenario(
    initial_distance: float = 3.0,
    user_speed: float = 1.0,
    target_speed: float = 1.4,
    lateral_offset: float = 0.0,
    target_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 5.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a parameterized S04 Moving Away scenario."""
    ego = AgentSpecification(
        agent_id="ego_user",
        class_id="user",
        initial_position=Vector2(x=0.0, z=0.0),
        initial_velocity=Vector2(x=0.0, z=user_speed),
        physical_radius=user_radius,
    )

    target = AgentSpecification(
        agent_id=1,
        class_id="person",
        initial_position=Vector2(x=lateral_offset, z=initial_distance),
        initial_velocity=Vector2(x=0.0, z=target_speed),
        physical_radius=target_radius,
        behavior="constant_velocity",
    )

    params = {
        "initial_distance": initial_distance,
        "user_speed": user_speed,
        "target_speed": target_speed,
        "lateral_offset": lateral_offset,
        "relative_speed": target_speed - user_speed,
    }

    return ScenarioDefinition(
        scenario_id=f"S04_moving_away_run_{run_idx:04d}",
        scenario_family="S04_moving_away",
        name=f"Moving Away (init_dist={initial_distance}m, v_target={target_speed}m/s)",
        description="Target moves ahead of user in the same direction faster than user speed (departing motion).",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(target,),
        environment=EnvironmentSpecification(description="open_corridor"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
