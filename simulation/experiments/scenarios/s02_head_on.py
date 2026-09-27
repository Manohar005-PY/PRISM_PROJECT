"""S02 — Head-On Approach Scenario Family.

Tests predictive collision geometry, high relative closing velocities, finite TTC calculation,
dCPA estimation, and timely avoidance trigger against an oncoming pedestrian.
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s02_head_on_scenario(
    initial_separation: float = 6.0,
    user_speed: float = 1.0,
    person_speed: float = 1.0,
    lateral_offset: float = 0.0,
    person_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 5.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a parameterized S02 Head-On Approach scenario."""
    ego = AgentSpecification(
        agent_id="ego_user",
        class_id="user",
        initial_position=Vector2(x=0.0, z=0.0),
        initial_velocity=Vector2(x=0.0, z=user_speed),
        physical_radius=user_radius,
    )

    person = AgentSpecification(
        agent_id=1,
        class_id="person",
        initial_position=Vector2(x=lateral_offset, z=initial_separation),
        initial_velocity=Vector2(x=0.0, z=-person_speed),
        physical_radius=person_radius,
        behavior="constant_velocity",
    )

    params = {
        "initial_separation": initial_separation,
        "user_speed": user_speed,
        "person_speed": person_speed,
        "lateral_offset": lateral_offset,
        "person_radius": person_radius,
    }

    return ScenarioDefinition(
        scenario_id=f"S02_head_on_run_{run_idx:04d}",
        scenario_family="S02_head_on",
        name=f"Head-On Approach (sep={initial_separation}m, v_rel={user_speed + person_speed}m/s)",
        description="User and oncoming pedestrian approach each other head-on along the depth axis.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(person,),
        environment=EnvironmentSpecification(description="open_corridor"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
