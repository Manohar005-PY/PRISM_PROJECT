"""S06 — Multiple Objects Scenario Family.

Tests simultaneous multi-object tracking, identity preservation across tracks,
and threat prioritization (identifying the single most critical primary threat among multiple agents).
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s06_multiple_objects_scenario(
    num_objects: int = 3,
    user_speed: float = 1.0,
    user_radius: float = 0.35,
    duration: float = 6.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a multi-object scenario with 2, 3, or 5 simultaneous entities with mixed threat profiles."""
    ego = AgentSpecification(
        agent_id="ego_user",
        class_id="user",
        initial_position=Vector2(x=0.0, z=0.0),
        initial_velocity=Vector2(x=0.0, z=user_speed),
        physical_radius=user_radius,
    )

    agents: list[AgentSpecification] = []

    # Object 1: Primary Head-On Threat (CRITICAL)
    agents.append(
        AgentSpecification(
            agent_id=1,
            class_id="person",
            initial_position=Vector2(x=0.1, z=5.5),
            initial_velocity=Vector2(x=0.0, z=-1.0),
            physical_radius=0.35,
            behavior="constant_velocity",
        )
    )

    # Object 2: Lateral Crossing Agent (MEDIUM / CAUTION)
    agents.append(
        AgentSpecification(
            agent_id=2,
            class_id="person",
            initial_position=Vector2(x=-3.0, z=3.0),
            initial_velocity=Vector2(x=0.8, z=0.0),
            physical_radius=0.35,
            behavior="constant_velocity",
        )
    )

    if num_objects >= 3:
        # Object 3: Departing Agent (SAFE / LOW)
        agents.append(
            AgentSpecification(
                agent_id=3,
                class_id="person",
                initial_position=Vector2(x=1.8, z=2.0),
                initial_velocity=Vector2(x=0.0, z=1.2),
                physical_radius=0.35,
                behavior="constant_velocity",
            )
        )

    if num_objects >= 5:
        # Object 4: Stationary side furniture (SAFE)
        agents.append(
            AgentSpecification(
                agent_id=4,
                class_id="furniture",
                initial_position=Vector2(x=-1.5, z=4.0),
                initial_velocity=Vector2(x=0.0, z=0.0),
                physical_radius=0.30,
                behavior="stationary",
            )
        )
        # Object 5: Distant approaching pedestrian (LOW)
        agents.append(
            AgentSpecification(
                agent_id=5,
                class_id="person",
                initial_position=Vector2(x=1.0, z=9.0),
                initial_velocity=Vector2(x=-0.2, z=-0.8),
                physical_radius=0.35,
                behavior="constant_velocity",
            )
        )

    params = {
        "num_objects": len(agents),
        "user_speed": user_speed,
        "primary_threat_id": 1,
    }

    return ScenarioDefinition(
        scenario_id=f"S06_multi_obj_{len(agents)}agents_run_{run_idx:04d}",
        scenario_family="S06_multiple_objects",
        name=f"Multiple Objects ({len(agents)} simultaneous agents, primary=ID#1)",
        description="Multiple independent dynamic and static obstacles testing multi-target tracking and threat ranking.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=tuple(agents),
        environment=EnvironmentSpecification(description="crowded_hallway"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
