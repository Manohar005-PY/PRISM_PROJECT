"""S03 — Lateral Crossing Scenario Family.

Tests trajectory intersection, perpendicular/oblique motion prediction, and discriminating
between true collision paths (S03-COLLISION) versus safe near-miss crossings (S03-MISS).
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s03_crossing_collision_scenario(
    crossing_z: float = 3.5,
    user_speed: float = 1.0,
    pedestrian_speed: float = 1.0,
    start_x: float = -3.5,
    person_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 6.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a crossing scenario where user and pedestrian trajectories collide at (0, crossing_z)."""
    # Time to intersection for user: t_int = crossing_z / user_speed
    t_int = crossing_z / user_speed
    # Pedestrian starts at start_x (negative) and moves toward +X to arrive at X=0 at t_int
    vx_ped = abs(start_x) / t_int

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
        initial_position=Vector2(x=start_x, z=crossing_z),
        initial_velocity=Vector2(x=vx_ped, z=0.0),
        physical_radius=person_radius,
        behavior="constant_velocity",
    )

    params = {
        "crossing_type": "COLLISION",
        "crossing_z": crossing_z,
        "user_speed": user_speed,
        "pedestrian_speed": vx_ped,
        "start_x": start_x,
    }

    return ScenarioDefinition(
        scenario_id=f"S03_crossing_coll_run_{run_idx:04d}",
        scenario_family="S03_lateral_crossing",
        name=f"Lateral Crossing Collision (z={crossing_z}m, t_int={t_int:.2f}s)",
        description="Pedestrian crosses path perpendicularly with intersecting spatial and temporal coordinates (Collision).",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(person,),
        environment=EnvironmentSpecification(description="open_intersection"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )


def create_s03_crossing_miss_scenario(
    crossing_z: float = 3.5,
    user_speed: float = 1.0,
    time_offset: float = 1.8,
    start_x: float = -3.5,
    person_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 6.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a crossing scenario where pedestrian passes safely ahead of or behind user (Near Miss)."""
    t_int_user = crossing_z / user_speed
    # Pedestrian arrives at X=0 earlier or later by time_offset
    t_int_ped = max(0.5, t_int_user - time_offset)
    vx_ped = abs(start_x) / t_int_ped

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
        initial_position=Vector2(x=start_x, z=crossing_z),
        initial_velocity=Vector2(x=vx_ped, z=0.0),
        physical_radius=person_radius,
        behavior="constant_velocity",
    )

    params = {
        "crossing_type": "MISS",
        "crossing_z": crossing_z,
        "user_speed": user_speed,
        "pedestrian_speed": vx_ped,
        "time_offset": time_offset,
        "start_x": start_x,
    }

    return ScenarioDefinition(
        scenario_id=f"S03_crossing_miss_run_{run_idx:04d}",
        scenario_family="S03_lateral_crossing",
        name=f"Lateral Crossing Miss (z={crossing_z}m, offset={time_offset:.2f}s)",
        description="Pedestrian crosses path perpendicularly but passes safely ahead/behind user without collision.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(person,),
        environment=EnvironmentSpecification(description="open_intersection"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
