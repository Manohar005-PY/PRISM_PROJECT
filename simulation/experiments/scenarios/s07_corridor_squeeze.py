"""S07 — Corridor Squeeze Scenario Family.

Tests local avoidance policy and spatial clearance estimation under lateral boundary constraints
(e.g. hallway walls at +/- W_corridor) when encountering an obstacle positioned off-center.
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s07_corridor_squeeze_scenario(
    corridor_width: float = 2.4,
    obstacle_x: float = 0.4,  # Offset to the right, leaving left side open
    obstacle_z: float = 4.5,
    user_speed: float = 1.0,
    obstacle_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 6.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a parameterized S07 Corridor Squeeze scenario."""
    ego = AgentSpecification(
        agent_id="ego_user",
        class_id="user",
        initial_position=Vector2(x=0.0, z=0.0),
        initial_velocity=Vector2(x=0.0, z=user_speed),
        physical_radius=user_radius,
    )

    # Calculate available lateral clearance
    half_w = corridor_width / 2.0
    left_clearance = (obstacle_x - obstacle_radius) - (-half_w)
    right_clearance = half_w - (obstacle_x + obstacle_radius)

    obstacle = AgentSpecification(
        agent_id=1,
        class_id="person",
        initial_position=Vector2(x=obstacle_x, z=obstacle_z),
        initial_velocity=Vector2(x=0.0, z=-0.6),
        physical_radius=obstacle_radius,
        behavior="constant_velocity",
    )

    params = {
        "corridor_width": corridor_width,
        "obstacle_x": obstacle_x,
        "obstacle_z": obstacle_z,
        "left_clearance": round(left_clearance, 2),
        "right_clearance": round(right_clearance, 2),
        "expected_evasion_direction": "LEFT" if left_clearance > right_clearance else "RIGHT",
    }

    env = EnvironmentSpecification(
        corridor_width=corridor_width,
        corridor_length=15.0,
        description="bounded_narrow_corridor",
        metadata={"walls": [-half_w, half_w]},
    )

    return ScenarioDefinition(
        scenario_id=f"S07_corridor_squeeze_run_{run_idx:04d}",
        scenario_family="S07_corridor_squeeze",
        name=f"Corridor Squeeze (width={corridor_width}m, obs_x={obstacle_x}m)",
        description="Navigating a narrow bounded corridor where an obstacle creates asymmetric lateral clearances.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(obstacle,),
        environment=env,
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
