"""S05 — Direction Change Scenario Family.

Tests tracking adaptability, Kalman velocity re-estimation, and predictive risk updates
when an obstacle changes direction dynamically at time t_turn.
Future trajectory is strictly protected; PRISM receives only observations at current time t.
"""

from __future__ import annotations

import math

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)


def create_s05_direction_change_scenario(
    initial_position: tuple[float, float] = (1.5, 5.0),
    initial_velocity: tuple[float, float] = (0.0, -1.0),
    turn_time: float = 2.5,
    turn_angle_rad: float = math.pi / 2.0,  # 90-degree turn
    user_speed: float = 1.0,
    person_radius: float = 0.35,
    user_radius: float = 0.35,
    duration: float = 6.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a parameterized S05 Direction Change scenario."""
    ego = AgentSpecification(
        agent_id="ego_user",
        class_id="user",
        initial_position=Vector2(x=0.0, z=0.0),
        initial_velocity=Vector2(x=0.0, z=user_speed),
        physical_radius=user_radius,
    )

    speed = math.sqrt(initial_velocity[0] ** 2 + initial_velocity[1] ** 2)
    behavior_params = {
        "turn_time": turn_time,
        "turn_angle_rad": turn_angle_rad,
        "speed": speed,
    }

    target = AgentSpecification(
        agent_id=1,
        class_id="person",
        initial_position=Vector2(x=initial_position[0], z=initial_position[1]),
        initial_velocity=Vector2(x=initial_velocity[0], z=initial_velocity[1]),
        physical_radius=person_radius,
        behavior="direction_change",
        behavior_params=behavior_params,
    )

    params = {
        "initial_position": list(initial_position),
        "initial_velocity": list(initial_velocity),
        "turn_time": turn_time,
        "turn_angle_deg": round(math.degrees(turn_angle_rad), 2),
        "user_speed": user_speed,
    }

    return ScenarioDefinition(
        scenario_id=f"S05_direction_change_run_{run_idx:04d}",
        scenario_family="S05_direction_change",
        name=f"Direction Change (turn_t={turn_time}s, angle={math.degrees(turn_angle_rad):.0f}deg)",
        description="Target approaches along one trajectory, then turns dynamically at t_turn onto a new heading.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(target,),
        environment=EnvironmentSpecification(description="open_lobby"),
        noise_config=NoiseConfiguration(enabled=False, seed=seed),
        parameters=params,
    )
