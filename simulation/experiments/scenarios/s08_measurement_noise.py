"""S08 — Measurement Noise & Perturbations Scenario Family.

Applies controlled Gaussian position noise, velocity noise, and observation dropout
across canonical head-on and crossing trajectories to benchmark Kalman filter state estimation
and tracking robustness under realistic sensor degradation.
"""

from __future__ import annotations

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)

NOISE_PROFILES: dict[str, dict[str, float]] = {
    "NO_NOISE": {"sigma_pos_x": 0.0, "sigma_pos_z": 0.0, "dropout": 0.0},
    "LOW_NOISE": {"sigma_pos_x": 0.05, "sigma_pos_z": 0.05, "dropout": 0.0},
    "MEDIUM_NOISE": {"sigma_pos_x": 0.15, "sigma_pos_z": 0.15, "dropout": 0.05},
    "HIGH_NOISE": {"sigma_pos_x": 0.30, "sigma_pos_z": 0.30, "dropout": 0.10},
    "DROPOUT_ONLY": {"sigma_pos_x": 0.0, "sigma_pos_z": 0.0, "dropout": 0.15},
}


def create_s08_noise_scenario(
    noise_level: str = "MEDIUM_NOISE",
    initial_separation: float = 6.0,
    user_speed: float = 1.0,
    person_speed: float = 1.0,
    user_radius: float = 0.35,
    person_radius: float = 0.35,
    duration: float = 5.0,
    dt: float = 0.05,
    seed: int = 42,
    run_idx: int = 1,
) -> ScenarioDefinition:
    """Create a scenario with controlled observation noise and dropout."""
    profile = NOISE_PROFILES.get(noise_level, NOISE_PROFILES["MEDIUM_NOISE"])

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
        initial_position=Vector2(x=0.0, z=initial_separation),
        initial_velocity=Vector2(x=0.0, z=-person_speed),
        physical_radius=person_radius,
        behavior="constant_velocity",
    )

    noise_cfg = NoiseConfiguration(
        enabled=(noise_level != "NO_NOISE"),
        sigma_position_x=profile["sigma_pos_x"],
        sigma_position_z=profile["sigma_pos_z"],
        dropout_probability=profile["dropout"],
        seed=seed,
    )

    params = {
        "noise_level": noise_level,
        "sigma_pos_x": profile["sigma_pos_x"],
        "sigma_pos_z": profile["sigma_pos_z"],
        "dropout_prob": profile["dropout"],
        "initial_separation": initial_separation,
    }

    return ScenarioDefinition(
        scenario_id=f"S08_noise_{noise_level.lower()}_run_{run_idx:04d}",
        scenario_family="S08_measurement_noise",
        name=f"Measurement Noise ({noise_level}, sigma={profile['sigma_pos_z']}m, drop={profile['dropout']})",
        description="Head-on trajectory evaluated under controlled stochastic observation noise and dropout.",
        duration=duration,
        dt=dt,
        random_seed=seed,
        ego_agent=ego,
        agents=(person,),
        environment=EnvironmentSpecification(description="sensor_degradation_corridor"),
        noise_config=noise_cfg,
        parameters=params,
    )
