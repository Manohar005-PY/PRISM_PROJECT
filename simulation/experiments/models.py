"""Domain models and data contracts for synthetic experiments in PRISM Part 16."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import platform
import sys
from typing import Any, Mapping, Sequence

from app.core.types import Vector2
from app.perception.model import PhysicalObservation


@dataclass(frozen=True)
class AgentSpecification:
    """Ground-truth specification of an agent (user or dynamic/static obstacle)."""

    agent_id: int | str
    class_id: str
    initial_position: Vector2
    initial_velocity: Vector2
    initial_acceleration: Vector2 = field(default_factory=lambda: Vector2(0.0, 0.0))
    physical_radius: float = 0.35
    behavior: str = "constant_velocity"
    behavior_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, (int, str)):
            raise TypeError(f"agent_id must be int or str, got {type(self.agent_id)}")
        if not isinstance(self.class_id, str):
            raise TypeError(f"class_id must be str, got {type(self.class_id)}")
        if not isinstance(self.initial_position, Vector2):
            raise TypeError(f"initial_position must be Vector2, got {type(self.initial_position)}")
        if not isinstance(self.initial_velocity, Vector2):
            raise TypeError(f"initial_velocity must be Vector2, got {type(self.initial_velocity)}")
        if not isinstance(self.initial_acceleration, Vector2):
            raise TypeError(f"initial_acceleration must be Vector2, got {type(self.initial_acceleration)}")
        if not isinstance(self.physical_radius, (int, float)) or self.physical_radius <= 0.0:
            raise ValueError(f"physical_radius must be a positive float, got {self.physical_radius}")
        object.__setattr__(self, "physical_radius", float(self.physical_radius))


@dataclass(frozen=True)
class EnvironmentSpecification:
    """Environmental bounds and constraints for a scenario."""

    corridor_width: float | None = None
    corridor_length: float | None = None
    description: str = "open_space"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoiseConfiguration:
    """Stochastic observation noise and dropout settings."""

    enabled: bool = False
    sigma_position_x: float = 0.0
    sigma_position_z: float = 0.0
    sigma_velocity_x: float = 0.0
    sigma_velocity_z: float = 0.0
    dropout_probability: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled)}")
        for attr in ("sigma_position_x", "sigma_position_z", "sigma_velocity_x", "sigma_velocity_z", "dropout_probability"):
            val = getattr(self, attr)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{attr} must be a non-negative finite float, got {val!r}")
            object.__setattr__(self, attr, float(val))
        if self.dropout_probability > 1.0:
            raise ValueError(f"dropout_probability must be in [0.0, 1.0], got {self.dropout_probability}")


@dataclass(frozen=True)
class ScenarioDefinition:
    """Complete parameter specification for a deterministic experimental run."""

    scenario_id: str
    scenario_family: str
    name: str
    description: str
    duration: float
    dt: float
    random_seed: int
    ego_agent: AgentSpecification
    agents: tuple[AgentSpecification, ...]
    environment: EnvironmentSpecification = field(default_factory=EnvironmentSpecification)
    noise_config: NoiseConfiguration = field(default_factory=NoiseConfiguration)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError(f"scenario_id must be non-empty string, got {self.scenario_id!r}")
        if not isinstance(self.scenario_family, str):
            raise TypeError(f"scenario_family must be str, got {type(self.scenario_family)}")
        if not isinstance(self.duration, (int, float)) or self.duration <= 0.0:
            raise ValueError(f"duration must be a positive float, got {self.duration}")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0.0:
            raise ValueError(f"dt must be a positive float, got {self.dt}")
        if not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be int, got {type(self.random_seed)}")
        if not isinstance(self.ego_agent, AgentSpecification):
            raise TypeError(f"ego_agent must be AgentSpecification, got {type(self.ego_agent)}")
        if not isinstance(self.agents, (tuple, list)):
            raise TypeError(f"agents must be a sequence, got {type(self.agents)}")
        tuple_agents = tuple(self.agents)
        for i, a in enumerate(tuple_agents):
            if not isinstance(a, AgentSpecification):
                raise TypeError(f"Agent at index {i} must be AgentSpecification, got {type(a)}")
        object.__setattr__(self, "agents", tuple_agents)
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "dt", float(self.dt))


@dataclass(frozen=True)
class GroundTruthCollisionState:
    """Exact ground-truth physical collision reference calculated independently from PRISM."""

    has_collided: bool
    time_to_collision: float | None
    closest_point_of_approach: float
    t_cpa: float
    first_collision_timestamp: float | None = None
    colliding_agent_id: int | str | None = None


@dataclass(frozen=True)
class GroundTruthObjectState:
    """Exact physical state of a single simulated entity at a given timestamp."""

    object_id: int | str
    class_id: str
    x: float
    z: float
    vx: float
    vz: float
    ax: float
    az: float
    radius: float
    heading_rad: float


@dataclass(frozen=True)
class GroundTruthState:
    """Exact physical ground-truth snapshot of the entire simulated world at time t."""

    timestamp: float
    frame_index: int
    ego: GroundTruthObjectState
    objects: tuple[GroundTruthObjectState, ...]
    collision_state: GroundTruthCollisionState

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "ego": asdict(self.ego),
            "objects": [asdict(obj) for obj in self.objects],
            "collision_state": asdict(self.collision_state),
        }


@dataclass(frozen=True)
class ObservationItem:
    """A single synthetic perception observation passed to PRISM."""

    object_id: int | str
    class_id: str
    x: float
    z: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    valid: bool = True


@dataclass(frozen=True)
class ObservationRecord:
    """Snapshot of observations emitted to PRISM at timestamp t."""

    timestamp: float
    frame_index: int
    observations: tuple[ObservationItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "observations": [
                {
                    "object_id": obs.object_id,
                    "class_id": obs.class_id,
                    "x": obs.x,
                    "z": obs.z,
                    "covariance": obs.covariance,
                    "confidence": obs.confidence,
                    "valid": obs.valid,
                }
                for obs in self.observations
            ],
        }


@dataclass(frozen=True)
class TrackOutputRecord:
    """PRISM tracker output for a single track at timestamp t."""

    track_id: int
    class_name: str
    lifecycle_state: str
    x: float
    z: float
    vx: float
    vz: float
    hits: int
    missed_frames: int


@dataclass(frozen=True)
class PRISMOutputRecord:
    """PRISM pipeline outputs recorded at timestamp t."""

    timestamp: float
    frame_index: int
    runtime_status: str
    decision_action: str
    decision_reason: str
    recommended_heading_rad: float | None
    recommended_heading_deg: float | None
    primary_threat_id: int | None
    risk_score: float
    risk_level: str
    predicted_ttc: float | None
    predicted_cpa: float | None
    active_track_count: int
    active_tracks: tuple[TrackOutputRecord, ...]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "runtime_status": self.runtime_status,
            "decision_action": self.decision_action,
            "decision_reason": self.decision_reason,
            "recommended_heading_rad": self.recommended_heading_rad,
            "recommended_heading_deg": self.recommended_heading_deg,
            "primary_threat_id": self.primary_threat_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "predicted_ttc": self.predicted_ttc,
            "predicted_cpa": self.predicted_cpa,
            "active_track_count": self.active_track_count,
            "active_tracks": [asdict(trk) for trk in self.active_tracks],
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class RunMetadata:
    """Metadata recorded for every individual experimental run."""

    experiment_id: str
    scenario_id: str
    scenario_family: str
    run_id: str
    random_seed: int
    scenario_version: str = "1.0.0"
    experiment_config_version: str = "1.0.0"
    generator_version: str = "16.0.0"
    git_commit: str = "unknown"
    dt: float = 0.05
    duration: float = 5.0
    num_frames: int = 0
    noise_config: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    platform_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetManifest:
    """Dataset-level discovery and summary index."""

    dataset_name: str
    version: str
    scenario_families: list[str]
    total_scenarios: int
    total_runs: int
    total_frames: int
    created_at: str
    generator_version: str
    git_commit: str
    scenarios: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
