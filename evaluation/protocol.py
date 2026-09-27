"""Experiment protocol: Immutable, serializable experiment configuration ensuring reproducibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for synthetic observation noise and drop perturbations."""

    sigma_x: float = 0.05
    sigma_z: float = 0.05
    detection_drop_rate: float = 0.0
    false_detection_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.sigma_x, (int, float)) or not math.isfinite(self.sigma_x) or self.sigma_x < 0.0:
            raise ValueError(f"sigma_x must be non-negative finite float, got {self.sigma_x!r}")
        if not isinstance(self.sigma_z, (int, float)) or not math.isfinite(self.sigma_z) or self.sigma_z < 0.0:
            raise ValueError(f"sigma_z must be non-negative finite float, got {self.sigma_z!r}")
        if not isinstance(self.detection_drop_rate, (int, float)) or not (0.0 <= self.detection_drop_rate <= 1.0):
            raise ValueError(f"detection_drop_rate must be in [0.0, 1.0], got {self.detection_drop_rate!r}")
        if not isinstance(self.false_detection_rate, (int, float)) or not (0.0 <= self.false_detection_rate <= 1.0):
            raise ValueError(f"false_detection_rate must be in [0.0, 1.0], got {self.false_detection_rate!r}")

        object.__setattr__(self, "sigma_x", float(self.sigma_x))
        object.__setattr__(self, "sigma_z", float(self.sigma_z))
        object.__setattr__(self, "detection_drop_rate", float(self.detection_drop_rate))
        object.__setattr__(self, "false_detection_rate", float(self.false_detection_rate))


@dataclass(frozen=True)
class LatencyConfig:
    """Configuration for simulated perception latency and jitter."""

    simulated_latency_ms: float = 0.0
    jitter_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.simulated_latency_ms, (int, float)) or not math.isfinite(self.simulated_latency_ms) or self.simulated_latency_ms < 0.0:
            raise ValueError(f"simulated_latency_ms must be non-negative finite float, got {self.simulated_latency_ms!r}")
        if not isinstance(self.jitter_ms, (int, float)) or not math.isfinite(self.jitter_ms) or self.jitter_ms < 0.0:
            raise ValueError(f"jitter_ms must be non-negative finite float, got {self.jitter_ms!r}")

        object.__setattr__(self, "simulated_latency_ms", float(self.simulated_latency_ms))
        object.__setattr__(self, "jitter_ms", float(self.jitter_ms))


@dataclass(frozen=True)
class ExperimentProtocol:
    """Complete, immutable experiment protocol for scientific reproducibility."""

    experiment_id: str
    random_seed: int
    scenario_ids: tuple[str, ...]
    number_of_trials: int = 5
    timestep: float = 0.1
    duration: float = 5.0
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    latency_config: LatencyConfig = field(default_factory=LatencyConfig)
    prediction_horizon: float = 3.0
    system_config: Mapping[str, Any] = field(default_factory=dict)
    baseline_configs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError(f"experiment_id must be a non-empty string, got {self.experiment_id!r}")
        if not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be int, got {type(self.random_seed)}")
        if not isinstance(self.scenario_ids, tuple) or not self.scenario_ids:
            raise ValueError(f"scenario_ids must be a non-empty tuple of strings, got {self.scenario_ids!r}")
        if not isinstance(self.number_of_trials, int) or self.number_of_trials < 1:
            raise ValueError(f"number_of_trials must be positive int, got {self.number_of_trials!r}")

    def to_dict(self) -> dict[str, Any]:
        """Convert protocol to serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentProtocol:
        """Construct protocol instance from dictionary."""
        d = dict(data)
        d["scenario_ids"] = tuple(d["scenario_ids"])
        d["noise_config"] = NoiseConfig(**d.get("noise_config", {}))
        d["latency_config"] = LatencyConfig(**d.get("latency_config", {}))
        return cls(**d)
