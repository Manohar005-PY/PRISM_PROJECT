"""Runtime configuration parameters for PRISM end-to-end execution."""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.decision.model import NavigationAction


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable configuration for PRISM real-time pipeline execution.

    Attributes:
        target_fps: Target pipeline execution rate in frames per second.
        max_frame_processing_time: Maximum allowable processing latency per frame in seconds.
        stale_observation_timeout: Timeout threshold in seconds before sensor observations are deemed stale.
        prediction_horizon: Forward trajectory extrapolation horizon in seconds.
        enable_metrics: Whether to collect rolling stage timing and health diagnostics.
        fail_safe_action: Default navigation action when unrecoverable subsystem errors occur.
        maximum_consecutive_failures: Number of consecutive frame failures before entering FAILED status.
        min_valid_dt: Minimum valid physical delta-time between consecutive frames (seconds).
        max_valid_dt: Maximum valid physical delta-time before resetting filter timing (seconds).
    """

    target_fps: float = 30.0
    max_frame_processing_time: float = 0.100
    stale_observation_timeout: float = 1.0
    prediction_horizon: float = 3.0
    enable_metrics: bool = True
    fail_safe_action: NavigationAction = NavigationAction.STOP
    maximum_consecutive_failures: int = 3
    min_valid_dt: float = 0.001
    max_valid_dt: float = 1.0

    def __post_init__(self) -> None:
        for name, val in [
            ("target_fps", self.target_fps),
            ("max_frame_processing_time", self.max_frame_processing_time),
            ("stale_observation_timeout", self.stale_observation_timeout),
            ("prediction_horizon", self.prediction_horizon),
            ("min_valid_dt", self.min_valid_dt),
            ("max_valid_dt", self.max_valid_dt),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val!r}")

        if not isinstance(self.maximum_consecutive_failures, int) or self.maximum_consecutive_failures <= 0:
            raise ValueError(f"maximum_consecutive_failures must be positive int, got {self.maximum_consecutive_failures!r}")
        if not isinstance(self.enable_metrics, bool):
            raise TypeError(f"enable_metrics must be bool, got {type(self.enable_metrics)}")
        if not isinstance(self.fail_safe_action, NavigationAction):
            raise TypeError(f"fail_safe_action must be NavigationAction enum, got {type(self.fail_safe_action)}")

        object.__setattr__(self, "target_fps", float(self.target_fps))
        object.__setattr__(self, "max_frame_processing_time", float(self.max_frame_processing_time))
        object.__setattr__(self, "stale_observation_timeout", float(self.stale_observation_timeout))
        object.__setattr__(self, "prediction_horizon", float(self.prediction_horizon))
        object.__setattr__(self, "min_valid_dt", float(self.min_valid_dt))
        object.__setattr__(self, "max_valid_dt", float(self.max_valid_dt))
