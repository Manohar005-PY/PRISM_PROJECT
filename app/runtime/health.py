"""Runtime health monitoring, degraded state tracking, and failure recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from app.runtime.config import RuntimeConfig


class RuntimeStatus(str, Enum):
    """Operational health status of the PRISM runtime."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeHealth:
    """Snapshot of PRISM runtime operational status and diagnostic indicators."""

    status: RuntimeStatus
    frames_processed: int = 0
    frames_failed: int = 0
    consecutive_failures: int = 0
    dropped_frames: int = 0
    active_tracks: int = 0
    stale_tracks: int = 0
    average_latency_ms: float = 0.0
    current_fps: float = 0.0
    last_error: str | None = None

    def is_healthy(self) -> bool:
        """Check if runtime is in fully functional healthy state."""
        return self.status == RuntimeStatus.HEALTHY


class HealthMonitor:
    """Monitors frame results, failure rates, and performance thresholds."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.frames_processed = 0
        self.frames_failed = 0
        self.consecutive_failures = 0
        self.dropped_frames = 0
        self.last_error: str | None = None

    def record_success(self) -> None:
        """Record successful frame execution."""
        self.frames_processed += 1
        self.consecutive_failures = 0

    def record_failure(self, error_message: str) -> None:
        """Record frame failure."""
        self.frames_processed += 1
        self.frames_failed += 1
        self.consecutive_failures += 1
        self.last_error = error_message

    def record_dropped_frame(self) -> None:
        """Record frame drop due to processing queue saturation."""
        self.dropped_frames += 1

    def assess_health(
        self,
        active_tracks: int,
        stale_tracks: int,
        average_latency_ms: float,
        current_fps: float,
    ) -> RuntimeHealth:
        """Evaluate current runtime health status."""
        if self.consecutive_failures >= self.config.maximum_consecutive_failures:
            status = RuntimeStatus.FAILED
        elif (
            self.consecutive_failures > 0
            or stale_tracks > 0
            or average_latency_ms > (self.config.max_frame_processing_time * 1000.0)
        ):
            status = RuntimeStatus.DEGRADED
        else:
            status = RuntimeStatus.HEALTHY

        return RuntimeHealth(
            status=status,
            frames_processed=self.frames_processed,
            frames_failed=self.frames_failed,
            consecutive_failures=self.consecutive_failures,
            dropped_frames=self.dropped_frames,
            active_tracks=active_tracks,
            stale_tracks=stale_tracks,
            average_latency_ms=average_latency_ms,
            current_fps=current_fps,
            last_error=self.last_error,
        )
