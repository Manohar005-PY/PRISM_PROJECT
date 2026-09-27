"""Performance instrumentation, per-stage timing, and rolling latency statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class StageTiming:
    """Detailed timing breakdown for a single frame execution across all stages."""

    perception_ms: float = 0.0
    tracking_ms: float = 0.0
    estimation_ms: float = 0.0
    prediction_ms: float = 0.0
    risk_ms: float = 0.0
    decision_ms: float = 0.0
    total_ms: float = 0.0
    effective_fps: float = 0.0

    def __post_init__(self) -> None:
        for name in [
            "perception_ms",
            "tracking_ms",
            "estimation_ms",
            "prediction_ms",
            "risk_ms",
            "decision_ms",
            "total_ms",
            "effective_fps",
        ]:
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got {val!r}")
            object.__setattr__(self, name, float(val))


class RollingTimingProfiler:
    """Tracks latency statistics over a sliding window of frames."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._history: deque[StageTiming] = deque(maxlen=window_size)

    def record(self, timing: StageTiming) -> None:
        """Record frame timing."""
        self._history.append(timing)

    @property
    def sample_count(self) -> int:
        return len(self._history)

    @property
    def average_latency_ms(self) -> float:
        if not self._history:
            return 0.0
        return sum(t.total_ms for t in self._history) / len(self._history)

    @property
    def max_latency_ms(self) -> float:
        if not self._history:
            return 0.0
        return max(t.total_ms for t in self._history)

    @property
    def min_latency_ms(self) -> float:
        if not self._history:
            return 0.0
        return min(t.total_ms for t in self._history)

    @property
    def average_fps(self) -> float:
        avg_lat = self.average_latency_ms
        if avg_lat <= 0.0:
            return 0.0
        return 1000.0 / avg_lat

    def stage_averages(self) -> dict[str, float]:
        """Compute mean latency in milliseconds for each pipeline stage."""
        if not self._history:
            return {
                "perception": 0.0,
                "tracking": 0.0,
                "estimation": 0.0,
                "prediction": 0.0,
                "risk": 0.0,
                "decision": 0.0,
                "total": 0.0,
            }
        n = len(self._history)
        return {
            "perception": sum(t.perception_ms for t in self._history) / n,
            "tracking": sum(t.tracking_ms for t in self._history) / n,
            "estimation": sum(t.estimation_ms for t in self._history) / n,
            "prediction": sum(t.prediction_ms for t in self._history) / n,
            "risk": sum(t.risk_ms for t in self._history) / n,
            "decision": sum(t.decision_ms for t in self._history) / n,
            "total": sum(t.total_ms for t in self._history) / n,
        }
