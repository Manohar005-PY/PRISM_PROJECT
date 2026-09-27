"""Real-world performance metrics: Mean, Median, P95, P99 latency, FPS, and drop rate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class RealtimeMetrics:
    """Quantitative performance statistics for real-time camera and ML execution."""

    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    yolo_latency_ms: float
    input_fps: float
    processed_fps: float
    dropped_frames: int
    drop_rate: float

    def __post_init__(self) -> None:
        for name in [
            "mean_latency_ms",
            "median_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "yolo_latency_ms",
            "input_fps",
            "processed_fps",
            "drop_rate",
        ]:
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got {val!r}")
            object.__setattr__(self, name, float(val))
        if not isinstance(self.dropped_frames, int) or self.dropped_frames < 0:
            raise ValueError(f"dropped_frames must be non-negative int, got {self.dropped_frames!r}")


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    """Compute exact percentile value (0.0 to 100.0) from a sequence of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


class RealtimePerformanceTracker:
    """Monitors real-time frame timestamps, latencies, and percentiles."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._total_latencies: deque[float] = deque(maxlen=window_size)
        self._yolo_latencies: deque[float] = deque(maxlen=window_size)
        self._capture_timestamps: deque[float] = deque(maxlen=window_size)
        self._processed_timestamps: deque[float] = deque(maxlen=window_size)
        self._dropped_frames: int = 0
        self._total_captured: int = 0

    def record_capture(self, timestamp: float) -> None:
        """Record sensor acquisition event."""
        self._total_captured += 1
        self._capture_timestamps.append(timestamp)

    def record_processed(self, total_latency_ms: float, yolo_latency_ms: float, completion_time: float) -> None:
        """Record frame completion metrics."""
        self._total_latencies.append(total_latency_ms)
        self._yolo_latencies.append(yolo_latency_ms)
        self._processed_timestamps.append(completion_time)

    def record_drop(self) -> None:
        """Record a dropped frame."""
        self._dropped_frames += 1

    def compute_metrics(self) -> RealtimeMetrics:
        """Calculate comprehensive throughput and tail-latency percentiles."""
        if not self._total_latencies:
            return RealtimeMetrics(
                mean_latency_ms=0.0,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                yolo_latency_ms=0.0,
                input_fps=0.0,
                processed_fps=0.0,
                dropped_frames=self._dropped_frames,
                drop_rate=0.0,
            )

        n = len(self._total_latencies)
        mean_lat = sum(self._total_latencies) / n
        median_lat = compute_percentile(self._total_latencies, 50.0)
        p95_lat = compute_percentile(self._total_latencies, 95.0)
        p99_lat = compute_percentile(self._total_latencies, 99.0)
        mean_yolo = sum(self._yolo_latencies) / len(self._yolo_latencies) if self._yolo_latencies else 0.0

        # Compute effective input FPS
        input_fps = 0.0
        if len(self._capture_timestamps) > 1:
            span = self._capture_timestamps[-1] - self._capture_timestamps[0]
            if span > 0.0:
                input_fps = (len(self._capture_timestamps) - 1) / span

        # Compute effective processed FPS
        processed_fps = 0.0
        if len(self._processed_timestamps) > 1:
            span = self._processed_timestamps[-1] - self._processed_timestamps[0]
            if span > 0.0:
                processed_fps = (len(self._processed_timestamps) - 1) / span

        drop_rate = 0.0
        if self._total_captured > 0:
            drop_rate = (self._dropped_frames / self._total_captured) * 100.0

        return RealtimeMetrics(
            mean_latency_ms=mean_lat,
            median_latency_ms=median_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            yolo_latency_ms=mean_yolo,
            input_fps=input_fps,
            processed_fps=processed_fps,
            dropped_frames=self._dropped_frames,
            drop_rate=drop_rate,
        )
