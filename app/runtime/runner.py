"""PRISMRunner: Frame loop runner supporting sequential processing, frame drop policies, and graceful shutdown."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Iterator

from app.prediction.relative_motion import EgoMotionState
from app.runtime.config import RuntimeConfig
from app.runtime.model import RuntimeResult
from app.runtime.pipeline import PRISMPipeline


class PRISMRunner:
    """Controls execution of PRISMPipeline across frame streams with bounded latency management."""

    def __init__(
        self,
        pipeline: PRISMPipeline,
        config: RuntimeConfig | None = None,
        callback: Callable[[RuntimeResult], None] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config if config is not None else pipeline.config
        self.callback = callback
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def stop(self) -> None:
        """Signal graceful shutdown of runner."""
        self._is_running = False

    def run_sequence(
        self,
        frame_sequence: Iterable[tuple[Any, float]],
        ego_sequence: Iterable[EgoMotionState | None] | None = None,
        max_frames: int | None = None,
    ) -> list[RuntimeResult]:
        """Execute pipeline over a pre-recorded or synthetic frame sequence.

        Args:
            frame_sequence: Iterable yielding (raw_frame, timestamp) pairs.
            ego_sequence: Optional iterable yielding corresponding EgoMotionState objects.
            max_frames: Maximum number of frames to process before stopping.

        Returns:
            List of collected RuntimeResult instances.
        """
        self._is_running = True
        results: list[RuntimeResult] = []
        ego_iter: Iterator[EgoMotionState | None] | None = iter(ego_sequence) if ego_sequence is not None else None

        for count, (raw_frame, timestamp) in enumerate(frame_sequence):
            if not self._is_running:
                break
            if max_frames is not None and count >= max_frames:
                break

            ego_st = next(ego_iter) if ego_iter is not None else None
            res = self.pipeline.process_frame(raw_frame, timestamp, ego_state=ego_st)
            results.append(res)

            if self.callback is not None:
                self.callback(res)

        self._is_running = False
        return results
