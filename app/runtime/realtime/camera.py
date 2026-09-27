"""Real-world camera acquisition, monotonic acquisition timestamps, and mock sources."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Protocol, runtime_checkable

from app.runtime.realtime.config import CameraConfig


@dataclass(frozen=True)
class CameraFrame:
    """Immutable representation of a captured raw camera frame with acquisition timestamp."""

    frame: Any
    timestamp: float
    frame_id: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative integer, got {self.frame_id!r}")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"width must be positive int, got {self.width!r}")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"height must be positive int, got {self.height!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))


@runtime_checkable
class CameraSource(Protocol):
    """Protocol for video frame acquisition sources."""

    def open(self) -> bool: ...
    def read_frame(self) -> CameraFrame | None: ...
    def release(self) -> None: ...
    def is_opened(self) -> bool: ...


class OpenCVCameraSource:
    """Concrete camera source wrapping OpenCV cv2.VideoCapture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config: CameraConfig = config if config is not None else CameraConfig()
        self._cap: Any = None
        self._frame_counter: int = 0
        self._is_opened: bool = False

    def open(self) -> bool:
        """Initialize physical video capture device."""
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            self._is_opened = False
            return False

        try:
            self._cap = cv2.VideoCapture(self.config.device_index)
            if not self._cap.isOpened():
                self._is_opened = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            self._is_opened = True
            return True
        except Exception:
            self._is_opened = False
            return False

    def read_frame(self) -> CameraFrame | None:
        """Acquire a raw frame from the physical camera with capture timestamp."""
        if not self._is_opened or self._cap is None:
            return None

        # Capture timestamp recorded immediately at sensor readout
        capture_time = time.monotonic()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        self._frame_counter += 1
        h, w = frame.shape[:2]

        return CameraFrame(
            frame=frame,
            timestamp=capture_time,
            frame_id=self._frame_counter,
            width=w,
            height=h,
        )

    def release(self) -> None:
        """Cleanly release video capture resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened


class MockCameraSource:
    """Deterministic mock camera source generating synthetic frames for automated tests."""

    def __init__(
        self,
        config: CameraConfig | None = None,
        frame_generator: Any | None = None,
    ) -> None:
        self.config: CameraConfig = config if config is not None else CameraConfig()
        self.frame_generator = frame_generator
        self._frame_counter: int = 0
        self._is_opened: bool = False

    def open(self) -> bool:
        self._is_opened = True
        return True

    def read_frame(self) -> CameraFrame | None:
        if not self._is_opened:
            return None

        capture_time = time.monotonic()
        self._frame_counter += 1

        raw = None
        if self.frame_generator is not None:
            try:
                raw = next(self.frame_generator)
            except (StopIteration, TypeError):
                raw = None

        return CameraFrame(
            frame=raw,
            timestamp=capture_time,
            frame_id=self._frame_counter,
            width=self.config.width,
            height=self.config.height,
        )

    def release(self) -> None:
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened
