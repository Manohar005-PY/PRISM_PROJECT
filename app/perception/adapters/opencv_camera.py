"""OpenCV Camera Adapter: Encapsulates OpenCV video capture without leaking dependencies into core domain."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Protocol, runtime_checkable

from app.runtime.realtime.camera import CameraFrame


@runtime_checkable
class CameraFrameSource(Protocol):
    """Protocol for physical or synthetic frame acquisition sources."""

    def open(self) -> bool: ...
    def read_frame(self) -> CameraFrame | None: ...
    def release(self) -> None: ...
    def is_opened(self) -> bool: ...


class OpenCVCameraAdapter:
    """Concrete frame source wrapping OpenCV VideoCapture."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 30.0,
        buffer_size: int = 1,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.buffer_size = buffer_size

        self._cap: Any = None
        self._frame_counter: int = 0
        self._is_opened: bool = False

    def open(self) -> bool:
        """Initialize physical video capture."""
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            self._is_opened = False
            return False

        try:
            self._cap = cv2.VideoCapture(self.device_index)
            if not self._cap.isOpened():
                self._is_opened = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self._is_opened = True
            return True
        except Exception:
            self._is_opened = False
            return False

    def read_frame(self) -> CameraFrame | None:
        """Read a single frame with high-precision monotonic timestamp."""
        if not self._is_opened or self._cap is None:
            return None

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
        """Release underlying camera resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened


class FakeCameraFrameSource:
    """Deterministic mock camera source for automated testing without physical hardware."""

    def __init__(self, frames_count: int = 10, width: int = 640, height: int = 480) -> None:
        self.frames_count = frames_count
        self.width = width
        self.height = height
        self._frame_counter = 0
        self._is_opened = False

    def open(self) -> bool:
        self._is_opened = True
        self._frame_counter = 0
        return True

    def read_frame(self) -> CameraFrame | None:
        if not self._is_opened or self._frame_counter >= self.frames_count:
            return None

        self._frame_counter += 1
        return CameraFrame(
            frame=None,
            timestamp=self._frame_counter * 0.033,
            frame_id=self._frame_counter,
            width=self.width,
            height=self.height,
        )

    def release(self) -> None:
        self._is_opened = False

    def is_opened(self) -> bool:
        return self._is_opened
