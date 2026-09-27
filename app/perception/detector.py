"""Structural protocols for object detectors and video/camera frame sources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.perception.model import Detection


@runtime_checkable
class Detector(Protocol):
    """Abstraction for 2D visual object detectors (Synthetic, YOLO, MobileNet, etc.)."""

    def detect(self, frame: object, timestamp: float) -> list[Detection]:
        """Detect objects in a raw image frame and produce image-space Detections.

        Args:
            frame: Raw camera frame or synthetic scene descriptor.
            timestamp: Monotonic capture timestamp in seconds.

        Returns:
            List of 2D image-space Detection instances.
        """
        ...


@runtime_checkable
class FrameSource(Protocol):
    """Abstraction for image/video streams (Synthetic sequences, Webcam, Video file)."""

    def get_frame(self) -> tuple[object, float]:
        """Fetch the next frame and its timestamp.

        Returns:
            (frame_data, timestamp) tuple.
        """
        ...

    def has_frames(self) -> bool:
        """Return True if more frames are available."""
        ...
