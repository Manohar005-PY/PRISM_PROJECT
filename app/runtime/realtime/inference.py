"""Real-time ML inference engine abstraction, YOLOv8 adapter, and hardware device diagnostics."""

from __future__ import annotations

import time
from typing import Any, Protocol, Sequence, runtime_checkable

from app.perception.adapters.yolo import YOLODetector
from app.perception.model import Detection
from app.runtime.realtime.config import InferenceConfig


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol for real-time visual object detection inference engines."""

    def infer(self, frame: Any, timestamp: float) -> list[Detection]: ...
    def is_available(self) -> bool: ...
    def device_info(self) -> str: ...


class YOLOInferenceEngine:
    """YOLOv8 deep learning detector adapter with hardware acceleration diagnostics."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config: InferenceConfig = config if config is not None else InferenceConfig()
        self._model: Any = None
        self._device: str = "unavailable"
        self._is_loaded: bool = False
        self._detector: YOLODetector | None = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load Ultralytics YOLO model and detect acceleration device."""
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
            import torch  # type: ignore[import-untyped]

            # Determine device
            req_dev = self.config.device.lower()
            if req_dev == "cuda" and torch.cuda.is_available():
                target_dev = "cuda"
            elif req_dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                target_dev = "mps"
            elif req_dev == "auto":
                target_dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_dev = "cpu"

            self._model = YOLO(self.config.model_path)
            self._device = target_dev
            self._is_loaded = True
        except Exception:
            self._model = None
            self._device = "unavailable"
            self._is_loaded = False

        self._detector = YOLODetector(
            model=self._model,
            confidence_threshold=self.config.confidence_threshold,
            target_classes=self.config.target_classes,
        )

    def is_available(self) -> bool:
        """Check if deep learning model is successfully loaded and ready for inference."""
        return self._is_loaded and self._detector is not None

    def device_info(self) -> str:
        """Return human-readable device acceleration status."""
        return self._device.upper()

    def infer(self, frame: Any, timestamp: float) -> list[Detection]:
        """Execute object detection inference on a single frame.

        Args:
            frame: Camera frame image array (e.g. numpy array from OpenCV or mock).
            timestamp: Sensor acquisition timestamp (seconds).

        Returns:
            List of Part 8 Detection domain objects (isolated from PyTorch/Ultralytics).
        """
        if not self._is_loaded or self._detector is None:
            return []

        try:
            return self._detector.detect(frame, timestamp)
        except Exception:
            return []


class MockInferenceEngine:
    """Deterministic mock inference engine for unit testing and offline regression."""

    def __init__(self, detections_sequence: Sequence[Sequence[Detection]] | None = None) -> None:
        self.detections_sequence = detections_sequence
        self._call_count: int = 0
        self._device: str = "CPU (MOCK)"

    def is_available(self) -> bool:
        return True

    def device_info(self) -> str:
        return self._device

    def infer(self, frame: Any, timestamp: float) -> list[Detection]:
        if self.detections_sequence is None or len(self.detections_sequence) == 0:
            return []

        idx = self._call_count % len(self.detections_sequence)
        self._call_count += 1
        return list(self.detections_sequence[idx])
