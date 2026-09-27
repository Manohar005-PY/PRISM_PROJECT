"""Configuration parameters for real-time camera capture and ML inference."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for physical or virtual video camera capture.

    Attributes:
        device_index: OpenCV video capture device index (typically 0 for built-in webcam).
        width: Capture resolution width in pixels.
        height: Capture resolution height in pixels.
        target_fps: Desired camera sensor acquisition frame rate.
        backend: Optional capture backend identifier ("AUTO", "DSHOW", "V4L2", etc.).
        buffer_size: Internal camera driver frame buffer queue size.
    """

    device_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: float = 30.0
    backend: str = "AUTO"
    buffer_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.device_index, int) or self.device_index < 0:
            raise ValueError(f"device_index must be a non-negative int, got {self.device_index!r}")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"width must be a positive int, got {self.width!r}")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"height must be a positive int, got {self.height!r}")
        if not isinstance(self.target_fps, (int, float)) or not math.isfinite(self.target_fps) or self.target_fps <= 0.0:
            raise ValueError(f"target_fps must be a positive float, got {self.target_fps!r}")
        if not isinstance(self.buffer_size, int) or self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be a positive int, got {self.buffer_size!r}")
        if not isinstance(self.backend, str):
            raise TypeError(f"backend must be str, got {type(self.backend)}")

        object.__setattr__(self, "target_fps", float(self.target_fps))


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for deep learning object detector inference (e.g. YOLOv8).

    Attributes:
        model_path: Path or identifier for the pretrained weights model file.
        device: Target execution device ('auto', 'cuda', 'cpu', 'mps').
        confidence_threshold: Minimum detection confidence score in [0.0, 1.0].
        iou_threshold: Non-Maximum Suppression (NMS) intersection-over-union threshold.
        image_size: Square inference image resolution in pixels (e.g. 640).
        target_classes: Optional filter tuple of semantic class names to detect.
    """

    model_path: str = "yolov8n.pt"
    device: str = "auto"
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45
    image_size: int = 640
    target_classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError(f"model_path must be non-empty string, got {self.model_path!r}")
        if not isinstance(self.device, str):
            raise TypeError(f"device must be string, got {type(self.device)}")
        for name, val in [("confidence_threshold", self.confidence_threshold), ("iou_threshold", self.iou_threshold)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be float in [0.0, 1.0], got {val!r}")
        if not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError(f"image_size must be positive int, got {self.image_size!r}")

        object.__setattr__(self, "confidence_threshold", float(self.confidence_threshold))
        object.__setattr__(self, "iou_threshold", float(self.iou_threshold))
        if self.target_classes is not None:
            object.__setattr__(self, "target_classes", tuple(self.target_classes))


@dataclass(frozen=True)
class RealtimeConfig:
    """Overall real-time runtime configuration integrating camera, ML inference, and backpressure.

    Attributes:
        camera: Camera sensor capture settings.
        inference: Object detection model inference settings.
        max_frame_age: Maximum allowable age of captured frame before dropping (seconds).
        max_processing_latency: Maximum allowable total pipeline execution latency (seconds).
        max_consecutive_camera_failures: Maximum camera capture drops before health failure.
        max_consecutive_inference_failures: Maximum ML inference errors before health failure.
        enable_display: Whether to render an OpenCV debug visualization window.
    """

    camera: CameraConfig = CameraConfig()
    inference: InferenceConfig = InferenceConfig()
    max_frame_age: float = 0.20
    max_processing_latency: float = 0.10
    max_consecutive_camera_failures: int = 3
    max_consecutive_inference_failures: int = 3
    enable_display: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraConfig):
            raise TypeError(f"camera must be CameraConfig, got {type(self.camera)}")
        if not isinstance(self.inference, InferenceConfig):
            raise TypeError(f"inference must be InferenceConfig, got {type(self.inference)}")
        for name, val in [("max_frame_age", self.max_frame_age), ("max_processing_latency", self.max_processing_latency)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val!r}")
        for name, val in [
            ("max_consecutive_camera_failures", self.max_consecutive_camera_failures),
            ("max_consecutive_inference_failures", self.max_consecutive_inference_failures),
        ]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be positive int, got {val!r}")
        if not isinstance(self.enable_display, bool):
            raise TypeError(f"enable_display must be bool, got {type(self.enable_display)}")

        object.__setattr__(self, "max_frame_age", float(self.max_frame_age))
        object.__setattr__(self, "max_processing_latency", float(self.max_processing_latency))
