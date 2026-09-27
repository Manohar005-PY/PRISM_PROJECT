"""Real-world 2D object detector interface supporting YOLO and OpenCV HOG backends."""

from __future__ import annotations

import math
from typing import Any, Sequence

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None

from simulation.real_world.models import RealWorldDetectionBox


class RealWorldDetector:
    """Detects humans/pedestrians in real-world camera video frames."""

    def __init__(
        self,
        detector_type: str = "auto",
        confidence_thresh: float = 0.35,
        model_name: str = "yolov8n.pt",
    ) -> None:
        self.detector_type = detector_type
        self.confidence_thresh = confidence_thresh
        self.model_name = model_name

        self._yolo_model: Any = None
        self._hog_detector: Any = None
        self._active_backend: str = "none"

        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize YOLO if available, otherwise fall back to OpenCV HOG detector."""
        if self.detector_type in ("auto", "yolo"):
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
                self._yolo_model = YOLO(self.model_name)
                self._active_backend = "yolo"
                return
            except Exception:
                if self.detector_type == "yolo":
                    pass  # Will fall through or raise

        if cv2 is not None:
            self._hog_detector = cv2.HOGDescriptor()
            self._hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._active_backend = "hog"
        else:
            self._active_backend = "none"

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def detect(self, bgr_frame: Any) -> list[RealWorldDetectionBox]:
        """Run 2D object detection on a raw BGR image array.
        
        Returns:
            list of RealWorldDetectionBox instances.
        """
        if self._active_backend == "yolo" and self._yolo_model is not None:
            return self._detect_yolo(bgr_frame)
        elif self._active_backend == "hog" and self._hog_detector is not None:
            return self._detect_hog(bgr_frame)
        return []

    def _detect_yolo(self, bgr_frame: Any) -> list[RealWorldDetectionBox]:
        """Detect using YOLO neural network."""
        results = self._yolo_model(bgr_frame, verbose=False, conf=self.confidence_thresh)
        detections: list[RealWorldDetectionBox] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls[0].item())
                # Class 0 is 'person' in COCO dataset
                if cls_id != 0:
                    continue
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist()
                detections.append(
                    RealWorldDetectionBox(
                        x_min=float(xyxy[0]),
                        y_min=float(xyxy[1]),
                        x_max=float(xyxy[2]),
                        y_max=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="person",
                    )
                )
        return detections

    def _detect_hog(self, bgr_frame: Any) -> list[RealWorldDetectionBox]:
        """Detect pedestrians using OpenCV HOG Descriptor."""
        if cv2 is None or self._hog_detector is None:
            return []

        # Resize for faster and more consistent HOG processing if frame is high res
        h, w = bgr_frame.shape[:2]
        boxes, weights = self._hog_detector.detectMultiScale(
            bgr_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )

        detections: list[RealWorldDetectionBox] = []
        for (x, y, bw, bh), weight in zip(boxes, weights):
            conf = float(weight[0]) if hasattr(weight, "__len__") else float(weight)
            # Map HOG SVM score to [0.0, 1.0] sigmoid confidence
            prob = 1.0 / (1.0 + math.exp(-conf * 2.0))
            if prob < self.confidence_thresh:
                continue

            detections.append(
                RealWorldDetectionBox(
                    x_min=float(x),
                    y_min=float(y),
                    x_max=float(x + bw),
                    y_max=float(y + bh),
                    confidence=prob,
                    class_id=0,
                    class_name="person",
                )
            )
        return detections
