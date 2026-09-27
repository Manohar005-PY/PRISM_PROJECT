"""YOLO detector adapter isolating neural-network runtime dependencies from core perception."""

from __future__ import annotations

from typing import Any, Sequence

from app.core.types import BoundingBox
from app.perception.detector import Detector
from app.perception.model import Detection


class YOLODetector:
    """Adapter translating YOLO model inferences into PRISM 2D Detection objects.

    Implements the app.perception.detector.Detector protocol.
    Isolates external ML libraries (Ultralytics, PyTorch) from downstream PRISM modules.
    """

    def __init__(
        self,
        model: Any | None = None,
        confidence_threshold: float = 0.25,
        target_classes: Sequence[int | str] | None = None,
    ) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None

    def parse_yolo_boxes(
        self,
        boxes_data: Sequence[Sequence[float]],
        classes_data: Sequence[int | str],
        confidences_data: Sequence[float],
        timestamp: float,
        class_names: dict[int | str, str] | None = None,
    ) -> list[Detection]:
        """Parse raw bounding box arrays into PRISM Detection instances without requiring YOLO runtime.

        Args:
            boxes_data: Sequence of [x_min, y_min, x_max, y_max] bounding boxes.
            classes_data: Sequence of class IDs (integers or strings).
            confidences_data: Sequence of confidence scores in [0.0, 1.0].
            timestamp: Frame timestamp in seconds.
            class_names: Optional dictionary mapping class ID to human-readable name.

        Returns:
            List of 2D Detection instances.
        """
        detections: list[Detection] = []
        names_map = class_names if class_names is not None else {}

        for idx, (box, cls_id, conf) in enumerate(zip(boxes_data, classes_data, confidences_data)):
            if conf < self.confidence_threshold:
                continue
            if self.target_classes is not None and cls_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            if x2 <= x1 or y2 <= y1:
                continue

            bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
            c_name = names_map.get(cls_id, str(cls_id))

            detections.append(
                Detection(
                    detection_id=f"yolo_{idx}_{int(timestamp * 1000)}",
                    class_id=cls_id,
                    class_name=c_name,
                    confidence=float(conf),
                    bbox=bbox,
                    timestamp=timestamp,
                )
            )

        return detections

    def detect(self, frame: object, timestamp: float) -> list[Detection]:
        """Run YOLO inference on a frame and return Detection objects.

        If model is None or inference produces results, parses the detections.
        """
        if self.model is None:
            # Model-less or test mode
            return []

        try:
            # Standard Ultralytics YOLO call
            results = self.model(frame)
            if not results:
                return []

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            names = getattr(r, "names", {})

            return self.parse_yolo_boxes(
                boxes_data=boxes,
                classes_data=clss,
                confidences_data=confs,
                timestamp=timestamp,
                class_names=names,
            )
        except Exception:
            # Gracefully handle non-fatal inference failures without crashing pipeline
            return []
