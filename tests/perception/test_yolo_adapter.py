"""Unit tests for YOLODetector adapter (model-less and box parsing)."""

import unittest

from app.perception.adapters.yolo import YOLODetector


class TestYOLOAdapter(unittest.TestCase):
    """Test suite for YOLO adapter."""

    def test_parse_yolo_boxes(self) -> None:
        adapter = YOLODetector(confidence_threshold=0.30)

        boxes = [[100.0, 150.0, 250.0, 400.0], [50.0, 50.0, 100.0, 120.0]]
        clss = [0, 1]
        confs = [0.85, 0.20]  # Second box below 0.30 threshold
        names = {0: "person", 1: "bicycle"}

        dets = adapter.parse_yolo_boxes(
            boxes_data=boxes,
            classes_data=clss,
            confidences_data=confs,
            timestamp=5.0,
            class_names=names,
        )

        self.assertEqual(len(dets), 1)
        det = dets[0]
        self.assertEqual(det.class_name, "person")
        self.assertEqual(det.confidence, 0.85)
        self.assertEqual(det.bbox.x_min, 100.0)
        self.assertEqual(det.bbox.y_max, 400.0)

    def test_model_none_detect_returns_empty(self) -> None:
        adapter = YOLODetector(model=None)
        self.assertEqual(adapter.detect(frame=None, timestamp=0.0), [])


if __name__ == "__main__":
    unittest.main()
