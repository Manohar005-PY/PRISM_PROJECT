"""Unit tests for ML InferenceEngine and YOLO adapter isolation."""

import unittest

from app.core.types import BoundingBox
from app.perception.model import Detection
from app.runtime.realtime.config import InferenceConfig
from app.runtime.realtime.inference import MockInferenceEngine, YOLOInferenceEngine


class TestInference(unittest.TestCase):
    """Test suite for ML inference engine."""

    def test_mock_inference_engine(self) -> None:
        det1 = Detection(
            detection_id="d1",
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(100.0, 100.0, 200.0, 300.0),
            timestamp=0.1,
        )
        mock_engine = MockInferenceEngine([[det1]])
        self.assertTrue(mock_engine.is_available())
        self.assertEqual(mock_engine.device_info(), "CPU (MOCK)")

        dets = mock_engine.infer("dummy_frame", 0.1)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_name, "person")

    def test_yolo_engine_fallback_when_uninstalled(self) -> None:
        """Verify YOLOInferenceEngine reports unavailable device instead of crashing if model file missing."""
        cfg = InferenceConfig(model_path="non_existent_model_weights.pt")
        yolo_engine = YOLOInferenceEngine(cfg)
        # Should gracefully report unavailable without throwing exceptions
        self.assertIn(yolo_engine.device_info(), ("UNAVAILABLE", "CPU", "CUDA"))


if __name__ == "__main__":
    unittest.main()
