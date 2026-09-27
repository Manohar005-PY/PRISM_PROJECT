"""Unit tests for Detector and FrameSource protocols."""

import unittest

from app.perception.adapters.synthetic import SyntheticDetector, SyntheticFrameSource
from app.perception.detector import Detector, FrameSource


class TestDetectorProtocols(unittest.TestCase):
    """Test suite for Detector protocol conformance."""

    def test_synthetic_detector_protocol_conformance(self) -> None:
        detector = SyntheticDetector()
        self.assertIsInstance(detector, Detector)

    def test_synthetic_frame_source_conformance(self) -> None:
        source = SyntheticFrameSource([("frame_data", 0.0), ("frame_data_2", 0.1)])
        self.assertIsInstance(source, FrameSource)
        self.assertTrue(source.has_frames())
        f1, t1 = source.get_frame()
        self.assertEqual(t1, 0.0)
        self.assertTrue(source.has_frames())
        f2, t2 = source.get_frame()
        self.assertEqual(t2, 0.1)
        self.assertFalse(source.has_frames())


if __name__ == "__main__":
    unittest.main()
