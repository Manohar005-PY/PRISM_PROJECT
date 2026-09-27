"""Unit and integration tests for PRISM Embedded Runtime Bridge."""

import json
import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction
from app.perception.model import PhysicalObservation
from app.perception.pipeline import PerceptionPipeline
from app.presentation.engine import PresentationEngine
from app.runtime.config import RuntimeConfig
from app.runtime.embedded_bridge import (
    EmbeddedBridgeConfig,
    EmbeddedPrismBridge,
    EmbeddedRuntimeState,
)
from app.runtime.pipeline import PRISMPipeline


class FakePerceptionPipeline(PerceptionPipeline):
    """Headless perception pipeline returning empty observations by default."""
    def __init__(self) -> None:
        pass
    def process_frame(self, raw_frame, timestamp):
        return [], []


class TestEmbeddedPrismBridge(unittest.TestCase):
    """Tests evaluating embedded runtime lifecycle, JSON serialization, and safety degradation."""

    def setUp(self) -> None:
        pipeline = PRISMPipeline(
            perception_pipeline=FakePerceptionPipeline(),
            config=RuntimeConfig(),
        )
        pres_engine = PresentationEngine()
        config = EmbeddedBridgeConfig(stale_frame_threshold_s=0.25)
        self.bridge = EmbeddedPrismBridge(pipeline, pres_engine, config)

    def test_lifecycle_transitions(self) -> None:
        self.assertEqual(self.bridge.state, EmbeddedRuntimeState.STOPPED)
        self.assertTrue(self.bridge.start())
        self.assertEqual(self.bridge.state, EmbeddedRuntimeState.RUNNING)
        self.assertTrue(self.bridge.stop())
        self.assertEqual(self.bridge.state, EmbeddedRuntimeState.STOPPED)

    def test_safe_frame_processing_payload(self) -> None:
        self.bridge.start()
        res = self.bridge.process_observations_payload([], timestamp=0.0)
        self.assertEqual(res["runtime_state"], "RUNNING")
        self.assertEqual(res["decision"]["action"], "continue")
        self.assertEqual(res["risk"]["risk_level"], "safe")
        self.assertEqual(res["health"]["status"], "healthy")
        self.assertEqual(len(res["presentation_cues"]), 1)
        self.assertEqual(res["presentation_cues"][0]["channel"], "visual")

    def test_obstacle_triggers_evasion_and_multimodal_cues(self) -> None:
        self.bridge.start()
        # Approach obstacle across multiple frames from 4.0m down to 1.8m
        for i in range(15):
            t = i * 0.1
            z_val = 4.0 - (i * 0.15)
            obs = [
                PhysicalObservation(
                    position=Vector2(0.2, z_val),
                    measurement_covariance=((0.05, 0.0), (0.0, 0.05)),
                    confidence=0.95,
                    timestamp=t,
                    detection_id=1,
                    class_name="person",
                )
            ]
            res = self.bridge.process_observations_payload(obs, timestamp=t)

        self.assertIn(res["decision"]["action"], ["turn_left", "turn_right", "slow_down", "caution", "stop", "emergency_stop"])
        channels = [c["channel"] for c in res["presentation_cues"]]
        self.assertTrue(any(ch in ["speech", "spatial_audio", "haptic", "visual"] for ch in channels))

    def test_stale_frame_triggers_degraded_state(self) -> None:
        self.bridge.start()
        self.bridge.process_observations_payload([], timestamp=0.0)

        # Gap of 0.5s > stale_frame_threshold_s (0.25s)
        res_stale = self.bridge.process_observations_payload([], timestamp=0.50)
        self.assertEqual(self.bridge.state, EmbeddedRuntimeState.DEGRADED)

    def test_out_of_order_timestamp_is_safely_rejected(self) -> None:
        self.bridge.start()
        self.bridge.process_observations_payload([], timestamp=1.0)

        # Out-of-order timestamp t=0.5 < 1.0
        res_err = self.bridge.process_observations_payload([], timestamp=0.5)
        self.assertEqual(res_err["decision"]["action"], "emergency_stop")
        self.assertEqual(self.bridge.dropped_frames, 1)

    def test_json_payload_bridge_interface(self) -> None:
        self.bridge.start()
        req_json = json.dumps({
            "timestamp": 1.0,
            "observations": [
                {"x": 0.0, "z": 2.0, "covariance": 0.05, "confidence": 0.95, "class_name": "person"}
            ]
        })
        resp_json_str = self.bridge.process_json_payload(req_json)
        resp = json.loads(resp_json_str)
        self.assertEqual(resp["frame_id"], 1)
        self.assertIn("decision", resp)
        self.assertIn("risk", resp)
        self.assertIn("timing", resp)


if __name__ == "__main__":
    unittest.main()
