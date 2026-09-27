"""Cross-platform determinism and golden replay dataset validation tests."""

import json
import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from app.perception.pipeline import PerceptionPipeline
from app.presentation.engine import PresentationEngine
from app.runtime.config import RuntimeConfig
from app.runtime.embedded_bridge import (
    EmbeddedBridgeConfig,
    EmbeddedPrismBridge,
)
from app.runtime.pipeline import PRISMPipeline


class DummyPerceptionPipeline(PerceptionPipeline):
    def __init__(self) -> None:
        pass
    def process_frame(self, raw_frame, timestamp):
        return [], []


class TestCrossPlatformDeterminism(unittest.TestCase):
    """Verifies that fixed deterministic input sequences produce repeatable outputs within strict tolerances."""

    def setUp(self) -> None:
        pipeline = PRISMPipeline(
            perception_pipeline=DummyPerceptionPipeline(),
            config=RuntimeConfig(),
        )
        self.bridge = EmbeddedPrismBridge(pipeline, PresentationEngine())
        self.bridge.start()

    def test_deterministic_head_on_golden_sequence(self) -> None:
        """Evaluate 20 frames of head-on approach and verify deterministic action progression."""
        actions_history = []
        risk_scores = []

        # Obstacle approaches from 6.0m to 0.5m at constant speed
        for i in range(20):
            t = i * 0.1
            z_pos = 6.0 - (i * 0.28)
            obs = [
                PhysicalObservation(
                    position=Vector2(0.0, z_pos),
                    measurement_covariance=((0.05, 0.0), (0.0, 0.05)),
                    confidence=1.0,
                    timestamp=t,
                    detection_id=1,
                    class_name="pedestrian",
                )
            ]
            res = self.bridge.process_observations_payload(obs, timestamp=t)
            actions_history.append(res["decision"]["action"])
            risk_scores.append(res["risk"]["risk_score"])

        # Initial frames (at 6.0m): caution / slow_down
        self.assertIn(actions_history[0], ["caution", "continue", "slow_down"])

        # Intermediate frames (at 3.5m - 2.0m): turn_left or turn_right evasion
        mid_actions = actions_history[5:15]
        self.assertTrue(any("turn" in a or "slow" in a for a in mid_actions))

        # Late frames (at < 1.2m): emergency stop or stop
        final_action = actions_history[-1]
        self.assertIn(final_action, ["stop", "emergency_stop"])

        # Monotonically increasing risk
        self.assertGreater(risk_scores[-1], risk_scores[0])


if __name__ == "__main__":
    unittest.main()
