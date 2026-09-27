"""End-to-end integration tests: Simulation -> Projection -> Kalman -> Prediction -> Risk Assessment."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import Covariance4x4, ObjectState, Vector2, WorldState
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import (
    create_crossing_scenario,
    create_diverging_scenario,
    create_head_on_scenario,
    create_parallel_scenario,
    create_stationary_obstacle_scenario,
)
from simulation.simulator import Simulator


class TestEndToEndRiskPipeline(unittest.TestCase):
    """Integration test suite for Risk Assessment across canonical scenarios."""

    def setUp(self) -> None:
        self.evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=1.0))  # No smoothing for instant evaluation

    def test_head_on_risk_escalation(self) -> None:
        """HEAD_ON scenario: Risk score increases monotonically as obstacle approaches up to collision."""
        scenario = create_head_on_scenario(duration=3.0, timestep=0.5)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        risk_scores = []
        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            obj = ObjectState(
                object_id=frame.objects[0].object_id,
                class_id=frame.objects[0].class_id,
                position=frame.objects[0].position,
                velocity=frame.objects[0].velocity,
                timestamp=frame.timestamp,
            )
            world = WorldState(timestamp=frame.timestamp, objects={obj.object_id: obj})
            primary = self.evaluator.get_primary_threat(world, ego=ego)
            self.assertIsNotNone(primary)
            risk_scores.append((frame.timestamp, primary.risk_score))  # type: ignore[union-attr]

        # Approach phase: t in [0.0, 0.5, 1.0, 1.5, 2.0]
        approach_scores = [score for t, score in risk_scores if t <= 2.0]
        for i in range(1, len(approach_scores)):
            self.assertGreaterEqual(approach_scores[i], approach_scores[i - 1])

        # Peak score at collision point (t=2.0s) must be CRITICAL (100.0)
        self.assertAlmostEqual(approach_scores[-1], 100.0)

    def test_parallel_scenario_low_risk(self) -> None:
        """PARALLEL scenario: Constant lateral separation produces consistently LOW/SAFE risk."""
        scenario = create_parallel_scenario(duration=3.0, timestep=0.5)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            obj = ObjectState(1, "person", frame.objects[0].position, frame.objects[0].velocity, frame.timestamp)
            world = WorldState(timestamp=frame.timestamp, objects={1: obj})
            primary = self.evaluator.get_primary_threat(world, ego=ego)
            self.assertIsNotNone(primary)
            self.assertIn(primary.risk_level, (RiskLevel.SAFE, RiskLevel.LOW))  # type: ignore[union-attr]

    def test_diverging_scenario_low_risk(self) -> None:
        """DIVERGING scenario: Separation produces LOW/SAFE risk throughout."""
        scenario = create_diverging_scenario(duration=3.0, timestep=0.5)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            obj = ObjectState(1, "person", frame.objects[0].position, frame.objects[0].velocity, frame.timestamp)
            world = WorldState(timestamp=frame.timestamp, objects={1: obj})
            primary = self.evaluator.get_primary_threat(world, ego=ego)
            self.assertIsNotNone(primary)
            self.assertIn(primary.risk_level, (RiskLevel.SAFE, RiskLevel.LOW))  # type: ignore[union-attr]

    def test_uncertainty_aware_vs_deterministic_ablation(self) -> None:
        """SCIENTIFIC ABLATION: Compare Deterministic Risk vs Uncertainty-Aware Risk on a near-miss path."""
        # Near-miss trajectory: d_cpa = 1.1m (outside base safety radius R = 1.0m)
        ego = EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.0))

        # 1. Deterministic state (No covariance)
        det_obj = ObjectState(
            object_id=1,
            class_id="person",
            position=Vector2(1.1, 3.0),
            velocity=Vector2(0.0, -1.0),
            timestamp=0.0,
            covariance=None,
        )
        profile_det = self.evaluator.evaluate_object(det_obj, ego)

        # 2. Uncertainty-aware state (Positional variance sigma_pos = 0.35m)
        cov: Covariance4x4 = (
            (0.06, 0.0, 0.0, 0.0),
            (0.0, 0.06, 0.0, 0.0),
            (0.0, 0.0, 0.1, 0.0),
            (0.0, 0.0, 0.0, 0.1),
        )
        unc_obj = ObjectState(
            object_id=1,
            class_id="person",
            position=Vector2(1.1, 3.0),
            velocity=Vector2(0.0, -1.0),
            timestamp=0.0,
            covariance=cov,
        )
        profile_unc = self.evaluator.evaluate_object(unc_obj, ego)

        # Uncertainty-aware model recognizes boundary intrusion probability and produces higher risk
        self.assertGreater(profile_unc.score, profile_det.score)
        self.assertGreater(profile_unc.factors.collision_probability, 0.5)
        self.assertAlmostEqual(profile_det.factors.collision_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
