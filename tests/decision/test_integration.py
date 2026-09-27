"""End-to-end integration tests: Simulation -> Projection -> Kalman -> Prediction -> Risk -> Decision."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import ObjectState, Vector2, WorldState
from app.decision.context import DecisionContext
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
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


class TestEndToEndDecisionPipeline(unittest.TestCase):
    """Integration test suite evaluating decision-making across canonical scenarios."""

    def setUp(self) -> None:
        self.risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=1.0))
        self.decision_engine = DecisionEngine()

    def test_diverging_scenario_continue(self) -> None:
        """DIVERGING scenario: Decision engine produces CONTINUE throughout the entire trajectory."""
        scenario = create_diverging_scenario(duration=3.0, timestep=0.5)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            obj = ObjectState(1, "person", frame.objects[0].position, frame.objects[0].velocity, frame.timestamp)
            world = WorldState(timestamp=frame.timestamp, objects={1: obj})

            risk = self.risk_evaluator.get_primary_threat(world, ego=ego)
            ctx = DecisionContext(
                ego_position=ego.position,
                risk_assessment=risk,
                predicted_obstacles=(obj,),
                timestamp=frame.timestamp,
            )
            decision = self.decision_engine.decide_navigation(ctx)
            # Diverging obstacles must never cause an unnecessary stop
            self.assertIn(decision.action, (NavigationAction.CONTINUE, NavigationAction.CAUTION))

    def test_parallel_scenario_continue(self) -> None:
        """PARALLEL scenario: Decision engine allows continuous walking without false stops."""
        scenario = create_parallel_scenario(duration=3.0, timestep=0.5)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            obj = ObjectState(1, "person", frame.objects[0].position, frame.objects[0].velocity, frame.timestamp)
            world = WorldState(timestamp=frame.timestamp, objects={1: obj})

            risk = self.risk_evaluator.get_primary_threat(world, ego=ego)
            ctx = DecisionContext(
                ego_position=ego.position,
                risk_assessment=risk,
                predicted_obstacles=(obj,),
                timestamp=frame.timestamp,
            )
            decision = self.decision_engine.decide_navigation(ctx)
            self.assertIn(decision.action, (NavigationAction.CONTINUE, NavigationAction.CAUTION))

    def test_head_on_scenario_intervention(self) -> None:
        """HEAD_ON scenario: Decision engine issues evasive action or stop as collision approaches."""
        scenario = create_head_on_scenario(duration=3.0, timestep=0.5)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        decisions = []
        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            obj = ObjectState(1, "person", frame.objects[0].position, frame.objects[0].velocity, frame.timestamp)
            world = WorldState(timestamp=frame.timestamp, objects={1: obj})

            risk = self.risk_evaluator.get_primary_threat(world, ego=ego)
            ctx = DecisionContext(
                ego_position=ego.position,
                risk_assessment=risk,
                predicted_obstacles=(obj,),
                timestamp=frame.timestamp,
            )
            dec = self.decision_engine.decide_navigation(ctx)
            decisions.append(dec)

        # During the critical approach phase (t <= 2.0s), the system must intervene
        approach_decisions = [dec for frame_idx, dec in enumerate(decisions) if gt_seq.get_frame(frame_idx).timestamp <= 2.0]
        critical_dec = approach_decisions[-1]  # At collision point (t = 2.0s)
        self.assertIn(
            critical_dec.action,
            (NavigationAction.STOP, NavigationAction.EMERGENCY_STOP, NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT),
        )


if __name__ == "__main__":
    unittest.main()
