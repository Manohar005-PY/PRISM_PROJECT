"""Integration test demonstrating PRISM foundational data flow.

Proves the contracts and transformations between core domain stages:
ObjectObservation -> ObjectState -> Trajectory -> RiskAssessment -> Decision.
"""

import unittest

from app.core.enums import DecisionType, RiskLevel
from app.core.types import (
    BoundingBox,
    Covariance4x4,
    Decision,
    ObjectObservation,
    ObjectState,
    RiskAssessment,
    Trajectory,
    TrajectoryPoint,
    Vector2,
    WorldState,
)


class TestPipelineDataFlowIntegration(unittest.TestCase):
    """Integration test verifying end-to-end data contracts across the pipeline."""

    def test_pipeline_data_flow(self) -> None:
        # 1. Perception Stage: ObjectObservation
        # "What did the camera see?"
        bbox = BoundingBox(x_min=300.0, y_min=100.0, x_max=380.0, y_max=400.0)
        observation = ObjectObservation(
            object_id=1,
            class_id="person",
            confidence=0.95,
            bbox=bbox,
            timestamp=100.0,
        )
        self.assertEqual(observation.object_id, 1)
        self.assertEqual(observation.class_id, "person")
        self.assertEqual(observation.bbox.bottom_center, (340.0, 400.0))

        # 2. State Estimation Stage: ObjectState & WorldState
        # "Where is the person on the ground plane, and how are they moving?"
        cov: Covariance4x4 = (
            (0.05, 0.0, 0.0, 0.0),
            (0.0, 0.05, 0.0, 0.0),
            (0.0, 0.0, 0.1, 0.0),
            (0.0, 0.0, 0.0, 0.1),
        )
        state = ObjectState(
            object_id=observation.object_id,
            class_id=observation.class_id,
            position=Vector2(x=0.0, z=5.0),  # 5 meters forward directly in path
            velocity=Vector2(x=0.0, z=-1.0),  # Moving toward user at 1 m/s
            timestamp=observation.timestamp,
            covariance=cov,
            confidence=0.95,
            age=5,
            time_since_update=0.0,
        )
        world_state = WorldState(
            timestamp=observation.timestamp,
            objects={state.object_id: state},
        )
        self.assertIn(1, world_state.objects)
        self.assertEqual(world_state.objects[1].position.z, 5.0)

        # 3. Motion Prediction Stage: Trajectory & TrajectoryPoints
        # "Where do we expect the person to be over the next 3 seconds?"
        dt1 = 1.0
        dt2 = 2.0
        dt3 = 3.0
        traj_points = [
            TrajectoryPoint(
                timestamp=state.timestamp + dt1,
                position=state.position + state.velocity * dt1,  # (0.0, 4.0)
                velocity=state.velocity,
                covariance=cov,
            ),
            TrajectoryPoint(
                timestamp=state.timestamp + dt2,
                position=state.position + state.velocity * dt2,  # (0.0, 3.0)
                velocity=state.velocity,
                covariance=cov,
            ),
            TrajectoryPoint(
                timestamp=state.timestamp + dt3,
                position=state.position + state.velocity * dt3,  # (0.0, 2.0)
                velocity=state.velocity,
                covariance=cov,
            ),
        ]
        trajectory = Trajectory(
            object_id=state.object_id,
            points=traj_points,
            prediction_horizon=3.0,
        )
        self.assertEqual(len(trajectory.points), 3)
        self.assertEqual(trajectory.points[0].position, Vector2(0.0, 4.0))
        self.assertEqual(trajectory.points[2].position, Vector2(0.0, 2.0))

        # 4. Risk Assessment Stage: RiskAssessment
        # "Does this predicted motion represent danger?"
        risk_assessment = RiskAssessment(
            object_id=trajectory.object_id,
            risk_level=RiskLevel.HIGH,
            risk_score=85.0,
            ttc=1.4,
            cpa_distance=0.35,
            confidence=0.95,
            timestamp=state.timestamp,
        )
        self.assertEqual(risk_assessment.risk_level, RiskLevel.HIGH)
        self.assertAlmostEqual(risk_assessment.risk_score, 85.0)
        self.assertEqual(risk_assessment.ttc, 1.4)
        self.assertEqual(risk_assessment.cpa_distance, 0.35)

        # 5. Decision Stage: Decision
        # "What should the assistive system do about it?"
        decision = Decision(
            decision_type=DecisionType.WARNING,
            object_id=risk_assessment.object_id,
            timestamp=risk_assessment.timestamp,
            message_id="pedestrian_approaching_ahead",
        )
        self.assertEqual(decision.decision_type, DecisionType.WARNING)
        self.assertEqual(decision.object_id, 1)
        self.assertEqual(decision.message_id, "pedestrian_approaching_ahead")


if __name__ == "__main__":
    unittest.main()
