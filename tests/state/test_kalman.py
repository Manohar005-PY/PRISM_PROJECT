"""Unit, scenario, covariance invariant, and ablation tests for PRISM Kalman filtering."""

import math
import random
import unittest

from app.core.types import Vector2
from app.state.kalman import (
    KalmanFilterConfig,
    KalmanStateEstimator,
    SingleObjectKalmanFilter,
    invert_2x2,
    mat4_add,
    mat4_mul,
    mat4_transpose,
    mat4_vec4_mul,
    symmetrize_mat4,
)
from app.state.metrics import (
    estimation_improvement,
    measurement_position_rmse,
    position_rmse,
    velocity_rmse,
)
from app.state.noise import (
    ConstantMeasurementNoise,
    DistanceDependentMeasurementNoise,
)


class TestKalmanMatrixAlgebra(unittest.TestCase):
    """Test suite for internal lightweight matrix algebra operations."""

    def test_mat4_mul_identity(self) -> None:
        I4 = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        A = (
            (1.0, 2.0, 3.0, 4.0),
            (5.0, 6.0, 7.0, 8.0),
            (9.0, 1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0, 7.0),
        )
        self.assertEqual(mat4_mul(A, I4), A)
        self.assertEqual(mat4_mul(I4, A), A)

    def test_mat4_transpose_and_symmetrize(self) -> None:
        A = (
            (1.0, 2.0, 3.0, 4.0),
            (5.0, 6.0, 7.0, 8.0),
            (9.0, 1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0, 7.0),
        )
        AT = mat4_transpose(A)
        self.assertEqual(AT[0][1], 5.0)
        self.assertEqual(AT[1][0], 2.0)

        sym = symmetrize_mat4(A)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(sym[i][j], sym[j][i])

    def test_invert_2x2(self) -> None:
        S = ((4.0, 2.0), (2.0, 3.0))  # det = 12 - 4 = 8
        S_inv = invert_2x2(S)
        # S_inv = [[3/8, -2/8], [-2/8, 4/8]] = [[0.375, -0.25], [-0.25, 0.5]]
        self.assertAlmostEqual(S_inv[0][0], 0.375)
        self.assertAlmostEqual(S_inv[0][1], -0.25)
        self.assertAlmostEqual(S_inv[1][0], -0.25)
        self.assertAlmostEqual(S_inv[1][1], 0.5)

    def test_invert_2x2_singular_raises(self) -> None:
        singular_S = ((2.0, 4.0), (1.0, 2.0))  # det = 4 - 4 = 0
        with self.assertRaises(ValueError):
            invert_2x2(singular_S)


class TestSingleObjectKalmanFilter(unittest.TestCase):
    """Test suite for SingleObjectKalmanFilter kinematics, convergence, and covariance."""

    def test_initialization(self) -> None:
        kf = SingleObjectKalmanFilter(
            object_id=1,
            class_id="person",
            initial_position=Vector2(1.0, 5.0),
            timestamp=10.0,
        )
        self.assertEqual(kf.object_id, 1)
        self.assertEqual(kf.class_id, "person")
        self.assertEqual(kf.timestamp, 10.0)
        self.assertEqual(kf.x[0], 1.0)
        self.assertEqual(kf.x[1], 5.0)
        self.assertEqual(kf.x[2], 0.0)
        self.assertEqual(kf.x[3], 0.0)
        self.assertEqual(kf.age, 1)
        self.assertEqual(kf.time_since_update, 0.0)

    def test_stationary_object_convergence(self) -> None:
        """Verify position converges, velocity dampens to zero, and RMSE improves over raw measurements."""
        rng = random.Random(42)
        true_pos = Vector2(2.0, 5.0)
        sigma = 0.2
        R = ((sigma*sigma, 0.0), (0.0, sigma*sigma))

        kf = SingleObjectKalmanFilter(
            object_id=1,
            class_id="chair",
            initial_position=Vector2(true_pos.x + rng.gauss(0, sigma), true_pos.z + rng.gauss(0, sigma)),
            timestamp=0.0,
        )

        noisy_positions = [Vector2(kf.x[0], kf.x[1])]
        estimated_positions = [Vector2(kf.x[0], kf.x[1])]
        estimated_velocities = [Vector2(kf.x[2], kf.x[3])]
        ground_truth_positions = [true_pos]
        ground_truth_velocities = [Vector2(0.0, 0.0)]

        for step in range(1, 40):
            t = step * 0.1
            meas = Vector2(true_pos.x + rng.gauss(0, sigma), true_pos.z + rng.gauss(0, sigma))
            state = kf.update(meas, t, R)

            noisy_positions.append(meas)
            estimated_positions.append(state.position)
            estimated_velocities.append(state.velocity)
            ground_truth_positions.append(true_pos)
            ground_truth_velocities.append(Vector2(0.0, 0.0))

        # Late-stage position and velocity must be close to true values
        final_state = kf.to_object_state()
        self.assertAlmostEqual(final_state.position.x, 2.0, delta=0.1)
        self.assertAlmostEqual(final_state.position.z, 5.0, delta=0.1)
        self.assertAlmostEqual(final_state.velocity.magnitude, 0.0, delta=0.15)

        # Quantitative RMSE comparison
        noisy_rmse = measurement_position_rmse(noisy_positions[10:], ground_truth_positions[10:])
        kalman_rmse = position_rmse(estimated_positions[10:], ground_truth_positions[10:])
        self.assertLess(kalman_rmse, noisy_rmse)
        improvement = estimation_improvement(noisy_rmse, kalman_rmse)
        self.assertGreater(improvement, 20.0)  # At least 20% improvement

    def test_constant_velocity_tracking(self) -> None:
        """Verify tracking of a moving object recovering true velocity [Vx=1.0, Vz=-0.8]."""
        rng = random.Random(123)
        p0 = Vector2(0.0, 8.0)
        v_true = Vector2(1.0, -0.8)
        sigma = 0.15
        R = ((sigma*sigma, 0.0), (0.0, sigma*sigma))

        kf = SingleObjectKalmanFilter(
            object_id=2,
            class_id="person",
            initial_position=Vector2(p0.x + rng.gauss(0, sigma), p0.z + rng.gauss(0, sigma)),
            timestamp=0.0,
        )

        estimated_vels = []
        for step in range(1, 50):
            t = step * 0.1
            true_pos_t = Vector2(p0.x + v_true.x * t, p0.z + v_true.z * t)
            meas = Vector2(true_pos_t.x + rng.gauss(0, sigma), true_pos_t.z + rng.gauss(0, sigma))
            state = kf.update(meas, t, R)
            if step > 20:
                estimated_vels.append(state.velocity)

        # Steady-state velocity should match ground truth
        avg_vx = sum(v.x for v in estimated_vels) / len(estimated_vels)
        avg_vz = sum(v.z for v in estimated_vels) / len(estimated_vels)
        self.assertAlmostEqual(avg_vx, 1.0, delta=0.15)
        self.assertAlmostEqual(avg_vz, -0.8, delta=0.15)

    def test_missing_measurements_and_recovery(self) -> None:
        """Verify predict-only during gaps (uncertainty grows) and recovery upon new measurement."""
        kf = SingleObjectKalmanFilter(
            object_id=1,
            class_id="person",
            initial_position=Vector2(0.0, 5.0),
            timestamp=0.0,
            initial_velocity=Vector2(0.0, -1.0),
        )
        R = ((0.04, 0.0), (0.0, 0.04))

        # Update at t=0.1 and t=0.2
        kf.update(Vector2(0.0, 4.9), 0.1, R)
        state_02 = kf.update(Vector2(0.0, 4.8), 0.2, R)
        trace_p_02 = state_02.covariance[0][0] + state_02.covariance[1][1]  # type: ignore[index]

        # Missing measurements at t=0.3 and t=0.4 (predict-only)
        state_03 = kf.predict_only(0.3)
        self.assertAlmostEqual(state_03.time_since_update, 0.1, delta=1e-6)
        self.assertAlmostEqual(state_03.position.z, 4.7, delta=0.05)

        state_04 = kf.predict_only(0.4)
        self.assertAlmostEqual(state_04.time_since_update, 0.2, delta=1e-6)
        trace_p_04 = state_04.covariance[0][0] + state_04.covariance[1][1]  # type: ignore[index]
        self.assertGreater(trace_p_04, trace_p_02)  # Covariance grew during prediction

        # Recovery measurement at t=0.5
        state_05 = kf.update(Vector2(0.0, 4.5), 0.5, R)
        self.assertEqual(state_05.time_since_update, 0.0)
        trace_p_05 = state_05.covariance[0][0] + state_05.covariance[1][1]  # type: ignore[index]
        self.assertLess(trace_p_05, trace_p_04)  # Covariance reduced after measurement

    def test_timestamp_irregularity_and_edge_cases(self) -> None:
        """Verify variable dt handling, out-of-order rejection, and zero dt."""
        kf = SingleObjectKalmanFilter(
            object_id=1,
            class_id="person",
            initial_position=Vector2(0.0, 5.0),
            timestamp=0.0,
        )
        R = ((0.04, 0.0), (0.0, 0.04))

        # Irregular timestamps
        kf.update(Vector2(0.0, 4.9), 0.07, R)
        kf.update(Vector2(0.0, 4.8), 0.23, R)
        kf.update(Vector2(0.0, 4.6), 0.45, R)
        self.assertEqual(kf.timestamp, 0.45)

        # Out-of-order timestamp must raise ValueError
        with self.assertRaises(ValueError):
            kf.predict(0.30)

        # Zero dt must not crash or cause infinite covariance
        state_same_t = kf.update(Vector2(0.0, 4.6), 0.45, R)
        self.assertEqual(state_same_t.timestamp, 0.45)
        self.assertTrue(math.isfinite(state_same_t.position.x))

    def test_measurement_noise_trust_sensitivity(self) -> None:
        """Low R trusts measurements heavily; High R trusts prediction heavily."""
        kf_low_R = SingleObjectKalmanFilter(1, "p", Vector2(0, 5), 0.0, initial_velocity=Vector2(0, 0))
        kf_high_R = SingleObjectKalmanFilter(1, "p", Vector2(0, 5), 0.0, initial_velocity=Vector2(0, 0))

        R_low = ((0.001, 0.0), (0.0, 0.001))
        R_high = ((100.0, 0.0), (0.0, 100.0))

        # Observe sudden jump from Z=5 to Z=6 at t=1.0
        jump_meas = Vector2(0.0, 6.0)
        s_low = kf_low_R.update(jump_meas, 1.0, R_low)
        s_high = kf_high_R.update(jump_meas, 1.0, R_high)

        # Low R filter moves much closer to 6.0 than High R filter
        self.assertAlmostEqual(s_low.position.z, 6.0, delta=0.1)
        self.assertLess(s_high.position.z, 5.5)

    def test_covariance_invariants(self) -> None:
        """Verify 4x4 shape, symmetry, positive diagonal elements, and finiteness."""
        kf = SingleObjectKalmanFilter(1, "p", Vector2(0, 5), 0.0)
        R = ((0.05, 0.0), (0.0, 0.05))

        for step in range(1, 10):
            state = kf.update(Vector2(step * 0.1, 5.0 - step * 0.1), step * 0.1, R)
            cov = state.covariance
            self.assertIsNotNone(cov)
            self.assertEqual(len(cov), 4)  # type: ignore[arg-type]
            for i in range(4):
                self.assertEqual(len(cov[i]), 4)  # type: ignore[index]
                self.assertTrue(cov[i][i] > 0.0)  # type: ignore[index]
                for j in range(4):
                    self.assertTrue(math.isfinite(cov[i][j]))  # type: ignore[index]
                    self.assertAlmostEqual(cov[i][j], cov[j][i], delta=1e-7)  # type: ignore[index]


class TestKalmanStateEstimator(unittest.TestCase):
    """Test suite for multi-object manager and track lifecycle management."""

    def test_multi_object_tracking(self) -> None:
        estimator = KalmanStateEstimator()

        # Frame 0: add object 1 and object 2
        s1 = estimator.update_measurement(1, "person", Vector2(0.0, 5.0), 0.0)
        s2 = estimator.update_measurement(2, "chair", Vector2(2.0, 3.0), 0.0)

        self.assertEqual(len(estimator.tracks), 2)
        self.assertEqual(s1.object_id, 1)
        self.assertEqual(s2.object_id, 2)

        # Frame 1: update object 1, predict object 2
        s1_next = estimator.update_measurement(1, "person", Vector2(0.0, 4.9), 0.1)
        s2_pred = estimator.predict_track(2, 0.1)
        self.assertIsNotNone(s2_pred)
        self.assertEqual(s2_pred.time_since_update, 0.1)  # type: ignore[union-attr]

    def test_stale_track_pruning(self) -> None:
        config = KalmanFilterConfig(max_staleness_seconds=0.5)
        estimator = KalmanStateEstimator(config=config)

        estimator.update_measurement(1, "person", Vector2(0.0, 5.0), 0.0)
        estimator.update_measurement(2, "chair", Vector2(2.0, 3.0), 0.0)

        # Update frame at t=0.2: only object 1 observed
        obs_frame1 = [
            # Synthetic observation for object 1
            type("MockObs", (), {
                "object_id": 1,
                "class_id": "person",
                "bbox": type("MockBox", (), {"center": (320.0, 240.0), "bottom_center": (320.0, 485.0)})(),
            })()
        ]
        world_state_1 = estimator.update(obs_frame1, 0.2)  # type: ignore[arg-type]
        self.assertIn(1, world_state_1.objects)
        self.assertIn(2, world_state_1.objects)

        # Update frame at t=0.8 (> 0.5s since object 2 last observed)
        obs_frame2 = [
            type("MockObs", (), {
                "object_id": 1,
                "class_id": "person",
                "bbox": type("MockBox", (), {"center": (320.0, 240.0), "bottom_center": (320.0, 480.0)})(),
            })()
        ]
        world_state_2 = estimator.update(obs_frame2, 0.8)  # type: ignore[arg-type]
        self.assertIn(1, world_state_2.objects)
        self.assertNotIn(2, world_state_2.objects)  # Track 2 pruned


class TestDistanceDependentNoiseAblation(unittest.TestCase):
    """Ablation experiment: Constant R vs. Distance-Dependent R."""

    def test_noise_models_execution(self) -> None:
        const_noise = ConstantMeasurementNoise(sigma_x=0.1, sigma_z=0.2)
        dist_noise = DistanceDependentMeasurementNoise(sigma_x0=0.1, sigma_z0=0.2, alpha=0.05)

        R_const_near = const_noise.measurement_covariance(Vector2(0.0, 2.0))
        R_const_far = const_noise.measurement_covariance(Vector2(0.0, 10.0))
        self.assertEqual(R_const_near, R_const_far)

        R_dist_near = dist_noise.measurement_covariance(Vector2(0.0, 2.0))
        R_dist_far = dist_noise.measurement_covariance(Vector2(0.0, 10.0))
        # At 10m, depth variance must be higher than at 2m
        self.assertGreater(R_dist_far[1][1], R_dist_near[1][1])


if __name__ == "__main__":
    unittest.main()
