"""Unit and scientific validation tests for the PRISM Simulator."""

import unittest

from app.core.types import ObjectObservation, Vector2
from simulation.ground_truth import GroundTruthSequence
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_head_on_scenario
from simulation.simulator import Simulator


class TestSimulator(unittest.TestCase):
    """Test suite for simulation execution, determinism, and ground-truth isolation."""

    def test_deterministic_noiseless_simulation(self) -> None:
        """Verify simulator steps, frame counts, and deterministic timestamps."""
        scenario = create_head_on_scenario(duration=2.0, timestep=0.1)
        sim = Simulator(scenario=scenario, noise_model=None)

        gt_seq, obs_history = sim.run()

        expected_frames = 21  # t=0.0, 0.1, ..., 2.0
        self.assertEqual(gt_seq.num_frames, expected_frames)
        self.assertEqual(len(obs_history), expected_frames)
        self.assertEqual(gt_seq.start_time, 0.0)
        self.assertEqual(gt_seq.end_time, 2.0)

        # Check frame 0
        f0 = gt_seq.get_frame(0)
        self.assertEqual(f0.timestamp, 0.0)
        self.assertEqual(f0.ego.position, Vector2(0.0, 0.0))
        self.assertEqual(f0.objects[0].position, Vector2(0.0, 5.0))

        # Check last frame
        f_last = gt_seq.get_frame(-1)
        self.assertEqual(f_last.timestamp, 2.0)
        self.assertEqual(f_last.ego.position, Vector2(0.0, 2.0))
        self.assertEqual(f_last.objects[0].position, Vector2(0.0, 3.0))

    def test_observation_structure(self) -> None:
        """Verify that observations produced conform to ObjectObservation contract."""
        scenario = create_head_on_scenario(duration=1.0, timestep=0.5)
        sim = Simulator(scenario=scenario, noise_model=None)

        _, obs_history = sim.run()

        for frame_obs in obs_history:
            self.assertEqual(len(frame_obs), 1)
            obs = frame_obs[0]
            self.assertIsInstance(obs, ObjectObservation)
            self.assertEqual(obs.object_id, 1)
            self.assertEqual(obs.class_id, "person")
            self.assertTrue(0.0 <= obs.confidence <= 1.0)
            self.assertTrue(obs.bbox.width > 0)
            self.assertTrue(obs.bbox.height > 0)

    def test_scientific_isolation_ground_truth_unaffected_by_noise(self) -> None:
        """SCIENTIFIC TEST: Verify noise affects observations but NEVER corrupts ground truth.

        Requirements:
        1. ground_truth(with_noise) == ground_truth(without_noise)
        2. observations(with_noise) != observations(without_noise)
        3. observations(seed=42) == observations(seed=42)
        4. observations(seed=42) != observations(seed=43)
        """
        scenario = create_head_on_scenario(duration=2.0, timestep=0.1)

        # 1. Noiseless run
        sim_clean = Simulator(scenario=scenario, noise_model=None)
        gt_clean, obs_clean = sim_clean.run()

        # 2. Noisy run with seed 42
        noise_model_42a = GaussianNoiseModel(sigma_x=0.15, sigma_z=0.25, seed=42)
        sim_noisy_42a = Simulator(scenario=scenario, noise_model=noise_model_42a)
        gt_noisy_42a, obs_noisy_42a = sim_noisy_42a.run()

        # 3. Repeat noisy run with seed 42
        noise_model_42b = GaussianNoiseModel(sigma_x=0.15, sigma_z=0.25, seed=42)
        sim_noisy_42b = Simulator(scenario=scenario, noise_model=noise_model_42b)
        gt_noisy_42b, obs_noisy_42b = sim_noisy_42b.run()

        # 4. Noisy run with different seed 43
        noise_model_43 = GaussianNoiseModel(sigma_x=0.15, sigma_z=0.25, seed=43)
        sim_noisy_43 = Simulator(scenario=scenario, noise_model=noise_model_43)
        gt_noisy_43, obs_noisy_43 = sim_noisy_43.run()

        # SCIENTIFIC ASSERTION 1: Ground truth is identical across all runs
        self.assertEqual(gt_clean, gt_noisy_42a)
        self.assertEqual(gt_clean, gt_noisy_42b)
        self.assertEqual(gt_clean, gt_noisy_43)

        # SCIENTIFIC ASSERTION 2: Observations with noise differ from clean observations
        clean_bboxes = [frame_obs[0].bbox for frame_obs in obs_clean]
        noisy_bboxes_42a = [frame_obs[0].bbox for frame_obs in obs_noisy_42a]
        self.assertNotEqual(clean_bboxes, noisy_bboxes_42a)

        # SCIENTIFIC ASSERTION 3: Identical seed produces identical observation sequences
        noisy_bboxes_42b = [frame_obs[0].bbox for frame_obs in obs_noisy_42b]
        self.assertEqual(noisy_bboxes_42a, noisy_bboxes_42b)

        # SCIENTIFIC ASSERTION 4: Different seeds produce different observation sequences
        noisy_bboxes_43 = [frame_obs[0].bbox for frame_obs in obs_noisy_43]
        self.assertNotEqual(noisy_bboxes_42a, noisy_bboxes_43)


if __name__ == "__main__":
    unittest.main()
