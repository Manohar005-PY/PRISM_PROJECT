"""Unit tests for GroundTruthProvider and trajectory state queries."""

import unittest

from evaluation.ground_truth import SyntheticGroundTruthProvider
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_head_on_scenario
from simulation.simulator import Simulator


class TestGroundTruth(unittest.TestCase):
    """Test suite for SyntheticGroundTruthProvider."""

    def test_ground_truth_lookup_and_hazard_detection(self) -> None:
        scenario = create_head_on_scenario(duration=3.0, timestep=0.1)
        sim = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0))
        gt_seq, _ = sim.run()

        provider = SyntheticGroundTruthProvider(gt_seq)

        # Early in scenario (t=0.1s, distance ~4.8m) -> no hazard yet
        states_early = provider.get_state(0.1)
        self.assertIn(1, states_early)

        # Later in scenario (t=2.0s, converging close) -> hazard is True
        self.assertTrue(provider.is_hazard(2.0))


if __name__ == "__main__":
    unittest.main()
