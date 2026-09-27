"""Unit tests for EvaluationRunner scenario execution."""

import unittest

from evaluation.baselines import DistanceBaseline
from evaluation.runner import EvaluationRunner
from simulation.scenarios import create_head_on_scenario


class TestEvaluationRunner(unittest.TestCase):
    """Test suite for EvaluationRunner."""

    def test_run_scenario_execution(self) -> None:
        runner = EvaluationRunner(random_seed=42)
        scenario = create_head_on_scenario(duration=2.0, timestep=0.1)
        evaluator = DistanceBaseline()

        samples = runner.run_scenario(scenario, evaluator)
        self.assertEqual(len(samples), 21)
        self.assertEqual(samples[0].frame_id, 1)


if __name__ == "__main__":
    unittest.main()
