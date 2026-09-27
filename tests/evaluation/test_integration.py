"""Full comparative integration test evaluating Distance Baseline, TTC Baseline, and PRISM."""

import unittest

from evaluation.baselines import DistanceBaseline, TTCBaseline
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_unnecessary_stop_rate,
)
from evaluation.runner import EvaluationRunner, create_prism_evaluator_system
from evaluation.scenarios import create_departing_scenario, create_head_on_scenario


class TestEvaluationIntegration(unittest.TestCase):
    """Integration test verifying comparative evaluation across scenarios."""

    def test_comparative_benchmark_head_on_and_departing(self) -> None:
        runner = EvaluationRunner(random_seed=42)

        sc_head_on = create_head_on_scenario(duration=3.0, timestep=0.1)
        sc_depart = create_departing_scenario(duration=3.0, timestep=0.1)

        base_dist = DistanceBaseline(stop_threshold=2.2, caution_threshold=3.5)
        base_ttc = TTCBaseline()
        prism = create_prism_evaluator_system()

        # Run HEAD_ON
        samples_dist_ho = runner.run_scenario(sc_head_on, base_dist)
        samples_ttc_ho = runner.run_scenario(sc_head_on, base_ttc)
        samples_prism_ho = runner.run_scenario(sc_head_on, prism)

        det_dist = compute_hazard_detection_rate(samples_dist_ho)
        det_ttc = compute_hazard_detection_rate(samples_ttc_ho)
        det_prism = compute_hazard_detection_rate(samples_prism_ho)

        self.assertGreater(det_dist, 0.0)
        self.assertGreater(det_ttc, 0.0)
        self.assertGreater(det_prism, 0.0)

        # Run DEPARTING
        samples_dist_dep = runner.run_scenario(sc_depart, base_dist)
        samples_prism_dep = runner.run_scenario(sc_depart, prism)

        stops_dist = compute_unnecessary_stop_rate(samples_dist_dep)
        stops_prism = compute_unnecessary_stop_rate(samples_prism_dep)

        # Distance baseline with stop_threshold=2.2 triggers false stop on departing obstacle (starts at 2.0m)
        self.assertGreaterEqual(stops_dist, stops_prism)


if __name__ == "__main__":
    unittest.main()
