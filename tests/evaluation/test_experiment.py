"""Unit tests for ExperimentRun, TrialResult, and MetricResult models."""

import unittest

from evaluation.experiment import ExperimentRun, MetricResult, TrialResult
from evaluation.protocol import ExperimentProtocol
from evaluation.statistics import compute_statistics


class TestExperimentModels(unittest.TestCase):
    """Test suite for experiment data models."""

    def test_metric_result_denominators(self) -> None:
        metric = MetricResult(
            metric_name="hazard_detection_rate",
            numerator=8.0,
            denominator=10.0,
            value=80.0,
            unit="percent",
        )
        self.assertEqual(metric.numerator, 8.0)
        self.assertEqual(metric.denominator, 10.0)
        self.assertEqual(metric.value, 80.0)

    def test_experiment_run_immutability(self) -> None:
        proto = ExperimentProtocol(
            experiment_id="TEST_EXP",
            random_seed=42,
            scenario_ids=("HEAD_ON",),
        )
        trial = TrialResult(
            scenario_id="HEAD_ON",
            trial_index=1,
            seed=42,
            duration_seconds=3.0,
            sample_count=30,
            failure_count=0,
            metrics={"hazard_detection_rate": MetricResult("hazard_detection_rate", 10.0, 10.0, 100.0, "%")},
            samples=(),
        )
        run = ExperimentRun(
            protocol=proto,
            system_name="PRISM",
            trial_results=(trial,),
            aggregated_metrics={"hazard_detection_rate": compute_statistics([100.0])},
        )
        self.assertEqual(run.system_name, "PRISM")
        self.assertEqual(len(run.trial_results), 1)


if __name__ == "__main__":
    unittest.main()
