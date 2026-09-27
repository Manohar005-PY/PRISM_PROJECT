"""Unit tests for ControlledAblationStudy."""

import unittest

from evaluation.ablation import ControlledAblationStudy
from evaluation.scenarios import create_head_on_scenario


class TestAblation(unittest.TestCase):
    """Test suite for controlled component ablation."""

    def test_run_ablations(self) -> None:
        study = ControlledAblationStudy(random_seed=42)
        sc = create_head_on_scenario(duration=2.0, timestep=0.1)

        results = study.run_all_ablations(scenarios={"HEAD_ON": sc})
        self.assertGreater(len(results), 5)
        ab_ids = [r.ablation_id for r in results]
        self.assertIn("FULL_PRISM", ab_ids)
        self.assertIn("A_NO_SMOOTHING", ab_ids)
        self.assertIn("G_NO_PREDICTION", ab_ids)


if __name__ == "__main__":
    unittest.main()
