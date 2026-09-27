"""Unit and integration test suite for PRISM Part 18 Baseline Comparison & Predictive Advantage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simulation.baselines.models import ReactiveBaselineConfig, ReactiveBaselineDecision
from simulation.baselines.reactive_distance import ReactiveDistanceBaseline
from simulation.comparison.aggregation import ComparisonAggregator
from simulation.comparison.comparator import PRISMBaselineComparator
from simulation.comparison.failure_analysis import ComparativeFailureAnalyzer
from simulation.comparison.metrics import ComparativeMetricsEvaluator
from simulation.comparison.models import (
    ComparativeAvoidanceStats,
    ComparativeWarningStats,
    ComparisonConfig,
    PairwiseRunComparison,
)


class TestBaselineComparison(unittest.TestCase):
    """Test suite for Part 18 baseline execution, comparative metrics, and orchestrator."""

    def test_baseline_deterministic_decision_mapping(self) -> None:
        """Verify baseline deterministic action and heading mapping across distance zones."""
        config = ReactiveBaselineConfig(
            warning_distance_m=3.0,
            avoidance_distance_m=1.5,
            stop_distance_m=0.8,
            hysteresis_frames=0,  # Disable hysteresis for pure threshold testing
        )
        baseline = ReactiveDistanceBaseline(config)

        # 1. Empty observation -> CONTINUE
        d_empty = baseline.step(0.0, 0, [])
        self.assertEqual(d_empty.action, "CONTINUE")
        self.assertEqual(d_empty.warning_level, "SAFE")
        self.assertFalse(d_empty.is_warning)

        # 2. Far obstacle (d = 4.0m) -> CONTINUE
        d_far = baseline.step(0.05, 1, [{"x": 0.0, "z": 4.0, "valid": True}])
        self.assertEqual(d_far.action, "CONTINUE")
        self.assertFalse(d_far.is_warning)

        # 3. Caution zone (d = 2.0m) -> SLOW_DOWN
        d_caution = baseline.step(0.10, 2, [{"x": 0.0, "z": 2.0, "valid": True}])
        self.assertEqual(d_caution.action, "SLOW_DOWN")
        self.assertEqual(d_caution.warning_level, "HIGH")
        self.assertTrue(d_caution.is_warning)

        # 4. Emergency lateral avoidance ahead/right (x = 0.2m, z = 1.0m, d = 1.02m) -> TURN_LEFT
        d_avoid_l = baseline.step(0.15, 3, [{"x": 0.2, "z": 1.0, "valid": True}])
        self.assertEqual(d_avoid_l.action, "TURN_LEFT")
        self.assertAlmostEqual(d_avoid_l.recommended_heading_rad, 0.35, places=2)
        self.assertTrue(d_avoid_l.is_warning)

        # 5. Emergency lateral avoidance left (x = -0.5m, z = 1.0m, d = 1.12m) -> TURN_RIGHT
        d_avoid_r = baseline.step(0.20, 4, [{"x": -0.5, "z": 1.0, "valid": True}])
        self.assertEqual(d_avoid_r.action, "TURN_RIGHT")
        self.assertAlmostEqual(d_avoid_r.recommended_heading_rad, -0.35, places=2)

        # 6. Critical contact zone (d = 0.5m) -> STOP
        d_stop = baseline.step(0.25, 5, [{"x": 0.0, "z": 0.5, "valid": True}])
        self.assertEqual(d_stop.action, "STOP")
        self.assertEqual(d_stop.warning_level, "CRITICAL")
        self.assertTrue(d_stop.is_warning)

    def test_baseline_hysteresis_and_clean_reset(self) -> None:
        """Verify action escalation and clean state reset between runs."""
        config = ReactiveBaselineConfig(hysteresis_frames=3)
        baseline = ReactiveDistanceBaseline(config)

        # Escalate to STOP
        baseline.step(0.0, 0, [{"x": 0.0, "z": 0.5, "valid": True}])
        self.assertEqual(baseline._last_action, "STOP")

        # Obstacle disappears for 1 frame: hysteresis holds STOP
        d_hold = baseline.step(0.05, 1, [])
        self.assertEqual(d_hold.action, "STOP")

        # Reset between runs
        baseline.reset()
        self.assertEqual(baseline._last_action, "CONTINUE")
        d_after_reset = baseline.step(0.0, 0, [])
        self.assertEqual(d_after_reset.action, "CONTINUE")

    def test_comparative_metrics_lead_time_and_earlier_detection(self) -> None:
        """Verify paired warning lead-time calculations and delta lead-time metrics."""
        evaluator = ComparativeMetricsEvaluator()

        # 100 timesteps at dt=0.05s (duration = 5.0s), collision at t=4.0s
        # PRISM warns at t=1.0s -> lead time = 3.0s
        # Baseline warns at t=2.5s -> lead time = 1.5s
        # Delta lead time = +1.5s (PRISM earlier)
        gt_records = []
        prism_records = []
        baseline_decisions = []

        for i in range(100):
            t = i * 0.05
            has_col = (t >= 4.0)
            gt_records.append({
                "timestamp": t,
                "frame_index": i,
                "ego": {"x": 0.0, "z": t, "radius": 0.35},
                "objects": [{"x": 0.0, "z": 4.0, "radius": 0.35}],
                "collision_state": {"has_collided": has_col},
            })

            # PRISM alert starting at t=1.0s
            pr_warn = (t >= 1.0)
            prism_records.append({
                "timestamp": t,
                "frame_index": i,
                "decision_action": "SLOW_DOWN" if pr_warn else "CONTINUE",
                "risk_level": "HIGH" if pr_warn else "SAFE",
            })

            # Baseline alert starting at t=2.5s
            bl_warn = (t >= 2.5)
            baseline_decisions.append(
                ReactiveBaselineDecision(
                    timestamp=t,
                    frame_index=i,
                    action="SLOW_DOWN" if bl_warn else "CONTINUE",
                    warning_level="HIGH" if bl_warn else "SAFE",
                    recommended_heading_rad=0.0,
                    closest_obstacle_id=1 if bl_warn else None,
                    min_observed_distance_m=2.0 if bl_warn else 5.0,
                    is_warning=bl_warn,
                )
            )

        warn_stats, avoid_stats = evaluator.evaluate_run(
            gt_records=gt_records,
            prism_records=prism_records,
            baseline_decisions=baseline_decisions,
        )

        self.assertTrue(warn_stats.has_collision)
        self.assertTrue(warn_stats.prism_warned)
        self.assertTrue(warn_stats.baseline_warned)
        self.assertAlmostEqual(warn_stats.prism_lead_time_s, 3.0, places=3)
        self.assertAlmostEqual(warn_stats.baseline_lead_time_s, 1.5, places=3)
        self.assertAlmostEqual(warn_stats.delta_lead_time_s, 1.5, places=3)
        self.assertEqual(warn_stats.earlier_warning_system, "PRISM")

    def test_missed_baseline_warning_no_zero_substitution(self) -> None:
        """Verify that when baseline misses warning, lead time is None rather than 0.0s."""
        evaluator = ComparativeMetricsEvaluator()

        gt_records = [
            {"timestamp": 0.0, "frame_index": 0, "ego": {}, "objects": [], "collision_state": {"has_collided": False}},
            {"timestamp": 1.0, "frame_index": 1, "ego": {}, "objects": [], "collision_state": {"has_collided": True}},
        ]
        prism_records = [
            {"timestamp": 0.0, "frame_index": 0, "decision_action": "SLOW_DOWN", "risk_level": "HIGH"},
            {"timestamp": 1.0, "frame_index": 1, "decision_action": "STOP", "risk_level": "CRITICAL"},
        ]
        # Baseline never warned
        baseline_decisions = [
            ReactiveBaselineDecision(0.0, 0, "CONTINUE", "SAFE", 0.0, None, None, False),
            ReactiveBaselineDecision(1.0, 1, "CONTINUE", "SAFE", 0.0, None, None, False),
        ]

        warn_stats, _ = evaluator.evaluate_run(gt_records, prism_records, baseline_decisions)
        self.assertTrue(warn_stats.prism_warned)
        self.assertFalse(warn_stats.baseline_warned)
        self.assertEqual(warn_stats.prism_lead_time_s, 1.0)
        self.assertIsNone(warn_stats.baseline_lead_time_s)
        self.assertIsNone(warn_stats.delta_lead_time_s)
        self.assertEqual(warn_stats.earlier_warning_system, "PRISM")

    def test_comparative_failure_analyzer(self) -> None:
        """Verify extraction of comparative discrepancies."""
        analyzer = ComparativeFailureAnalyzer()

        warn_stats = ComparativeWarningStats(
            has_collision=True,
            prism_warned=True,
            baseline_warned=True,
            prism_first_warning_time_s=1.0,
            baseline_first_warning_time_s=3.5,
            collision_time_s=4.0,
            prism_lead_time_s=3.0,
            baseline_lead_time_s=0.5,  # Late warning (< 1.0s)
            delta_lead_time_s=2.5,
            earlier_warning_system="PRISM",
            prism_maintained_ratio=1.0,
            baseline_maintained_ratio=1.0,
            prism_false_warning=False,
            baseline_false_warning=False,
        )

        failures = analyzer.extract_failures(
            scenario_id="sc_01",
            scenario_family="S02",
            run_id="run_01",
            gt_records=[{}],
            prism_records=[{}],
            baseline_decisions=[],
            warning_stats=warn_stats,
        )

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].failure_type, "BASELINE_LATE_WARNING")
        self.assertEqual(failures[0].baseline_value, 0.5)

    def test_comparison_aggregator(self) -> None:
        """Verify predictive advantage aggregation across runs."""
        aggregator = ComparisonAggregator()

        def make_dummy_comparison(run_id: str, pr_lead: float | None, bl_lead: float | None) -> PairwiseRunComparison:
            delta = (pr_lead - bl_lead) if (pr_lead is not None and bl_lead is not None) else None
            earlier = "PRISM" if (delta is not None and delta > 0) else "BASELINE"
            w = ComparativeWarningStats(
                has_collision=True,
                prism_warned=(pr_lead is not None),
                baseline_warned=(bl_lead is not None),
                prism_first_warning_time_s=0.0,
                baseline_first_warning_time_s=0.0,
                collision_time_s=5.0,
                prism_lead_time_s=pr_lead,
                baseline_lead_time_s=bl_lead,
                delta_lead_time_s=delta,
                earlier_warning_system=earlier,
                prism_maintained_ratio=1.0,
                baseline_maintained_ratio=1.0,
                prism_false_warning=False,
                baseline_false_warning=False,
            )
            a = ComparativeAvoidanceStats(
                prism_decision_changes=2,
                baseline_decision_changes=4,
                prism_steering_reversals=0,
                baseline_steering_reversals=1,
                prism_min_clearance_m=0.6,
                baseline_min_clearance_m=0.4,
                prism_safe_clearance_ratio=1.0,
                baseline_safe_clearance_ratio=0.8,
            )
            return PairwiseRunComparison(
                scenario_id="sc",
                scenario_family="S01",
                run_id=run_id,
                num_frames=100,
                duration=5.0,
                noise_condition="NO_NOISE",
                warning_comparison=w,
                avoidance_comparison=a,
                failures=[],
            )

        runs = [
            make_dummy_comparison("run_1", pr_lead=3.0, bl_lead=1.0),  # Delta = +2.0s
            make_dummy_comparison("run_2", pr_lead=2.5, bl_lead=1.5),  # Delta = +1.0s
        ]

        report = aggregator.aggregate_comparisons("test_dataset", runs)
        adv = report.predictive_advantage
        self.assertEqual(adv.total_collision_runs, 2)
        self.assertEqual(adv.paired_lead_time_samples, 2)
        self.assertAlmostEqual(adv.mean_delta_lead_time_s, 1.5, places=3)
        self.assertEqual(adv.earlier_warning_count, 2)
        self.assertAlmostEqual(adv.earlier_warning_rate, 1.0, places=3)
        self.assertAlmostEqual(adv.prism_mean_lead_time_s, 2.75, places=3)
        self.assertAlmostEqual(adv.baseline_mean_lead_time_s, 1.25, places=3)
        # Improvement = (2.75 - 1.25) / 1.25 = 120%
        self.assertAlmostEqual(adv.relative_lead_time_improvement_pct, 120.0, places=1)

    def test_end_to_end_comparator(self) -> None:
        """Test full comparative execution workflow using a temporary dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            run_dir = dataset_dir / "S01_stationary" / "S01_stationary_run_0001"
            run_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "dataset_name": "temp_test_dataset",
                "scenarios": [
                    {
                        "scenario_id": "S01_stationary_run_0001",
                        "scenario_family": "S01_stationary",
                        "run_dir": "S01_stationary/S01_stationary_run_0001",
                    }
                ],
            }
            with open(dataset_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            meta = {
                "scenario_id": "S01_stationary_run_0001",
                "scenario_family": "S01_stationary",
                "run_id": "S01_stationary_run_0001",
                "duration": 0.1,
                "parameters": {"noise_level": "NO_NOISE"},
            }
            with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

            gt_lines = [
                {"timestamp": 0.0, "frame_index": 0, "ego": {"x": 0.0, "z": 0.0, "radius": 0.35}, "objects": [{"object_id": 1, "x": 0.0, "z": 2.0, "radius": 0.35}], "collision_state": {"has_collided": False, "time_to_collision": 2.0}},
                {"timestamp": 0.05, "frame_index": 1, "ego": {"x": 0.0, "z": 0.1, "radius": 0.35}, "objects": [{"object_id": 1, "x": 0.0, "z": 2.0, "radius": 0.35}], "collision_state": {"has_collided": True, "time_to_collision": 0.0}},
            ]
            with open(run_dir / "ground_truth.jsonl", "w", encoding="utf-8") as f:
                for line in gt_lines:
                    f.write(json.dumps(line) + "\n")

            obs_lines = [
                {"timestamp": 0.0, "frame_index": 0, "observations": [{"object_id": 1, "x": 0.0, "z": 2.0, "valid": True}]},
                {"timestamp": 0.05, "frame_index": 1, "observations": [{"object_id": 1, "x": 0.0, "z": 1.9, "valid": True}]},
            ]
            with open(run_dir / "observations.jsonl", "w", encoding="utf-8") as f:
                for line in obs_lines:
                    f.write(json.dumps(line) + "\n")

            prism_lines = [
                {"timestamp": 0.0, "frame_index": 0, "decision_action": "SLOW_DOWN", "risk_level": "HIGH"},
                {"timestamp": 0.05, "frame_index": 1, "decision_action": "STOP", "risk_level": "CRITICAL"},
            ]
            with open(run_dir / "prism_output.jsonl", "w", encoding="utf-8") as f:
                for line in prism_lines:
                    f.write(json.dumps(line) + "\n")

            comp_cfg = ComparisonConfig(dataset_dir=str(dataset_dir), output_dir=str(output_dir))
            comparator = PRISMBaselineComparator(comp_cfg)
            report, runs = comparator.compare_dataset(output_dir=output_dir, export_csv=True)

            self.assertEqual(len(runs), 1)
            self.assertTrue((output_dir / "config" / "baseline_config.json").exists())
            self.assertTrue((output_dir / "runs" / "S01_stationary_run_0001.json").exists())
            self.assertTrue((output_dir / "aggregate" / "overall_comparison.json").exists())
            self.assertTrue((output_dir / "aggregate" / "predictive_advantage.json").exists())
            self.assertTrue((output_dir / "failures" / "comparative_failures.jsonl").exists())
            self.assertTrue((output_dir / "csv" / "run_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
