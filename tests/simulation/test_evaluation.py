"""Unit and integration test suite for PRISM Part 17 Quantitative Evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from simulation.evaluation.aggregation import MetricsAggregator
from simulation.evaluation.alignment import AlignedFrame, SpatialTemporalAligner
from simulation.evaluation.avoidance_metrics import AvoidanceEvaluator
from simulation.evaluation.dcpa_metrics import DCPAEvaluator
from simulation.evaluation.evaluator import PRISMEvaluator
from simulation.evaluation.failure_analysis import FailureAnalyzer
from simulation.evaluation.loader import DatasetLoader
from simulation.evaluation.models import (
    AggregateEvaluationReport,
    EvaluationConfig,
    FailureCase,
    MetricStats,
    RunEvaluationResult,
)
from simulation.evaluation.prediction_metrics import PredictionEvaluator
from simulation.evaluation.risk_metrics import RiskEvaluator
from simulation.evaluation.robustness import RobustnessEvaluator
from simulation.evaluation.state_metrics import StateEstimationEvaluator, compute_metric_stats
from simulation.evaluation.tracking_metrics import TrackingEvaluator
from simulation.evaluation.ttc_metrics import TTCEvaluator
from simulation.evaluation.warning_metrics import WarningEvaluator


class TestEvaluationFramework(unittest.TestCase):
    """Test suite covering statistical computation, matching, metrics, and orchestrator."""

    def test_metric_stats_computation(self) -> None:
        """Verify statistical metrics calculation."""
        empty_stats = compute_metric_stats([])
        self.assertEqual(empty_stats.n, 0)
        self.assertIsNone(empty_stats.mean)

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_metric_stats(vals)
        self.assertEqual(stats.n, 5)
        self.assertEqual(stats.mean, 3.0)
        self.assertEqual(stats.median, 3.0)
        self.assertEqual(stats.min, 1.0)
        self.assertEqual(stats.max, 5.0)
        self.assertEqual(stats.mae, 3.0)
        self.assertAlmostEqual(stats.rmse, math.sqrt(55 / 5), places=3)
        self.assertEqual(stats.p25, 2.0)
        self.assertEqual(stats.p75, 4.0)

    def test_spatial_temporal_alignment(self) -> None:
        """Verify distance-gated Hungarian matching of GT objects to PRISM tracks."""
        aligner = SpatialTemporalAligner(gating_threshold_m=1.5)

        gt_records = [
            {
                "timestamp": 0.0,
                "frame_index": 0,
                "ego": {"x": 0.0, "z": 0.0, "radius": 0.35},
                "objects": [
                    {"object_id": 1, "x": 1.0, "z": 5.0, "vx": 0.0, "vz": -1.0, "radius": 0.35},
                    {"object_id": 2, "x": -4.0, "z": 8.0, "vx": 0.5, "vz": 0.0, "radius": 0.35},
                ],
                "collision_state": {"has_collided": False, "time_to_collision": 5.0, "closest_point_of_approach": 0.0},
            }
        ]
        obs_records = [{"timestamp": 0.0, "frame_index": 0, "detections": []}]
        prism_records = [
            {
                "timestamp": 0.0,
                "frame_index": 0,
                "active_tracks": [
                    {"track_id": 101, "x": 1.1, "z": 5.05, "vx": 0.0, "vz": -0.95, "lifecycle_state": "CONFIRMED"},
                    {"track_id": 102, "x": 10.0, "z": 10.0, "vx": 0.0, "vz": 0.0, "lifecycle_state": "TENTATIVE"},  # Distant (ungated)
                ],
                "predicted_ttc": 5.1,
                "predicted_cpa": 0.05,
                "risk_level": "HIGH",
                "risk_score": 75.0,
                "decision_action": "SLOW_DOWN",
            }
        ]

        aligned = aligner.align_run(gt_records, obs_records, prism_records)
        self.assertEqual(len(aligned), 1)
        frame = aligned[0]

        # Object 1 matches Track 101
        self.assertEqual(len(frame.matched_pairs), 1)
        self.assertEqual(frame.matched_pairs[0][0]["object_id"], 1)
        self.assertEqual(frame.matched_pairs[0][1]["track_id"], 101)

        # Object 2 is unmatched
        self.assertEqual(len(frame.unmatched_gt), 1)
        self.assertEqual(frame.unmatched_gt[0]["object_id"], 2)

        # Track 102 is unmatched (beyond 1.5m gate)
        self.assertEqual(len(frame.unmatched_tracks), 1)
        self.assertEqual(frame.unmatched_tracks[0]["track_id"], 102)

    def test_state_estimation_evaluator(self) -> None:
        """Verify position and velocity estimation error computation."""
        evaluator = StateEstimationEvaluator()

        frame = AlignedFrame(
            timestamp=0.0,
            frame_index=0,
            ego_gt={"x": 0.0, "z": 0.0},
            matched_pairs=(
                (
                    {"object_id": 1, "x": 1.0, "z": 2.0, "vx": 0.0, "vz": 1.0},
                    {"track_id": 10, "x": 1.3, "z": 2.4, "vx": 0.1, "vz": 1.1},
                ),
            ),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )

        res = evaluator.evaluate_run([frame])
        # dx = 0.3, dz = 0.4 -> dist = 0.5
        self.assertAlmostEqual(res["position_2d"]["mae"], 0.5, places=3)
        self.assertAlmostEqual(res["position_x"]["mae"], 0.3, places=3)
        self.assertAlmostEqual(res["position_z"]["mae"], 0.4, places=3)
        # dvx = 0.1, dvz = 0.1 -> vel_dist = sqrt(0.02) = 0.1414
        self.assertAlmostEqual(res["velocity_2d"]["mae"], math.sqrt(0.02), places=3)

    def test_tracking_evaluator(self) -> None:
        """Verify tracking confirmation, latency, ID switches, fragmentation, and MOTA/IDF1."""
        evaluator = TrackingEvaluator()

        # Frame 0: GT 1 matches Trk 10 (Tentative)
        f0 = AlignedFrame(
            timestamp=0.0,
            frame_index=0,
            ego_gt={},
            matched_pairs=(({"object_id": 1}, {"track_id": 10, "lifecycle_state": "TENTATIVE"}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )
        # Frame 1: GT 1 matches Trk 10 (Confirmed) -> latency = 0.05s
        f1 = AlignedFrame(
            timestamp=0.05,
            frame_index=1,
            ego_gt={},
            matched_pairs=(({"object_id": 1}, {"track_id": 10, "lifecycle_state": "CONFIRMED"}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )
        # Frame 2: GT 1 matches Trk 20 (ID switch)
        f2 = AlignedFrame(
            timestamp=0.10,
            frame_index=2,
            ego_gt={},
            matched_pairs=(({"object_id": 1}, {"track_id": 20, "lifecycle_state": "CONFIRMED"}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )

        res = evaluator.evaluate_run([f0, f1, f2])
        self.assertEqual(res["total_gt_objects"], 1)
        self.assertEqual(res["confirmed_gt_objects"], 1)
        self.assertEqual(res["confirmation_rate"], 1.0)
        self.assertAlmostEqual(res["confirmation_latency_s"]["mean"], 0.05, places=3)
        self.assertEqual(res["id_switch_count"], 1)
        self.assertGreaterEqual(res["mota"], 0.0)
        self.assertGreaterEqual(res["idf1"], 0.0)

    def test_prediction_evaluator(self) -> None:
        """Verify multi-horizon trajectory displacement errors."""
        evaluator = PredictionEvaluator(horizons=(1.0,))

        # Constant velocity along z: v = -1.0 m/s, dt = 0.5s
        # Frame 0 at t=0.0, pos=(0, 10), vel=(0, -1) -> Pred at t+1.0s is (0, 8)
        # Frame 2 at t=1.0, true pos=(0, 8.2) -> error = 0.2m
        f0 = AlignedFrame(
            timestamp=0.0,
            frame_index=0,
            ego_gt={},
            matched_pairs=(({"object_id": 1, "x": 0.0, "z": 10.0}, {"track_id": 1, "x": 0.0, "z": 10.0, "vx": 0.0, "vz": -1.0}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )
        f1 = AlignedFrame(
            timestamp=0.5,
            frame_index=1,
            ego_gt={},
            matched_pairs=(({"object_id": 1, "x": 0.0, "z": 9.1}, {"track_id": 1, "x": 0.0, "z": 9.1, "vx": 0.0, "vz": -1.0}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )
        f2 = AlignedFrame(
            timestamp=1.0,
            frame_index=2,
            ego_gt={},
            matched_pairs=(({"object_id": 1, "x": 0.0, "z": 8.2}, {"track_id": 1, "x": 0.0, "z": 8.2, "vx": 0.0, "vz": -1.0}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={},
            prism_output={},
        )

        res = evaluator.evaluate_run([f0, f1, f2])
        self.assertIn("horizon_1.0s", res)
        self.assertAlmostEqual(res["horizon_1.0s"]["mae"], 0.8, places=3)

    def test_ttc_and_dcpa_evaluator(self) -> None:
        """Verify TTC and dCPA error stats and finite/infinite disagreement classifications."""
        ttc_eval = TTCEvaluator()
        dcpa_eval = DCPAEvaluator()

        f0 = AlignedFrame(
            timestamp=0.0,
            frame_index=0,
            ego_gt={},
            matched_pairs=(),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"time_to_collision": 4.0, "closest_point_of_approach": 0.20},
            prism_output={"predicted_ttc": 4.2, "predicted_cpa": 0.25},
        )
        f1 = AlignedFrame(
            timestamp=0.1,
            frame_index=1,
            ego_gt={},
            matched_pairs=(),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"time_to_collision": None, "closest_point_of_approach": 2.50},
            prism_output={"predicted_ttc": 3.0, "predicted_cpa": 0.40},  # False collision prediction
        )
        f2 = AlignedFrame(
            timestamp=0.2,
            frame_index=2,
            ego_gt={},
            matched_pairs=(),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"time_to_collision": 2.0, "closest_point_of_approach": 0.10},
            prism_output={"predicted_ttc": None, "predicted_cpa": 1.50},  # Missed collision prediction
        )

        ttc_res = ttc_eval.evaluate_run([f0, f1, f2])
        self.assertEqual(ttc_res["finite_vs_finite_count"], 1)
        self.assertEqual(ttc_res["false_predicted_collision_count"], 1)
        self.assertEqual(ttc_res["missed_predicted_collision_count"], 1)
        self.assertAlmostEqual(ttc_res["error_stats"]["mae"], 0.2, places=3)

        dcpa_res = dcpa_eval.evaluate_run([f0, f1, f2])
        self.assertEqual(dcpa_res["error_stats"]["n"], 3)
        self.assertAlmostEqual(dcpa_res["error_stats"]["mae"], (0.05 + 2.10 + 1.40) / 3, places=3)

    def test_risk_evaluator_multi_horizon_and_calibration(self) -> None:
        """Verify multi-horizon event detection and empirical risk calibration binning."""
        risk_eval = RiskEvaluator(event_horizons=(1.0,), risk_bins=((0.0, 50.0), (50.0, 100.0)))

        # Frame 0: score=85, risk=HIGH -> Collision occurs at Frame 2 (t=1.0s) -> TP for H=1.0s
        f0 = AlignedFrame(
            timestamp=0.0,
            frame_index=0,
            ego_gt={},
            matched_pairs=(),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"has_collided": False},
            prism_output={"risk_level": "HIGH", "risk_score": 85.0},
        )
        f1 = AlignedFrame(
            timestamp=0.5,
            frame_index=1,
            ego_gt={},
            matched_pairs=(),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"has_collided": False},
            prism_output={"risk_level": "HIGH", "risk_score": 90.0},
        )
        f2 = AlignedFrame(
            timestamp=1.0,
            frame_index=2,
            ego_gt={},
            matched_pairs=(),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"has_collided": True},
            prism_output={"risk_level": "CRITICAL", "risk_score": 98.0},
        )

        res = risk_eval.evaluate_run([f0, f1, f2])
        h1 = res["multi_horizon_detection"]["horizon_1.0s"]
        self.assertGreaterEqual(h1["tp"], 1)

        cal_bins = res["risk_calibration_bins"]
        self.assertEqual(len(cal_bins), 2)
        # All frames had score > 50, so [50-100] bin has 3 samples and empirical collision freq > 0
        bin_high = cal_bins[1]
        self.assertEqual(bin_high["sample_count"], 3)
        self.assertGreater(bin_high["collision_event_count"], 0)

    def test_warning_evaluator(self) -> None:
        """Verify warning lead time calculation and false warnings."""
        evaluator = WarningEvaluator()

        # Warning issued at t=1.0s, Collision occurs at t=3.5s -> Lead time = 2.5s
        frames = []
        for i in range(50):
            t = i * 0.1
            is_warn = (t >= 1.0)
            has_col = (t >= 3.5)
            frames.append(
                AlignedFrame(
                    timestamp=t,
                    frame_index=i,
                    ego_gt={},
                    matched_pairs=(),
                    unmatched_gt=(),
                    unmatched_tracks=(),
                    gt_collision_state={"has_collided": has_col},
                    prism_output={
                        "risk_level": "HIGH" if is_warn else "SAFE",
                        "decision_action": "SLOW_DOWN" if is_warn else "CONTINUE",
                    },
                )
            )

        res = evaluator.evaluate_run(frames)
        self.assertTrue(res["has_collision_event"])
        self.assertTrue(res["warning_issued"])
        self.assertAlmostEqual(res["lead_time_s"], 2.5, places=3)
        self.assertFalse(res["false_warning_detected"])
        self.assertGreaterEqual(res["warning_maintained_ratio"], 0.99)

    def test_avoidance_evaluator(self) -> None:
        """Verify avoidance decision changes, steering reversals, and clearance ratios."""
        evaluator = AvoidanceEvaluator(min_safe_clearance_m=0.50)

        # Turn left, then turn right (reversal)
        actions = ["CONTINUE", "TURN_LEFT", "TURN_LEFT", "TURN_RIGHT", "STOP"]
        frames = []
        for i, act in enumerate(actions):
            frames.append(
                AlignedFrame(
                    timestamp=i * 0.1,
                    frame_index=i,
                    ego_gt={"x": 0.0, "z": 0.0, "radius": 0.35},
                    matched_pairs=(
                        ({"object_id": 1, "x": 0.0, "z": 2.0, "radius": 0.35}, {"track_id": 1}),
                    ),
                    unmatched_gt=(),
                    unmatched_tracks=(),
                    gt_collision_state={"has_collided": False},
                    prism_output={"decision_action": act},
                )
            )

        res = evaluator.evaluate_run(frames)
        self.assertEqual(res["decision_change_count"], 3)
        self.assertEqual(res["steering_reversal_count"], 1)
        # clearance = 2.0 - 0.70 = 1.30m >= 0.50m
        self.assertAlmostEqual(res["min_ground_clearance_m"], 1.30, places=3)
        self.assertEqual(res["safe_clearance_ratio"], 1.0)

    def test_failure_analyzer(self) -> None:
        """Verify extraction of diagnostic failure cases."""
        config = EvaluationConfig(
            diagnostic_ttc_error_threshold_s=2.0,
            diagnostic_pred_error_threshold_m=1.5,
        )
        analyzer = FailureAnalyzer(config)

        # Frame with TTC Error > 2.0s
        frame = AlignedFrame(
            timestamp=1.0,
            frame_index=10,
            ego_gt={},
            matched_pairs=(({"object_id": 5}, {"track_id": 105}),),
            unmatched_gt=(),
            unmatched_tracks=(),
            gt_collision_state={"time_to_collision": 2.0, "closest_point_of_approach": 0.1, "colliding_agent_id": 5},
            prism_output={"predicted_ttc": 6.5, "predicted_cpa": 1.2, "primary_threat_id": 105},
        )

        failures = analyzer.extract_failures(
            scenario_id="test_sc",
            scenario_family="test_fam",
            run_id="test_run_01",
            aligned_frames=[frame],
        )

        self.assertGreaterEqual(len(failures), 1)
        ttc_fails = [f for f in failures if f.failure_type == "TTC_ESTIMATION_ERROR"]
        self.assertEqual(len(ttc_fails), 1)
        self.assertEqual(ttc_fails[0].metric_value, 4.5)

    def test_robustness_evaluator(self) -> None:
        """Verify noise condition performance comparisons."""
        evaluator = RobustnessEvaluator()

        def make_dummy_result(run_id: str, noise: str, pos_mae: float) -> RunEvaluationResult:
            return RunEvaluationResult(
                scenario_id="sc_01",
                scenario_family="S01",
                run_id=run_id,
                num_frames=100,
                duration=5.0,
                noise_condition=noise,
                state_estimation={"position_2d": {"mae": pos_mae}, "velocity_2d": {"mae": 0.10}},
                tracking={"confirmation_rate": 1.0, "id_switch_count": 0, "mota": 0.95},
                prediction={"horizon_1.0s": {"mae": 0.20}, "horizon_2.0s": {"mae": 0.40}},
                ttc_dcpa={"error_stats": {"mae": 0.15}},
                risk={"multi_horizon_detection": {}},
                warning={"lead_time_s": 2.0, "false_warning_detected": False},
                avoidance={"decision_change_count": 1},
                failures=[],
            )

        runs = [
            make_dummy_result("run_1", "NO_NOISE", 0.05),
            make_dummy_result("run_2", "NO_NOISE", 0.07),
            make_dummy_result("run_3", "HIGH", 0.45),
        ]

        summary = evaluator.evaluate_noise_conditions(runs)
        self.assertIn("NO_NOISE", summary)
        self.assertIn("HIGH", summary)
        self.assertEqual(summary["NO_NOISE"]["run_count"], 2)
        self.assertAlmostEqual(summary["NO_NOISE"]["mean_position_mae_m"], 0.06, places=3)
        self.assertEqual(summary["HIGH"]["run_count"], 1)
        self.assertAlmostEqual(summary["HIGH"]["mean_position_mae_m"], 0.45, places=3)

    def test_prism_evaluator_end_to_end(self) -> None:
        """Test full end-to-end dataset evaluation using temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "dataset"
            output_dir = tmp_path / "output"
            run_dir = dataset_dir / "S01_stationary" / "S01_stationary_run_0001"
            run_dir.mkdir(parents=True, exist_ok=True)

            # 1. Create Manifest
            manifest = {
                "dataset_name": "test_dataset",
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

            # 2. Create Run Files
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
                {"timestamp": 0.0, "frame_index": 0, "ego": {"x": 0.0, "z": 0.0, "radius": 0.35}, "objects": [{"object_id": 1, "x": 0.0, "z": 3.0, "vx": 0.0, "vz": 0.0, "radius": 0.35}], "collision_state": {"has_collided": False, "time_to_collision": 3.0, "closest_point_of_approach": 0.0}},
                {"timestamp": 0.05, "frame_index": 1, "ego": {"x": 0.0, "z": 0.05, "radius": 0.35}, "objects": [{"object_id": 1, "x": 0.0, "z": 3.0, "vx": 0.0, "vz": 0.0, "radius": 0.35}], "collision_state": {"has_collided": False, "time_to_collision": 2.95, "closest_point_of_approach": 0.0}},
            ]
            with open(run_dir / "ground_truth.jsonl", "w", encoding="utf-8") as f:
                for line in gt_lines:
                    f.write(json.dumps(line) + "\n")

            obs_lines = [
                {"timestamp": 0.0, "frame_index": 0, "detections": []},
                {"timestamp": 0.05, "frame_index": 1, "detections": []},
            ]
            with open(run_dir / "observations.jsonl", "w", encoding="utf-8") as f:
                for line in obs_lines:
                    f.write(json.dumps(line) + "\n")

            prism_lines = [
                {"timestamp": 0.0, "frame_index": 0, "active_tracks": [{"track_id": 1, "x": 0.0, "z": 3.0, "vx": 0.0, "vz": 0.0, "lifecycle_state": "CONFIRMED"}], "predicted_ttc": 3.0, "predicted_cpa": 0.0, "risk_level": "LOW", "risk_score": 30.0, "decision_action": "CONTINUE"},
                {"timestamp": 0.05, "frame_index": 1, "active_tracks": [{"track_id": 1, "x": 0.0, "z": 3.0, "vx": 0.0, "vz": 0.0, "lifecycle_state": "CONFIRMED"}], "predicted_ttc": 2.95, "predicted_cpa": 0.0, "risk_level": "LOW", "risk_score": 32.0, "decision_action": "CONTINUE"},
            ]
            with open(run_dir / "prism_output.jsonl", "w", encoding="utf-8") as f:
                for line in prism_lines:
                    f.write(json.dumps(line) + "\n")

            # 3. Run Evaluation
            config = EvaluationConfig(dataset_dir=str(dataset_dir), output_dir=str(output_dir))
            evaluator = PRISMEvaluator(config)
            report, runs = evaluator.evaluate_dataset(output_dir=output_dir, export_csv=True)

            self.assertEqual(len(runs), 1)
            self.assertEqual(report.total_runs_evaluated, 1)
            self.assertEqual(report.total_frames_evaluated, 2)
            self.assertTrue((output_dir / "aggregate" / "aggregate_report.json").exists())
            self.assertTrue((output_dir / "run_metrics" / "S01_stationary_run_0001.json").exists())
            self.assertTrue((output_dir / "failures" / "failure_cases.jsonl").exists())
            self.assertTrue((output_dir / "csv" / "runs_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
