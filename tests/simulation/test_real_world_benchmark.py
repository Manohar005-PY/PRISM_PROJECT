"""Unit and integration test suite for PRISM Part 20 Real-World Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation.real_world.aggregation import (
    RealWorldAggregator,
    compute_mean,
    compute_median,
    compute_std,
)
from simulation.real_world.analyzer import PRISMRealWorldAnalyzer
from simulation.real_world.detector import RealWorldDetector
from simulation.real_world.failure_analysis import RealWorldFailureAnalyzer
from simulation.real_world.loader import RealWorldDatasetLoader
from simulation.real_world.metrics import RealWorldMetricsEvaluator
from simulation.real_world.models import (
    RealWorldCameraConfig,
    RealWorldConfig,
    RealWorldDetectionBox,
    RealWorldFailureCase,
    RealWorldFrameResult,
    RealWorldRunMetadata,
    RealWorldRunResult,
)
from simulation.real_world.projector import RealWorldGroundProjector


class TestRealWorldBenchmark(unittest.TestCase):
    """Test suite for Part 20 real-world video benchmark framework."""

    def test_ground_plane_projector_geometry(self) -> None:
        """Verify 2D pixel bottom-center point correctly projects to 3D Cartesian ground plane."""
        cam_cfg = RealWorldCameraConfig(
            mounting_height_m=1.25,
            pitch_rad=-0.05,
            resolution=(720, 1280),
            hfov_deg=65.0,
            vfov_deg=80.0,
        )
        projector = RealWorldGroundProjector(cam_cfg)

        # Centered bounding box located in lower portion of image (close obstacle)
        box_close = RealWorldDetectionBox(
            x_min=260.0,
            y_min=500.0,
            x_max=460.0,
            y_max=1100.0,
            confidence=0.90,
        )
        obs_close = projector.project_detection(box_close, timestamp=0.0)
        self.assertIsNotNone(obs_close)
        self.assertAlmostEqual(obs_close.position.x, 0.0, delta=0.2)
        self.assertGreater(obs_close.position.z, 0.5)
        self.assertLess(obs_close.position.z, 4.0)

        # Centered bounding box located near horizon (far obstacle)
        box_far = RealWorldDetectionBox(
            x_min=330.0,
            y_min=650.0,
            x_max=390.0,
            y_max=750.0,
            confidence=0.85,
        )
        obs_far = projector.project_detection(box_far, timestamp=0.0)
        self.assertIsNotNone(obs_far)
        self.assertGreater(obs_far.position.z, obs_close.position.z)

    def test_detector_interface_and_fallback(self) -> None:
        """Verify detector initializes and runs on dummy frame."""
        detector = RealWorldDetector(detector_type="auto")
        self.assertIn(detector.active_backend, ("yolo", "hog", "none"))

        dummy_frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame)
        self.assertIsInstance(dets, list)

    def test_manifest_loader(self) -> None:
        """Verify loading and resolving runs from dataset_manifest.json."""
        dataset_dir = "experiments/datasets/real_world"
        if not Path(dataset_dir).exists():
            self.skipTest("Real-world dataset directory not found.")

        loader = RealWorldDatasetLoader(dataset_dir)
        runs = list(loader.iter_runs())
        self.assertGreaterEqual(len(runs), 10)
        for r in runs:
            self.assertTrue(Path(r.video_path).exists())
            self.assertIn(r.scenario_family, [
                "R01_stationary", "R02_head_on", "R03_lateral_crossing",
                "R04_moving_away", "R05_direction_change", "R06_multiple_objects"
            ])

    def test_metrics_evaluator_lead_time_and_false_alarms(self) -> None:
        """Verify lead time calculation on collision run and false alarm detection on safe run."""
        evaluator = RealWorldMetricsEvaluator()

        # 1. Collision run where PRISM warns at t=1.0s, collision at t=3.5s -> Lead = 2.5s
        meta_col = RealWorldRunMetadata(
            run_id="R02_test",
            scenario_family="R02_head_on",
            video_path="",
            has_collision=True,
            collision_timestamp_s=3.5,
            num_frames=100,
            duration_s=3.33,
            fps=30.0,
            resolution=(720, 1280),
            description="",
            environment="indoor",
        )
        frames_col = [
            RealWorldFrameResult(
                timestamp=round(i * 0.0333, 4),
                frame_index=i,
                raw_detections_count=1,
                projected_observations_count=1,
                active_tracks_count=1,
                decision_action="STOP" if i >= 30 else "CONTINUE",
                decision_reason="EMERGENCY_COLLISION_RISK" if i >= 30 else "NONE",
                risk_level="HIGH" if i >= 30 else "SAFE",
                risk_score=85.0 if i >= 30 else 10.0,
                predicted_ttc=1.5 if i >= 30 else None,
                predicted_cpa=0.2 if i >= 30 else None,
                primary_threat_id=1,
                recommended_heading_deg=0.0,
                latency_ms=1.5,
            )
            for i in range(100)
        ]
        res_col = evaluator.evaluate_run(meta_col, frames_col)
        self.assertTrue(res_col.warned)
        self.assertFalse(res_col.false_warning)
        self.assertAlmostEqual(res_col.warning_lead_time_s, 2.50, delta=0.05)
        self.assertEqual(res_col.tracking_continuity_ratio, 1.0)

        # 2. Safe run with no collision
        meta_safe = RealWorldRunMetadata(
            run_id="R04_test",
            scenario_family="R04_moving_away",
            video_path="",
            has_collision=False,
            collision_timestamp_s=None,
            num_frames=100,
            duration_s=3.33,
            fps=30.0,
            resolution=(720, 1280),
            description="",
            environment="indoor",
        )
        frames_safe = [
            RealWorldFrameResult(
                timestamp=round(i * 0.0333, 4),
                frame_index=i,
                raw_detections_count=1,
                projected_observations_count=1,
                active_tracks_count=1,
                decision_action="CONTINUE",
                decision_reason="NONE",
                risk_level="SAFE",
                risk_score=5.0,
                predicted_ttc=None,
                predicted_cpa=None,
                primary_threat_id=None,
                recommended_heading_deg=None,
                latency_ms=1.5,
            )
            for i in range(100)
        ]
        res_safe = evaluator.evaluate_run(meta_safe, frames_safe)
        self.assertFalse(res_safe.warned)
        self.assertFalse(res_safe.false_warning)
        self.assertIsNone(res_safe.warning_lead_time_s)

    def test_failure_analyzer_classification(self) -> None:
        """Verify automated failure classification of missed warning and steering flutter."""
        analyzer = RealWorldFailureAnalyzer()

        # Missed warning result
        res_miss = RealWorldRunResult(
            run_id="R02_miss",
            scenario_family="R02_head_on",
            video_path="",
            has_collision=True,
            collision_timestamp_s=3.0,
            first_warning_timestamp_s=None,
            warning_lead_time_s=None,
            warned=False,
            false_warning=False,
            total_frames=100,
            duration_s=3.33,
            fps=30.0,
            mean_detection_count=1.0,
            tracking_continuity_ratio=0.8,
            mean_latency_ms=1.2,
            decision_changes=0,
            steering_reversals=0,
        )
        fails = analyzer.analyze_run(res_miss)
        self.assertTrue(any(f.failure_type == "MISSED_WARNING" for f in fails))

    def test_tracking_continuity_bounded_and_coasting_metrics(self) -> None:
        """Verify tracking continuity remains <= 100% and coasting/dropouts are preserved."""
        evaluator = RealWorldMetricsEvaluator()

        meta = RealWorldRunMetadata(
            run_id="R05_coasting_test",
            scenario_family="R05_direction_change",
            video_path="",
            has_collision=True,
            collision_timestamp_s=2.5,
            num_frames=73,
            duration_s=2.43,
            fps=30.0,
            resolution=(720, 1280),
            description="Coasting test",
            environment="indoor",
        )

        # 66 frames with raw detection and track (0..65)
        # 7 frames with NO raw detection but active track due to Kalman coasting (66..72)
        frames = []
        for i in range(73):
            is_detected = (i < 66)
            has_track = True  # Coasting through all 7 dropout frames
            frames.append(
                RealWorldFrameResult(
                    timestamp=round(i * 0.0333, 4),
                    frame_index=i,
                    raw_detections_count=1 if is_detected else 0,
                    projected_observations_count=1 if is_detected else 0,
                    active_tracks_count=1 if has_track else 0,
                    decision_action="CONTINUE",
                    decision_reason="NONE",
                    risk_level="SAFE",
                    risk_score=5.0,
                    predicted_ttc=None,
                    predicted_cpa=None,
                    primary_threat_id=None,
                    recommended_heading_deg=None,
                    latency_ms=1.0,
                )
            )

        res = evaluator.evaluate_run(meta, frames)

        # 1. Tracking continuity ratio must be bounded to <= 1.0 (Detection-Track Association Coverage)
        self.assertIsNotNone(res.tracking_continuity_ratio)
        self.assertLessEqual(res.tracking_continuity_ratio, 1.0)
        self.assertEqual(res.tracking_continuity_ratio, 1.0)
        self.assertEqual(res.detection_association_coverage_ratio, 1.0)

        # 2. Track Coverage = 73 / 73 = 1.0 (100%)
        self.assertEqual(res.track_coverage_ratio, 1.0)

        # 3. Zero-Detection Frames = 7
        self.assertEqual(res.zero_detection_frames, 7)
        self.assertEqual(res.detector_dropout_frames, 7)

        # 4. Coasting Frames = 7 (raw_det == 0 and active_tracks > 0)
        self.assertEqual(res.coasting_frames, 7)

        # 5. Gap Bridging Rate = 7 / 7 = 1.0 (100%)
        self.assertEqual(res.gap_bridging_rate, 1.0)
        self.assertEqual(res.dropout_bridging_rate, 1.0)

    def test_undefined_ratio_division_by_zero(self) -> None:
        """Verify that zero denominators return None (undefined), not artificial numbers or division errors."""
        evaluator = RealWorldMetricsEvaluator()
        meta_empty = RealWorldRunMetadata(
            run_id="empty_run",
            scenario_family="R01_stationary",
            video_path="",
            has_collision=False,
            collision_timestamp_s=None,
            num_frames=50,
            duration_s=1.67,
            fps=30.0,
            resolution=(720, 1280),
            description="",
            environment="indoor",
        )

        # 50 frames with NO detections and NO tracks
        frames_empty = [
            RealWorldFrameResult(
                timestamp=round(i * 0.0333, 4),
                frame_index=i,
                raw_detections_count=0,
                projected_observations_count=0,
                active_tracks_count=0,
                decision_action="CONTINUE",
                decision_reason="NONE",
                risk_level="SAFE",
                risk_score=0.0,
                predicted_ttc=None,
                predicted_cpa=None,
                primary_threat_id=None,
                recommended_heading_deg=None,
                latency_ms=1.0,
            )
            for i in range(50)
        ]

        res = evaluator.evaluate_run(meta_empty, frames_empty)
        # Because raw_detections_count == 0 across all frames:
        self.assertIsNone(res.detection_association_coverage_ratio)
        self.assertIsNone(res.tracking_continuity_ratio)
        self.assertEqual(res.track_coverage_ratio, 0.0)
        self.assertEqual(res.zero_detection_frames, 50)
        self.assertEqual(res.coasting_frames, 0)
        self.assertEqual(res.gap_bridging_rate, 0.0)

        # Aggregation distinguishing None from 0.0
        aggregator = RealWorldAggregator()
        summary = aggregator.compile_summary([res])
        self.assertIsNone(summary["collision_event_recall"])  # 0 collision runs -> None (N/A), not 0.0
        self.assertEqual(summary["proximity_caution_alert_rate"], 0.0)  # 1 safe run with 0 caution -> 0.0 (0%)
        self.assertEqual(summary["false_collision_warning_rate"], 0.0)

    def test_warning_event_semantics_separation(self) -> None:
        """Verify separation of collision recall, proximity caution alerts, and false collision warnings."""
        evaluator = RealWorldMetricsEvaluator()

        meta_safe = RealWorldRunMetadata(
            run_id="safe_caution_run",
            scenario_family="R04_moving_away",
            video_path="",
            has_collision=False,
            collision_timestamp_s=None,
            num_frames=20,
            duration_s=0.67,
            fps=30.0,
            resolution=(720, 1280),
            description="",
            environment="indoor",
        )

        # Safe run where PRISM issues cautionary SLOW_DOWN (proximity alert) but NOT emergency STOP
        frames_caution = [
            RealWorldFrameResult(
                timestamp=round(i * 0.0333, 4),
                frame_index=i,
                raw_detections_count=1,
                projected_observations_count=1,
                active_tracks_count=1,
                decision_action="SLOW_DOWN" if i >= 10 else "CONTINUE",
                decision_reason="SPEED_ADAPTATION" if i >= 10 else "NONE",
                risk_level="MODERATE" if i >= 10 else "SAFE",
                risk_score=40.0 if i >= 10 else 5.0,
                predicted_ttc=None,
                predicted_cpa=1.8,
                primary_threat_id=1,
                recommended_heading_deg=None,
                latency_ms=1.0,
            )
            for i in range(20)
        ]

        res = evaluator.evaluate_run(meta_safe, frames_caution)
        self.assertTrue(res.caution_alert_issued)
        self.assertTrue(res.proximity_caution_alert)
        self.assertFalse(res.emergency_warning_issued)
        self.assertFalse(res.false_collision_warning)
        self.assertFalse(res.collision_recalled)
        self.assertFalse(res.collision_missed)

    def test_statistical_helpers(self) -> None:
        """Verify basic math helper functions and None handling."""
        vals = [2.0, 4.0, 6.0]
        self.assertAlmostEqual(compute_mean(vals), 4.0)
        self.assertAlmostEqual(compute_median(vals), 4.0)
        self.assertAlmostEqual(compute_std(vals), 2.0)

        # None filtering
        vals_with_none = [2.0, None, 4.0, 6.0]
        self.assertAlmostEqual(compute_mean(vals_with_none), 4.0)
        self.assertAlmostEqual(compute_median(vals_with_none), 4.0)
        self.assertAlmostEqual(compute_std(vals_with_none), 2.0)

        # All None
        self.assertIsNone(compute_mean([None, None]))
        self.assertIsNone(compute_median([None]))
        self.assertIsNone(compute_std([None]))

    def test_end_to_end_single_video_run(self) -> None:
        """End-to-end integration test of PRISMRealWorldAnalyzer on a real video clip."""
        dataset_dir = "experiments/datasets/real_world"
        if not Path(dataset_dir).exists():
            self.skipTest("Real-world dataset directory not found.")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RealWorldConfig(
                dataset_dir=dataset_dir,
                output_dir=tmpdir,
                detector_type="auto",
            )
            analyzer = PRISMRealWorldAnalyzer(config)

            # Evaluate just one scenario run
            results = analyzer.evaluate_dataset(
                scenario_filter="R01_stationary",
                run_filter="R01_stationary_001",
                output_dir=tmpdir,
                export_csv=True,
            )

            self.assertEqual(results["total_runs_evaluated"], 1)
            self.assertTrue((Path(tmpdir) / "aggregate" / "overall_report.json").exists())
            self.assertTrue((Path(tmpdir) / "csv" / "overall.csv").exists())


if __name__ == "__main__":
    unittest.main()
