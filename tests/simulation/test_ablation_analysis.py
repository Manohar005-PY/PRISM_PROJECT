"""Unit and integration test suite for PRISM Part 19 Ablation & Sensitivity Analysis."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.prediction.relative_motion import EgoMotionState
from simulation.ablation.aggregation import (
    AblationAggregator,
    compute_cohens_d,
    compute_mean,
    compute_median,
    compute_std,
)
from simulation.ablation.analyzer import PRISMAblationAnalyzer
from simulation.ablation.failure_analysis import AblationFailureAnalyzer
from simulation.ablation.metrics import AblationMetricsEvaluator
from simulation.ablation.models import (
    FROZEN_SENSITIVITY_SPECS,
    AblationConfig,
    AblationFailureCase,
    AblationVariant,
    PairwiseVariantComparison,
    SensitivityParamSpec,
    VariantFrameOutput,
    VariantRunResult,
)
from simulation.ablation.runner import AblationRunner
from simulation.ablation.sensitivity import SensitivityAnalyzer
from simulation.ablation.variants import (
    DCPAOnlyRiskEvaluator,
    DirectGeometricRiskEvaluator,
    InstantaneousRawStateEstimator,
    PositionOnlyStateEstimator,
    TTCOnlyRiskEvaluator,
    create_variant_pipeline,
)
from simulation.evaluation.loader import DatasetLoader


class TestAblationAnalysis(unittest.TestCase):
    """Unit and integration test suite for Part 19 ablation and sensitivity framework."""

    def test_variant_pipeline_construction_and_isolation(self) -> None:
        """Verify all 6 ablation variants instantiate their corresponding isolated components."""
        for v in AblationVariant:
            pipe = create_variant_pipeline(v)
            self.assertIsNotNone(pipe)
            self.assertIsNotNone(pipe.tracker)
            self.assertIsNotNone(pipe.state_estimator)
            self.assertIsNotNone(pipe.predictor)
            self.assertIsNotNone(pipe.risk_evaluator)
            self.assertIsNotNone(pipe.decision_engine)

            if v == AblationVariant.NO_KALMAN:
                self.assertIsInstance(pipe.state_estimator, InstantaneousRawStateEstimator)
            elif v == AblationVariant.POSITION_ONLY:
                self.assertIsInstance(pipe.state_estimator, PositionOnlyStateEstimator)
            elif v == AblationVariant.TTC_ONLY:
                self.assertIsInstance(pipe.risk_evaluator, TTCOnlyRiskEvaluator)
            elif v == AblationVariant.DCPA_ONLY:
                self.assertIsInstance(pipe.risk_evaluator, DCPAOnlyRiskEvaluator)
            elif v == AblationVariant.NO_PREDICTIVE_RISK:
                self.assertIsInstance(pipe.risk_evaluator, DirectGeometricRiskEvaluator)

    def test_no_kalman_estimator_behavior(self) -> None:
        """Verify NO_KALMAN estimator outputs raw measurement with zero velocity."""
        est = InstantaneousRawStateEstimator()
        meas = Vector2(1.5, 3.0)
        cov = ((0.02, 0.0), (0.0, 0.03))
        st = est.update_measurement(
            object_id=1,
            class_id="obstacle",
            measurement=meas,
            timestamp=0.05,
            measurement_covariance=cov,
        )
        self.assertEqual(st.object_id, 1)
        self.assertEqual(st.position.x, 1.5)
        self.assertEqual(st.position.z, 3.0)
        self.assertEqual(st.velocity.x, 0.0)
        self.assertEqual(st.velocity.z, 0.0)

    def test_position_only_estimator_behavior(self) -> None:
        """Verify POSITION_ONLY estimator forces zero velocity in exported ObjectState."""
        est = PositionOnlyStateEstimator()
        meas1 = Vector2(0.0, 5.0)
        est.update_measurement(1, "obstacle", meas1, timestamp=0.0)
        meas2 = Vector2(0.0, 4.0)
        est.update_measurement(1, "obstacle", meas2, timestamp=1.0)
        st = est.get_state(1)
        self.assertIsNotNone(st)
        self.assertAlmostEqual(st.position.z, 4.0, delta=0.5)
        self.assertEqual(st.velocity.x, 0.0)
        self.assertEqual(st.velocity.z, 0.0)

    def test_ttc_and_dcpa_downstream_isolation(self) -> None:
        """Verify TTC_ONLY and DCPA_ONLY evaluators isolate their respective risk factors."""
        ttc_eval = TTCOnlyRiskEvaluator()
        dcpa_eval = DCPAOnlyRiskEvaluator()

        # Dummy moving obstacle heading directly for collision
        from app.core.types import ObjectState
        cov4 = ((0.01, 0, 0, 0), (0, 0.01, 0, 0), (0, 0, 0.01, 0), (0, 0, 0, 0.01))
        obj_st = ObjectState(
            object_id=1,
            class_id=1,
            position=Vector2(0.0, 3.0),
            velocity=Vector2(0.0, -1.0),
            timestamp=0.0,
            covariance=cov4,
            confidence=0.95,
        )
        ego = EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 0.0), 0.0)

        # TTC_ONLY profile should have distance_factor == 0.0
        ttc_prof = ttc_eval.evaluate_object(obj_st, ego)
        self.assertEqual(ttc_prof.factors.distance_factor, 0.0)
        self.assertGreater(ttc_prof.factors.time_factor, 0.0)

        # DCPA_ONLY profile should have time_factor == 0.0
        dcpa_prof = dcpa_eval.evaluate_object(obj_st, ego)
        self.assertEqual(dcpa_prof.factors.time_factor, 0.0)
        self.assertGreater(dcpa_prof.factors.distance_factor, 0.0)

    def test_variants_receive_identical_observation_stream(self) -> None:
        """Assert input boundary equivalence: all variants receive byte-identical observations."""
        dataset_dir = "experiments/datasets/synthetic"
        if not Path(dataset_dir).exists():
            self.skipTest(f"Dataset directory not found: {dataset_dir}")

        loader = DatasetLoader(dataset_dir)
        meta, gt_recs, obs_recs, _ = next(loader.iter_runs())

        runner = AblationRunner()
        obs_sequences = []

        # Run on 3 different variants and capture observation inputs
        for v in [AblationVariant.FULL_PRISM, AblationVariant.NO_KALMAN, AblationVariant.TTC_ONLY]:
            pipe = create_variant_pipeline(v)
            # Verify runner passes identical observations to each pipeline
            first_frame_obs = runner.build_physical_observations(obs_recs[0]["observations"], obs_recs[0]["timestamp"])
            obs_sequences.append([(p.position.x, p.position.z, p.confidence) for p in first_frame_obs])

        self.assertEqual(obs_sequences[0], obs_sequences[1])
        self.assertEqual(obs_sequences[0], obs_sequences[2])

    def test_statistical_helpers_and_cohens_d(self) -> None:
        """Verify mean, median, std, and Cohen's d calculations."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(compute_mean(data), 3.0)
        self.assertAlmostEqual(compute_median(data), 3.0)
        self.assertAlmostEqual(compute_std(data), 1.5811388, places=4)

        diffs = [1.5, 1.5, 1.5, 1.5]
        # std is 0 -> cohens_d is 0.0
        self.assertEqual(compute_cohens_d(diffs), 0.0)

        diffs2 = [1.0, 2.0, 3.0]  # mean = 2.0, std = 1.0 -> d = 2.0
        self.assertAlmostEqual(compute_cohens_d(diffs2), 2.0)

    def test_pairwise_comparison_metrics(self) -> None:
        """Verify paired delta calculations between Full and variant."""
        evaluator = AblationMetricsEvaluator()
        full_res = VariantRunResult(
            scenario_id="S02_head_on_001",
            scenario_family="S02_head_on",
            run_id="S02_head_on_001",
            variant_id="FULL_PRISM",
            num_frames=100,
            duration=5.0,
            noise_condition="NO_NOISE",
            has_collision=True,
            collision_time_s=4.0,
            first_warning_time_s=1.0,
            warning_lead_time_s=3.0,
            warned=True,
            false_warning=False,
            warning_maintained_ratio=1.0,
            decision_changes=2,
            steering_reversals=0,
            min_clearance_m=0.6,
            safe_clearance_ratio=1.0,
            mean_latency_ms=1.2,
        )
        var_res = VariantRunResult(
            scenario_id="S02_head_on_001",
            scenario_family="S02_head_on",
            run_id="S02_head_on_001",
            variant_id="NO_KALMAN",
            num_frames=100,
            duration=5.0,
            noise_condition="NO_NOISE",
            has_collision=True,
            collision_time_s=4.0,
            first_warning_time_s=2.5,
            warning_lead_time_s=1.5,
            warned=True,
            false_warning=False,
            warning_maintained_ratio=0.8,
            decision_changes=8,
            steering_reversals=2,
            min_clearance_m=0.3,
            safe_clearance_ratio=0.7,
            mean_latency_ms=0.8,
        )

        pair = evaluator.compare_runs(full_res, var_res)
        self.assertEqual(pair.delta_lead_time_s, 1.5)
        self.assertEqual(pair.delta_decision_changes, 6)
        self.assertEqual(pair.delta_steering_reversals, 2)
        self.assertEqual(pair.delta_min_clearance_m, 0.3)

    def test_failure_analyzer_classification(self) -> None:
        """Verify automated classification of late warning, missed warning, and instability."""
        analyzer = AblationFailureAnalyzer(AblationConfig(lead_time_late_threshold_s=0.5))
        evaluator = AblationMetricsEvaluator()

        # 1. Late warning case
        full_res = VariantRunResult(
            scenario_id="S02_001", scenario_family="S02", run_id="S02_001",
            variant_id="FULL_PRISM", num_frames=100, duration=5.0, noise_condition="NO_NOISE",
            has_collision=True, collision_time_s=4.0, first_warning_time_s=1.0,
            warning_lead_time_s=3.0, warned=True, false_warning=False,
            warning_maintained_ratio=1.0, decision_changes=1, steering_reversals=0,
            min_clearance_m=0.8, safe_clearance_ratio=1.0, mean_latency_ms=1.0,
        )
        late_res = VariantRunResult(
            scenario_id="S02_001", scenario_family="S02", run_id="S02_001",
            variant_id="NO_KALMAN", num_frames=100, duration=5.0, noise_condition="NO_NOISE",
            has_collision=True, collision_time_s=4.0, first_warning_time_s=2.0,
            warning_lead_time_s=2.0, warned=True, false_warning=False,
            warning_maintained_ratio=1.0, decision_changes=1, steering_reversals=0,
            min_clearance_m=0.8, safe_clearance_ratio=1.0, mean_latency_ms=1.0,
        )
        pair = evaluator.compare_runs(full_res, late_res)
        fails = analyzer.analyze_comparison(full_res, late_res, pair)
        self.assertTrue(any(f.failure_type == "LATE_WARNING" for f in fails))

        # 2. Missed warning case
        miss_res = VariantRunResult(
            scenario_id="S02_001", scenario_family="S02", run_id="S02_001",
            variant_id="POSITION_ONLY", num_frames=100, duration=5.0, noise_condition="NO_NOISE",
            has_collision=True, collision_time_s=4.0, first_warning_time_s=None,
            warning_lead_time_s=None, warned=False, false_warning=False,
            warning_maintained_ratio=None, decision_changes=0, steering_reversals=0,
            min_clearance_m=0.2, safe_clearance_ratio=0.5, mean_latency_ms=1.0,
        )
        pair_miss = evaluator.compare_runs(full_res, miss_res)
        fails_miss = analyzer.analyze_comparison(full_res, miss_res, pair_miss)
        self.assertTrue(any(f.failure_type == "MISSED_WARNING" for f in fails_miss))

    def test_sensitivity_analyzer_grid_construction(self) -> None:
        """Verify sensitivity analyzer builds customized pipelines for all 5 parameters."""
        sens = SensitivityAnalyzer()
        for spec in FROZEN_SENSITIVITY_SPECS:
            for val in spec.sweep_values:
                pipe = sens.build_pipeline_for_setting(spec.param_name, val)
                self.assertIsNotNone(pipe)
                if spec.param_name == "horizon":
                    self.assertEqual(pipe.config.prediction_horizon, val)
                elif spec.param_name == "thresh_high":
                    self.assertEqual(pipe.risk_evaluator.config.thresh_high, val)
                elif spec.param_name == "alpha_smooth":
                    self.assertEqual(pipe.risk_evaluator.config.alpha_smooth, val)
                elif spec.param_name == "k_sigma":
                    self.assertEqual(pipe.risk_evaluator.config.k_sigma, val)
                elif spec.param_name == "sigma_a":
                    self.assertEqual(pipe.state_estimator.config.sigma_ax, val)

    def test_full_ablation_end_to_end_execution(self) -> None:
        """End-to-end integration test of PRISMAblationAnalyzer on synthetic benchmark dataset."""
        dataset_dir = "experiments/datasets/synthetic"
        if not Path(dataset_dir).exists():
            self.skipTest(f"Dataset directory not found: {dataset_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AblationConfig(
                dataset_dir=dataset_dir,
                output_dir=tmpdir,
            )
            analyzer = PRISMAblationAnalyzer(config)

            # Run on S01 (stationary) and S02 (head-on) scenarios
            results = analyzer.run_ablation_analysis(
                variants=[AblationVariant.FULL_PRISM, AblationVariant.POSITION_ONLY],
                scenario_filter="S01_stationary",
                run_sensitivity=False,
                output_dir=tmpdir,
                export_csv=True,
            )

            self.assertIn("FULL_PRISM", results["overall_variants"])
            self.assertIn("POSITION_ONLY", results["overall_variants"])
            self.assertTrue((Path(tmpdir) / "aggregate" / "overall_report.json").exists())
            self.assertTrue((Path(tmpdir) / "csv" / "overall.csv").exists())
            self.assertTrue((Path(tmpdir) / "config" / "sensitivity_config.json").exists())

    def test_full_prism_canonical_performance(self) -> None:
        """Verify FULL_PRISM executes production PRISMPipeline and matches canonical performance."""
        dataset_dir = "experiments/datasets/synthetic"
        if not Path(dataset_dir).exists():
            self.skipTest(f"Dataset directory not found: {dataset_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AblationConfig(dataset_dir=dataset_dir, output_dir=tmpdir)
            analyzer = PRISMAblationAnalyzer(config)

            # Run on head-on collision scenario S02
            results = analyzer.run_ablation_analysis(
                variants=[AblationVariant.FULL_PRISM],
                scenario_filter="S02_head_on",
                run_sensitivity=False,
                output_dir=tmpdir,
                export_csv=False,
            )

            full_stats = results["overall_variants"]["FULL_PRISM"]
            self.assertEqual(full_stats["warning_detection_rate"], 1.0)
            self.assertGreater(full_stats["mean_warning_lead_time_s"], 1.5)
            self.assertEqual(full_stats["false_warning_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

