"""Comprehensive unit and integration tests for PRISM Part 16 Synthetic Experimental Framework."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from app.core.types import Vector2
from simulation.experiments.ground_truth_calc import (
    compute_constant_velocity_dcpa,
    compute_constant_velocity_ttc,
    evaluate_ground_truth_for_agent,
)
from simulation.experiments.models import (
    AgentSpecification,
    EnvironmentSpecification,
    NoiseConfiguration,
    ScenarioDefinition,
)
from simulation.experiments.noise_model import SyntheticPerceptionModel
from simulation.experiments.runner import BatchExperimentRunner, ExperimentRunner
from simulation.experiments.scenarios.registry import ScenarioRegistry
from simulation.experiments.scenarios.s01_stationary import create_s01_stationary_scenario
from simulation.experiments.scenarios.s02_head_on import create_s02_head_on_scenario
from simulation.experiments.scenarios.s03_lateral_crossing import (
    create_s03_crossing_collision_scenario,
    create_s03_crossing_miss_scenario,
)
from simulation.experiments.scenarios.s04_moving_away import create_s04_moving_away_scenario
from simulation.experiments.scenarios.s05_direction_change import create_s05_direction_change_scenario
from simulation.experiments.scenarios.s06_multiple_objects import create_s06_multiple_objects_scenario
from simulation.experiments.scenarios.s07_corridor_squeeze import create_s07_corridor_squeeze_scenario
from simulation.experiments.scenarios.s08_measurement_noise import create_s08_noise_scenario
from simulation.experiments.validator import DatasetValidationError, DatasetValidator


class TestPart16SyntheticExperiments(unittest.TestCase):
    """Test suite for Part 16 experimental scenario generation, ground-truth solvers, and runner."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="prism_test_part16_")
        self.output_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Scenario Generation & Registry Tests
    # -------------------------------------------------------------------------
    def test_all_eight_scenario_families_instantiate(self) -> None:
        """Verify all 8 scenario families instantiate with valid configurations."""
        scenarios = [
            create_s01_stationary_scenario(seed=101),
            create_s02_head_on_scenario(seed=102),
            create_s03_crossing_collision_scenario(seed=103),
            create_s03_crossing_miss_scenario(seed=104),
            create_s04_moving_away_scenario(seed=105),
            create_s05_direction_change_scenario(seed=106),
            create_s06_multiple_objects_scenario(num_objects=3, seed=107),
            create_s07_corridor_squeeze_scenario(seed=108),
            create_s08_noise_scenario(noise_level="MEDIUM_NOISE", seed=109),
        ]
        self.assertEqual(len(scenarios), 9)
        for sc in scenarios:
            self.assertIsInstance(sc, ScenarioDefinition)
            self.assertGreater(sc.duration, 0.0)
            self.assertGreater(sc.dt, 0.0)
            self.assertGreater(len(sc.agents), 0)

    def test_milestone_1_and_2_scenario_suites(self) -> None:
        """Verify registry creates correct scenario counts for Milestone 1 (15) and Milestone 2 (40)."""
        m1_scenarios = ScenarioRegistry.generate_milestone_1_scenarios(runs_per_family=5)
        self.assertEqual(len(m1_scenarios), 15)

        m2_scenarios = ScenarioRegistry.generate_milestone_2_benchmark_scenarios(runs_per_family=5)
        self.assertEqual(len(m2_scenarios), 40)

        # Check all 8 families are represented in Milestone 2
        families = {s.scenario_family for s in m2_scenarios}
        self.assertEqual(len(families), 8)
        self.assertEqual(families, set(ScenarioRegistry.get_supported_families()))

    # -------------------------------------------------------------------------
    # 2. Independent Ground-Truth Reference Calculation Tests
    # -------------------------------------------------------------------------
    def test_independent_head_on_analytical_ttc_and_dcpa(self) -> None:
        """Verify analytical TTC and dCPA for known head-on approach."""
        rel_pos = Vector2(0.0, 6.0)
        rel_vel = Vector2(0.0, -2.0)  # User +1.0 m/s, target -1.0 m/s => v_rel = -2.0 m/s
        r_col = 0.70  # User radius 0.35 + target radius 0.35

        t_cpa, d_cpa = compute_constant_velocity_dcpa(rel_pos, rel_vel)
        self.assertAlmostEqual(t_cpa, 3.0, places=4)
        self.assertAlmostEqual(d_cpa, 0.0, places=4)

        ttc = compute_constant_velocity_ttc(rel_pos, rel_vel, r_col)
        # Expected collision when dist = 0.70 => t = (6.0 - 0.70) / 2.0 = 2.65s
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, 2.65, places=4)

    def test_independent_crossing_collision_and_miss_solvers(self) -> None:
        """Verify collision vs near-miss differentiation in lateral crossing."""
        # 1. Direct Collision
        rel_pos = Vector2(-3.5, 3.5)
        rel_vel = Vector2(1.0, -1.0)
        r_col = 0.70

        t_cpa_coll, d_cpa_coll = compute_constant_velocity_dcpa(rel_pos, rel_vel)
        self.assertAlmostEqual(t_cpa_coll, 3.5, places=4)
        self.assertAlmostEqual(d_cpa_coll, 0.0, places=4)
        ttc_coll = compute_constant_velocity_ttc(rel_pos, rel_vel, r_col)
        self.assertIsNotNone(ttc_coll)

        # 2. Near Miss (crossing separated by time)
        rel_pos_miss = Vector2(-3.5, 3.5)
        rel_vel_miss = Vector2(2.5, -1.0)  # Faster crossing
        t_cpa_miss, d_cpa_miss = compute_constant_velocity_dcpa(rel_pos_miss, rel_vel_miss)
        self.assertGreater(d_cpa_miss, r_col)
        ttc_miss = compute_constant_velocity_ttc(rel_pos_miss, rel_vel_miss, r_col)
        self.assertIsNone(ttc_miss)

    def test_departing_motion_yields_none_ttc(self) -> None:
        """Verify departing motion has non-positive closing velocity and None TTC."""
        rel_pos = Vector2(0.0, 3.0)
        rel_vel = Vector2(0.0, 0.5)  # Moving away at +0.5 m/s
        r_col = 0.70

        t_cpa, d_cpa = compute_constant_velocity_dcpa(rel_pos, rel_vel)
        self.assertEqual(t_cpa, 0.0)
        self.assertAlmostEqual(d_cpa, 3.0, places=4)

        ttc = compute_constant_velocity_ttc(rel_pos, rel_vel, r_col)
        self.assertIsNone(ttc)

    def test_piecewise_direction_change_evaluation(self) -> None:
        """Verify ground truth evaluation across piecewise turn at t_turn."""
        target_spec = AgentSpecification(
            agent_id=1,
            class_id="person",
            initial_position=Vector2(0.0, 4.0),
            initial_velocity=Vector2(0.0, -1.0),
            behavior="direction_change",
            behavior_params={"turn_time": 2.0, "turn_angle_rad": math.pi / 2.0, "speed": 1.0},
        )
        ego_pos = Vector2(0.0, 0.0)
        ego_vel = Vector2(0.0, 0.0)

        # Before turn (t = 1.0s): moving along -Z
        state_1, _ = evaluate_ground_truth_for_agent(1.0, ego_pos, ego_vel, 0.35, target_spec)
        self.assertAlmostEqual(state_1.x, 0.0, places=4)
        self.assertAlmostEqual(state_1.z, 3.0, places=4)
        self.assertAlmostEqual(state_1.vx, 0.0, places=4)
        self.assertAlmostEqual(state_1.vz, -1.0, places=4)

        # After turn (t = 3.0s, 1s after turn at Z=2.0): moving along +X
        state_2, _ = evaluate_ground_truth_for_agent(3.0, ego_pos, ego_vel, 0.35, target_spec)
        self.assertAlmostEqual(state_2.x, 1.0, places=4)
        self.assertAlmostEqual(state_2.z, 2.0, places=4)
        self.assertAlmostEqual(state_2.vx, 1.0, places=4)
        self.assertAlmostEqual(state_2.vz, 0.0, places=4)

    # -------------------------------------------------------------------------
    # 3. Noise Model Determinism & Dropout Tests
    # -------------------------------------------------------------------------
    def test_noise_model_determinism_and_dropout(self) -> None:
        """Verify random seeding produces identical noise and dropout."""
        cfg_noisy = NoiseConfiguration(
            enabled=True,
            sigma_position_x=0.10,
            sigma_position_z=0.10,
            dropout_probability=0.20,
            seed=42,
        )
        model_1 = SyntheticPerceptionModel(cfg_noisy)
        model_2 = SyntheticPerceptionModel(cfg_noisy)

        gt_state = AgentSpecification(
            agent_id=1,
            class_id="person",
            initial_position=Vector2(1.0, 3.0),
            initial_velocity=Vector2(0.0, 0.0),
        )
        obj_gt, _ = evaluate_ground_truth_for_agent(0.0, Vector2(0, 0), Vector2(0, 0), 0.35, gt_state)

        # Generate 20 frames from both models
        for i in range(20):
            t = i * 0.05
            items_1, obs_1 = model_1.generate_observations([obj_gt], t)
            items_2, obs_2 = model_2.generate_observations([obj_gt], t)

            self.assertEqual(len(items_1), len(items_2))
            self.assertEqual(items_1[0].valid, items_2[0].valid)
            self.assertAlmostEqual(items_1[0].x, items_2[0].x, places=6)
            self.assertAlmostEqual(items_1[0].z, items_2[0].z, places=6)

    # -------------------------------------------------------------------------
    # 4. Zero Future Information Leakage & Execution Pipeline Tests
    # -------------------------------------------------------------------------
    def test_experiment_runner_step_by_step_execution(self) -> None:
        """Execute single run and verify output file creation and record matching."""
        scenario = create_s02_head_on_scenario(duration=3.0, dt=0.1, seed=555)
        runner = ExperimentRunner()
        run_dir = self.output_path / "test_run_s02"

        metadata, gt_recs, obs_recs, pr_recs = runner.run_scenario(scenario, output_dir=run_dir)

        # Check line counts match
        expected_frames = int(round(3.0 / 0.1)) + 1  # 31 frames
        self.assertEqual(len(gt_recs), expected_frames)
        self.assertEqual(len(obs_recs), expected_frames)
        self.assertEqual(len(pr_recs), expected_frames)
        self.assertEqual(metadata.num_frames, expected_frames)

        # Verify files on disk
        self.assertTrue((run_dir / "metadata.json").exists())
        self.assertTrue((run_dir / "ground_truth.jsonl").exists())
        self.assertTrue((run_dir / "observations.jsonl").exists())
        self.assertTrue((run_dir / "prism_output.jsonl").exists())

        # Check PRISM receives decisions and transitions
        actions = [p.decision_action for p in pr_recs]
        self.assertTrue(any(a in ("SLOW_DOWN", "CONTINUE", "CAUTION") for a in actions))
        self.assertTrue(any(a in ("EMERGENCY_STOP", "STOP", "TURN_LEFT", "TURN_RIGHT") for a in actions))

    def test_reproducibility_within_tolerances(self) -> None:
        """Verify identical seed produces identical output and different seeds diverge."""
        sc_a = create_s08_noise_scenario(noise_level="MEDIUM_NOISE", seed=777)
        sc_b = create_s08_noise_scenario(noise_level="MEDIUM_NOISE", seed=777)
        sc_diff = create_s08_noise_scenario(noise_level="MEDIUM_NOISE", seed=888)

        runner = ExperimentRunner()
        _, gt_a, obs_a, pr_a = runner.run_scenario(sc_a)
        _, gt_b, obs_b, pr_b = runner.run_scenario(sc_b)
        _, gt_d, obs_d, pr_d = runner.run_scenario(sc_diff)

        # 1. Exact equality for identical seed
        self.assertEqual(len(obs_a), len(obs_b))
        for o_a, o_b in zip(obs_a, obs_b):
            self.assertEqual(o_a.observations[0].valid, o_b.observations[0].valid)
            self.assertAlmostEqual(o_a.observations[0].x, o_b.observations[0].x, places=5)
            self.assertAlmostEqual(o_a.observations[0].z, o_b.observations[0].z, places=5)

        for p_a, p_b in zip(pr_a, pr_b):
            self.assertEqual(p_a.decision_action, p_b.decision_action)
            self.assertAlmostEqual(p_a.risk_score, p_b.risk_score, places=4)

        # 2. Divergence for different seed (observations differ)
        differ_count = sum(
            1 for o_a, o_d in zip(obs_a, obs_d)
            if abs(o_a.observations[0].x - o_d.observations[0].x) > 1e-4
            or abs(o_a.observations[0].z - o_d.observations[0].z) > 1e-4
        )
        self.assertGreater(differ_count, 10)

    # -------------------------------------------------------------------------
    # 5. Batch Runner, Manifest & Dataset Validator Tests
    # -------------------------------------------------------------------------
    def test_batch_runner_and_dataset_validator(self) -> None:
        """Generate a 6-run batch dataset and validate with DatasetValidator."""
        scenarios = [
            create_s01_stationary_scenario(initial_distance=5.0, seed=1, run_idx=1),
            create_s01_stationary_scenario(initial_distance=6.0, seed=2, run_idx=2),
            create_s02_head_on_scenario(initial_separation=5.5, seed=3, run_idx=1),
            create_s02_head_on_scenario(initial_separation=6.5, seed=4, run_idx=2),
            create_s03_crossing_collision_scenario(crossing_z=3.5, seed=5, run_idx=1),
            create_s03_crossing_miss_scenario(crossing_z=3.5, seed=6, run_idx=2),
        ]

        batch_runner = BatchExperimentRunner(base_output_dir=self.output_path)
        manifest = batch_runner.run_scenarios(scenarios, dataset_name="test_part16_batch")

        self.assertEqual(manifest.total_runs, 6)
        self.assertTrue((self.output_path / "dataset_manifest.json").exists())

        # Validate with DatasetValidator
        validator = DatasetValidator(self.output_path)
        val_result = validator.validate_dataset()
        self.assertEqual(val_result["status"], "PASSED")
        self.assertEqual(val_result["total_runs"], 6)

    def test_dataset_validator_catches_corrupted_timestamps(self) -> None:
        """Verify DatasetValidator raises DatasetValidationError on corrupted files."""
        scenario = create_s01_stationary_scenario(duration=2.0, dt=0.1, seed=99)
        runner = ExperimentRunner()
        run_dir = self.output_path / "corrupted_run"
        runner.run_scenario(scenario, output_dir=run_dir)

        # Corrupt one line in ground_truth.jsonl
        gt_file = run_dir / "ground_truth.jsonl"
        with open(gt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Modify timestamp of line 5 to make it non-monotonic
        bad_record = json.loads(lines[5])
        bad_record["timestamp"] = 0.0  # Invalid non-monotonic timestamp
        lines[5] = json.dumps(bad_record) + "\n"

        with open(gt_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        validator = DatasetValidator(self.output_path)
        with self.assertRaises(DatasetValidationError):
            validator.validate_run_directory(run_dir)


if __name__ == "__main__":
    unittest.main()
