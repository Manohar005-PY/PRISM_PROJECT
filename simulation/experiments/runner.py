"""Step-by-step experiment execution engine and dataset generation runner."""

from __future__ import annotations

from dataclasses import asdict
import datetime
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from app.core.types import Vector2
from app.decision.engine import DecisionEngine
from app.perception.pipeline import PerceptionPipeline
from app.perception.adapters.synthetic import SyntheticDetector
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker

from simulation.experiments.ground_truth_calc import evaluate_ground_truth_for_agent
from simulation.experiments.models import (
    DatasetManifest,
    GroundTruthCollisionState,
    GroundTruthObjectState,
    GroundTruthState,
    ObservationRecord,
    PRISMOutputRecord,
    RunMetadata,
    ScenarioDefinition,
    TrackOutputRecord,
)
from simulation.experiments.noise_model import SyntheticPerceptionModel


def get_git_commit_hash() -> str:
    """Safely obtain current git commit hash, falling back to 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def create_standard_prism_pipeline() -> PRISMPipeline:
    """Create a standard PRISMPipeline instance configured for synthetic benchmark evaluation."""
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)
    detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=0.0, random_seed=42)
    perc_pipe = PerceptionPipeline(detector, projector)

    state_estimator = KalmanStateEstimator(KalmanFilterConfig())
    tracker = MultiObjectTracker(
        TrackingConfig(confirmation_hits=2, gating_threshold=9.21),
        state_estimator=state_estimator,
    )
    predictor = ConstantVelocityPredictor()
    risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=0.50))
    decision_engine = DecisionEngine()

    return PRISMPipeline(
        perception_pipeline=perc_pipe,
        tracker=tracker,
        state_estimator=state_estimator,
        predictor=predictor,
        risk_evaluator=risk_evaluator,
        decision_engine=decision_engine,
        config=RuntimeConfig(prediction_horizon=3.0),
    )


class ExperimentRunner:
    """Executes a single scenario run step-by-step with zero future information leakage."""

    def __init__(self, pipeline_factory: Callable[[], PRISMPipeline] | None = None) -> None:
        self.pipeline_factory: Callable[[], PRISMPipeline] = (
            pipeline_factory if pipeline_factory is not None else create_standard_prism_pipeline
        )

    def run_scenario(
        self,
        scenario: ScenarioDefinition,
        output_dir: Path | None = None,
    ) -> tuple[RunMetadata, list[GroundTruthState], list[ObservationRecord], list[PRISMOutputRecord]]:
        """Execute scenario definition and optionally write formatted dataset files to output_dir."""
        # 1. Instantiate clean pipeline and seed
        pipeline = self.pipeline_factory()
        perception_model = SyntheticPerceptionModel(scenario.noise_config)
        perception_model.reseed(scenario.random_seed)

        dt = scenario.dt
        duration = scenario.duration
        num_steps = int(round(duration / dt)) + 1

        gt_records: list[GroundTruthState] = []
        obs_records: list[ObservationRecord] = []
        prism_records: list[PRISMOutputRecord] = []

        ego_spec = scenario.ego_agent
        first_collision_time: float | None = None

        for frame_idx in range(num_steps):
            current_time = round(frame_idx * dt, 6)

            # A. Ground Truth Kinematic Advancement
            ego_pos_x = ego_spec.initial_position.x + ego_spec.initial_velocity.x * current_time
            ego_pos_z = ego_spec.initial_position.z + ego_spec.initial_velocity.z * current_time
            ego_pos = Vector2(x=ego_pos_x, z=ego_pos_z)
            ego_vel = ego_spec.initial_velocity

            ego_gt = GroundTruthObjectState(
                object_id=ego_spec.agent_id,
                class_id=ego_spec.class_id,
                x=round(ego_pos.x, 6),
                z=round(ego_pos.z, 6),
                vx=round(ego_vel.x, 6),
                vz=round(ego_vel.z, 6),
                ax=round(ego_spec.initial_acceleration.x, 6),
                az=round(ego_spec.initial_acceleration.z, 6),
                radius=ego_spec.physical_radius,
                heading_rad=round(math.atan2(ego_vel.x, ego_vel.z) if (ego_vel.x != 0 or ego_vel.z != 0) else 0.0, 6),
            )

            object_gts: list[GroundTruthObjectState] = []
            collision_states: list[GroundTruthCollisionState] = []

            for agent_spec in scenario.agents:
                obj_gt, col_state = evaluate_ground_truth_for_agent(
                    current_time=current_time,
                    ego_pos=ego_pos,
                    ego_vel=ego_vel,
                    ego_radius=ego_spec.physical_radius,
                    target_spec=agent_spec,
                )
                object_gts.append(obj_gt)
                collision_states.append(col_state)

            # Combined global collision state
            any_collided = any(c.has_collided for c in collision_states)
            if any_collided and first_collision_time is None:
                first_collision_time = current_time

            # Min TTC across agents
            valid_ttcs = [c.time_to_collision for c in collision_states if c.time_to_collision is not None]
            min_ttc = min(valid_ttcs) if valid_ttcs else None

            # Min dCPA across agents
            min_dcpa_state = min(collision_states, key=lambda c: c.closest_point_of_approach)

            global_collision_state = GroundTruthCollisionState(
                has_collided=any_collided,
                time_to_collision=min_ttc,
                closest_point_of_approach=min_dcpa_state.closest_point_of_approach,
                t_cpa=min_dcpa_state.t_cpa,
                first_collision_timestamp=first_collision_time,
                colliding_agent_id=min_dcpa_state.colliding_agent_id if any_collided else None,
            )

            gt_frame = GroundTruthState(
                timestamp=current_time,
                frame_index=frame_idx,
                ego=ego_gt,
                objects=tuple(object_gts),
                collision_state=global_collision_state,
            )
            gt_records.append(gt_frame)

            # B. Synthetic Observation Generation (No Future Leakage)
            obs_items, prism_phys_obs = perception_model.generate_observations(
                ground_truth_objects=object_gts,
                timestamp=current_time,
            )
            obs_record = ObservationRecord(
                timestamp=current_time,
                frame_index=frame_idx,
                observations=tuple(obs_items),
            )
            obs_records.append(obs_record)

            # C. Feed into PRISM Pipeline
            t0 = time.perf_counter()
            runtime_result = pipeline.process_frame(
                raw_frame=None,
                timestamp=current_time,
                ego_state=EgoMotionState(position=ego_pos, velocity=ego_vel, timestamp=current_time),
                explicit_observations=prism_phys_obs,
            )
            t_eval_ms = (time.perf_counter() - t0) * 1000.0

            # D. Extract PRISM Output Record
            active_tracks: list[TrackOutputRecord] = []
            for trk in pipeline.tracker.active_tracks:
                st = pipeline.state_estimator.get_state(trk.track_id)
                pos = st.position if st else Vector2(0.0, 0.0)
                vel = st.velocity if st else Vector2(0.0, 0.0)
                active_tracks.append(
                    TrackOutputRecord(
                        track_id=trk.track_id,
                        class_name=trk.class_name,
                        lifecycle_state=trk.status.name,
                        x=round(pos.x, 4),
                        z=round(pos.z, 4),
                        vx=round(vel.x, 4),
                        vz=round(vel.z, 4),
                        hits=trk.hits,
                        missed_frames=trk.missed_frames,
                    )
                )

            dec = runtime_result.decision
            risk = runtime_result.risk_assessment
            rec_heading_rad = dec.recommended_heading
            rec_heading_deg = round(math.degrees(rec_heading_rad), 2) if rec_heading_rad is not None else None

            # Primary threat CPA & TTC if available
            pred_ttc = risk.ttc if risk else None
            pred_cpa = risk.cpa_distance if risk else None

            prism_out = PRISMOutputRecord(
                timestamp=current_time,
                frame_index=frame_idx,
                runtime_status=runtime_result.runtime_status.name,
                decision_action=dec.action.name,
                decision_reason=dec.reason.name,
                recommended_heading_rad=round(rec_heading_rad, 4) if rec_heading_rad is not None else None,
                recommended_heading_deg=rec_heading_deg,
                primary_threat_id=dec.primary_threat_id,
                risk_score=round(risk.risk_score, 2) if risk else 0.0,
                risk_level=risk.risk_level.name if risk else "SAFE",
                predicted_ttc=round(pred_ttc, 4) if pred_ttc is not None else None,
                predicted_cpa=round(pred_cpa, 4) if pred_cpa is not None else None,
                active_track_count=runtime_result.active_track_count,
                active_tracks=tuple(active_tracks),
                latency_ms=round(t_eval_ms, 3),
            )
            prism_records.append(prism_out)

        # 2. Build Run Metadata
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata = RunMetadata(
            experiment_id=f"exp_{scenario.scenario_family}_{scenario.random_seed}",
            scenario_id=scenario.scenario_id,
            scenario_family=scenario.scenario_family,
            run_id=scenario.scenario_id,
            random_seed=scenario.random_seed,
            scenario_version="1.0.0",
            experiment_config_version="1.0.0",
            generator_version="16.0.0",
            git_commit=get_git_commit_hash(),
            dt=dt,
            duration=duration,
            num_frames=len(gt_records),
            noise_config=asdict(scenario.noise_config),
            parameters=scenario.parameters,
            created_at=now_str,
            platform_info={
                "system": platform.system(),
                "python_version": sys.version,
                "machine": platform.machine(),
            },
        )

        # 3. Write outputs if directory provided
        if output_dir is not None:
            self._write_run_files(output_dir, metadata, gt_records, obs_records, prism_records)

        return metadata, gt_records, obs_records, prism_records

    def _write_run_files(
        self,
        run_dir: Path,
        metadata: RunMetadata,
        gt_records: list[GroundTruthState],
        obs_records: list[ObservationRecord],
        prism_records: list[PRISMOutputRecord],
    ) -> None:
        """Write metadata.json, ground_truth.jsonl, observations.jsonl, and prism_output.jsonl."""
        run_dir.mkdir(parents=True, exist_ok=True)

        # metadata.json
        meta_path = run_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # ground_truth.jsonl
        gt_path = run_dir / "ground_truth.jsonl"
        with open(gt_path, "w", encoding="utf-8") as f:
            for gt in gt_records:
                f.write(json.dumps(gt.to_dict()) + "\n")

        # observations.jsonl
        obs_path = run_dir / "observations.jsonl"
        with open(obs_path, "w", encoding="utf-8") as f:
            for obs in obs_records:
                f.write(json.dumps(obs.to_dict()) + "\n")

        # prism_output.jsonl
        prism_path = run_dir / "prism_output.jsonl"
        with open(prism_path, "w", encoding="utf-8") as f:
            for p in prism_records:
                f.write(json.dumps(p.to_dict()) + "\n")


class BatchExperimentRunner:
    """Executes a suite of scenarios, writes dataset directory structure, and builds manifest."""

    def __init__(self, base_output_dir: Path | str = "experiments/datasets/synthetic") -> None:
        self.base_output_dir: Path = Path(base_output_dir)
        self.runner: ExperimentRunner = ExperimentRunner()

    def run_scenarios(
        self,
        scenarios: Sequence[ScenarioDefinition],
        dataset_name: str = "prism_synthetic_benchmark",
        version: str = "1.0.0",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> DatasetManifest:
        """Execute all scenarios and write dataset_manifest.json."""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        scenario_manifests: list[dict[str, Any]] = []
        total_frames = 0
        families_seen: set[str] = set()

        total = len(scenarios)
        for i, scenario in enumerate(scenarios, start=1):
            if progress_callback:
                progress_callback(i, total, scenario.scenario_id)

            run_dir = self.base_output_dir / scenario.scenario_family / scenario.scenario_id
            metadata, gt, obs, pr = self.runner.run_scenario(scenario, output_dir=run_dir)

            total_frames += len(gt)
            families_seen.add(scenario.scenario_family)

            scenario_manifests.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_family": scenario.scenario_family,
                    "run_dir": str(run_dir.relative_to(self.base_output_dir)),
                    "num_frames": len(gt),
                    "duration": scenario.duration,
                    "dt": scenario.dt,
                    "random_seed": scenario.random_seed,
                    "parameters": scenario.parameters,
                }
            )

        manifest = DatasetManifest(
            dataset_name=dataset_name,
            version=version,
            scenario_families=sorted(list(families_seen)),
            total_scenarios=len(scenarios),
            total_runs=len(scenarios),
            total_frames=total_frames,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            generator_version="16.0.0",
            git_commit=get_git_commit_hash(),
            scenarios=scenario_manifests,
        )

        manifest_path = self.base_output_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return manifest
