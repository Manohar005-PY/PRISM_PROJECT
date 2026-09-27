"""Execution engine running PRISM ablation variants over frozen Part 16 observation streams."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from app.prediction.relative_motion import EgoMotionState
from app.runtime.pipeline import PRISMPipeline
from simulation.ablation.models import AblationVariant, VariantFrameOutput, VariantRunResult
from simulation.ablation.variants import create_variant_pipeline


class AblationRunner:
    """Executes PRISM variants over frozen Part 16 observation streams with zero future leakage."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def build_physical_observations(
        obs_items: Sequence[dict[str, Any]],
        timestamp: float,
    ) -> list[PhysicalObservation]:
        """Convert recorded observation items into PRISM PhysicalObservation structs.
        
        Strictly consumes only the spatial measurements recorded in the dataset frame,
        preventing any ground-truth state leakage.
        """
        prism_obs: list[PhysicalObservation] = []
        for item in obs_items:
            # Respect sensor valid / dropout flag
            if not item.get("valid", True):
                continue

            obs_x = float(item["x"])
            obs_z = float(item["z"])
            cov = item.get("covariance", ((0.01, 0.0), (0.0, 0.01)))
            cov_tuple = ((float(cov[0][0]), float(cov[0][1])), (float(cov[1][0]), float(cov[1][1])))
            conf = float(item.get("confidence", 0.95))
            obj_id = item.get("object_id", "0")
            cls_id = str(item.get("class_id", "1"))

            p_obs = PhysicalObservation(
                position=Vector2(x=obs_x, z=obs_z),
                measurement_covariance=cov_tuple,
                confidence=conf,
                timestamp=timestamp,
                detection_id=f"det_{obj_id}_{int(round(timestamp * 1000))}",
                class_name=cls_id,
            )
            prism_obs.append(p_obs)
        return prism_obs

    def run_pipeline_on_stream(
        self,
        pipeline: PRISMPipeline,
        obs_records: Sequence[dict[str, Any]],
        gt_records: Sequence[dict[str, Any]],
    ) -> list[VariantFrameOutput]:
        """Execute a PRISMPipeline instance frame-by-frame across a run's observation stream.
        
        Args:
            pipeline: Instrumented or standard PRISMPipeline instance.
            obs_records: Recorded observation stream from Part 16 dataset.
            gt_records: Ground truth records (used solely for ego kinematic state extraction).
            
        Returns:
            list of VariantFrameOutput instances containing recorded frame actions and risk scores.
        """
        pipeline.reset()
        outputs: list[VariantFrameOutput] = []

        num_frames = len(obs_records)
        for f_idx in range(num_frames):
            obs_rec = obs_records[f_idx]
            t = float(obs_rec["timestamp"])
            items = obs_rec.get("observations", [])

            # Ego state extraction from ground truth (only ego's own kinematic reference)
            ego_dict = gt_records[f_idx].get("ego", {}) if f_idx < len(gt_records) else {}
            ego_pos = Vector2(x=float(ego_dict.get("x", 0.0)), z=float(ego_dict.get("z", 0.0)))
            ego_vel = Vector2(x=float(ego_dict.get("vx", 0.0)), z=float(ego_dict.get("vz", 0.0)))
            ego_state = EgoMotionState(position=ego_pos, velocity=ego_vel, timestamp=t)

            # Build physical observations from recorded observation frame
            phys_obs = self.build_physical_observations(items, timestamp=t)

            # Process frame through pipeline
            t0 = time.perf_counter()
            runtime_res = pipeline.process_frame(
                raw_frame=None,
                timestamp=t,
                ego_state=ego_state,
                explicit_observations=phys_obs,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0

            dec = runtime_res.decision
            risk = runtime_res.risk_assessment

            rec_heading = dec.recommended_heading
            pred_ttc = risk.ttc if risk else None
            pred_cpa = risk.cpa_distance if risk else None

            outputs.append(
                VariantFrameOutput(
                    timestamp=t,
                    frame_index=f_idx,
                    action=dec.action.name,
                    reason=dec.reason.name,
                    risk_level=risk.risk_level.name if risk else "SAFE",
                    risk_score=float(risk.risk_score) if risk else 0.0,
                    predicted_ttc=float(pred_ttc) if pred_ttc is not None else None,
                    predicted_cpa=float(pred_cpa) if pred_cpa is not None else None,
                    recommended_heading_rad=float(rec_heading) if rec_heading is not None else None,
                    active_track_count=runtime_res.active_track_count,
                    latency_ms=lat_ms,
                )
            )

        return outputs
