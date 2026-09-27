"""Execution engine running PRISMPipeline over real-world video streams."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from app.prediction.relative_motion import EgoMotionState
from app.runtime.pipeline import PRISMPipeline
from simulation.ablation.variants import create_variant_pipeline
from simulation.real_world.detector import RealWorldDetector
from simulation.real_world.models import (
    RealWorldConfig,
    RealWorldFrameResult,
    RealWorldRunMetadata,
)
from simulation.real_world.projector import RealWorldGroundProjector
from simulation.real_world.video_reader import VideoFrameReader


class RealWorldRunner:
    """Executes PRISMPipeline frame-by-frame on real-world video files."""

    def __init__(self, config: RealWorldConfig | None = None) -> None:
        self.config: RealWorldConfig = config if config is not None else RealWorldConfig()
        self.detector = RealWorldDetector(
            detector_type=self.config.detector_type,
            confidence_thresh=self.config.detector_confidence_thresh,
        )
        self.projector = RealWorldGroundProjector(self.config.camera_config)

    def run_video(
        self,
        metadata: RealWorldRunMetadata,
        pipeline: PRISMPipeline | None = None,
    ) -> list[RealWorldFrameResult]:
        """Process an entire real-world video clip through detection, projection, and PRISM."""
        pipe = (
            pipeline
            if pipeline is not None
            else create_variant_pipeline(
                variant="FULL_PRISM",
                prediction_horizon=self.config.prediction_horizon,
            )
        )
        pipe.reset()

        reader = VideoFrameReader(metadata.video_path)
        frame_results: list[RealWorldFrameResult] = []

        # Nominal user walking speed
        v_ego_z = 1.10  # m/s forward
        ego_pos_z = 0.0

        for frame_idx, t_s, bgr_frame in reader.iter_frames():
            t0 = time.perf_counter()

            # 1. 2D Object Detection
            raw_boxes = self.detector.detect(bgr_frame)

            # 2. Monocular Ground Plane Projection
            projected_obs: list[PhysicalObservation] = []
            for det_idx, box in enumerate(raw_boxes):
                obs = self.projector.project_detection(box, timestamp=t_s, detection_idx=det_idx)
                if obs is not None:
                    projected_obs.append(obs)

            # 3. Kinematic Ego State
            ego_state = EgoMotionState(
                position=Vector2(0.0, ego_pos_z),
                velocity=Vector2(0.0, v_ego_z),
                timestamp=t_s,
            )
            ego_pos_z += v_ego_z * (1.0 / max(1.0, metadata.fps))

            # 4. Feed into PRISMPipeline
            runtime_res = pipe.process_frame(
                raw_frame=None,
                timestamp=t_s,
                ego_state=ego_state,
                explicit_observations=projected_obs,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0

            dec = runtime_res.decision
            risk = runtime_res.risk_assessment

            rec_heading_deg = (
                round(math.degrees(dec.recommended_heading), 2)
                if dec.recommended_heading is not None
                else None
            )

            # Extract active track positions for trajectory analysis
            track_positions = []
            for trk in pipe.tracker.active_tracks:
                st = pipe.state_estimator.get_state(trk.track_id)
                if st is not None:
                    track_positions.append({
                        "track_id": trk.track_id,
                        "x": round(st.position.x, 3),
                        "z": round(st.position.z, 3),
                        "vx": round(st.velocity.x, 3),
                        "vz": round(st.velocity.z, 3),
                        "hits": trk.hits,
                    })

            frame_results.append(
                RealWorldFrameResult(
                    timestamp=t_s,
                    frame_index=frame_idx,
                    raw_detections_count=len(raw_boxes),
                    projected_observations_count=len(projected_obs),
                    active_tracks_count=runtime_res.active_track_count,
                    decision_action=dec.action.name,
                    decision_reason=dec.reason.name,
                    risk_level=risk.risk_level.name if risk else "SAFE",
                    risk_score=float(risk.risk_score) if risk else 0.0,
                    predicted_ttc=float(risk.ttc) if (risk and risk.ttc is not None) else None,
                    predicted_cpa=float(risk.cpa_distance) if (risk and risk.cpa_distance is not None) else None,
                    primary_threat_id=dec.primary_threat_id,
                    recommended_heading_deg=rec_heading_deg,
                    latency_ms=lat_ms,
                    track_positions=track_positions,
                )
            )

        return frame_results
