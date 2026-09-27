"""PRISM Embedded Runtime Bridge for Android and cross-platform native hosts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import time
from typing import Any, Mapping, Sequence

from app.core.enums import RiskLevel
from app.core.types import BoundingBox, RiskAssessment, Vector2
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.perception.model import Detection, PhysicalObservation
from app.perception.pipeline import PerceptionPipeline
from app.presentation.engine import PresentationEngine
from app.presentation.model import (
    HapticPattern,
    PresentationChannel,
    PresentationConfig,
    PresentationInstruction,
    PresentationPriority,
)
from app.runtime.config import RuntimeConfig
from app.runtime.health import RuntimeHealth, RuntimeStatus
from app.runtime.model import RuntimeResult
from app.runtime.pipeline import PRISMPipeline


class EmbeddedRuntimeState(str, Enum):
    """Explicit lifecycle states for embedded PRISM runtime."""

    STOPPED = "STOPPED"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    STOPPING = "STOPPING"


@dataclass(frozen=True)
class EmbeddedBridgeConfig:
    """Configuration options for embedded Android/native bridge."""

    stale_frame_threshold_s: float = 0.25
    decision_validity_window_s: float = 0.50
    max_queue_depth: int = 2
    execution_mode: str = "LIVE"  # "LIVE", "MOCK", "REPLAY"
    enable_speech: bool = True
    enable_spatial_audio: bool = True
    enable_haptic: bool = True
    enable_visual: bool = True


class EmbeddedPrismBridge:
    """Platform-independent bridge exposing PRISM core to Android/native embedded hosts."""

    def __init__(
        self,
        pipeline: PRISMPipeline,
        presentation_engine: PresentationEngine | None = None,
        config: EmbeddedBridgeConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.presentation_engine = (
            presentation_engine if presentation_engine is not None else PresentationEngine()
        )
        self.config = config if config is not None else EmbeddedBridgeConfig()

        self._state = EmbeddedRuntimeState.STOPPED
        self._last_processed_timestamp: float | None = None
        self._dropped_frames_count: int = 0
        self._processed_frames_count: int = 0
        self._total_frame_queue_drops: int = 0
        self._degraded_reasons: list[str] = []

    @property
    def state(self) -> EmbeddedRuntimeState:
        return self._state

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames_count

    @property
    def processed_frames(self) -> int:
        return self._processed_frames_count

    def start(self) -> bool:
        """Transition lifecycle state to INITIALIZING then RUNNING."""
        self._state = EmbeddedRuntimeState.INITIALIZING
        self.pipeline.reset()
        self.presentation_engine.reset()
        self._last_processed_timestamp = None
        self._dropped_frames_count = 0
        self._processed_frames_count = 0
        self._total_frame_queue_drops = 0
        self._degraded_reasons.clear()
        self._state = EmbeddedRuntimeState.RUNNING
        return True

    def stop(self) -> bool:
        """Transition lifecycle state to STOPPING then STOPPED."""
        self._state = EmbeddedRuntimeState.STOPPING
        self.pipeline.reset()
        self.presentation_engine.reset()
        self._state = EmbeddedRuntimeState.STOPPED
        return True

    def process_observations_payload(
        self,
        observations: Sequence[PhysicalObservation],
        timestamp: float,
    ) -> dict[str, Any]:
        """Execute full PRISM pipeline and presentation engine on incoming observation batch.

        Args:
            observations: Sequence of pre-projected PhysicalObservations from perception backend.
            timestamp: Capture timestamp of the frame in seconds.

        Returns:
            Dictionary containing structured decision, risk, timing, cues, and health status.
        """
        if self._state not in (EmbeddedRuntimeState.RUNNING, EmbeddedRuntimeState.DEGRADED):
            return self._build_error_payload("Runtime is not running", timestamp)

        # Stale frame rejection / queue drop policy
        if self._last_processed_timestamp is not None:
            dt = timestamp - self._last_processed_timestamp
            if dt < 0.0:
                self._dropped_frames_count += 1
                self._degraded_reasons.append("Out of order timestamp")
                self._state = EmbeddedRuntimeState.DEGRADED
                return self._build_error_payload(f"Out of order timestamp: {timestamp} < {self._last_processed_timestamp}", timestamp)
            elif dt > self.config.stale_frame_threshold_s:
                self._degraded_reasons.append(f"Stale frame delta: {dt:.3f}s")
                self._state = EmbeddedRuntimeState.DEGRADED

        self._last_processed_timestamp = timestamp
        self._processed_frames_count += 1

        t_start = time.perf_counter()

        # 1. Execute PRISM Pipeline
        runtime_result: RuntimeResult = self.pipeline.process_frame(
            raw_frame=None,
            timestamp=timestamp,
            explicit_observations=observations,
        )

        # 2. Evaluate Multimodal Presentation Engine
        t_pres_start = time.perf_counter()
        presentation_cues: list[PresentationInstruction] = self.presentation_engine.process(
            decision=runtime_result.decision,
            risk_assessment=runtime_result.risk_assessment,
        )
        t_pres_end = time.perf_counter()
        pres_ms = (t_pres_end - t_pres_start) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Update degraded/running state based on pipeline health
        if runtime_result.runtime_status in (RuntimeStatus.DEGRADED, RuntimeStatus.FAILED):
            self._state = EmbeddedRuntimeState.DEGRADED
        elif self._state == EmbeddedRuntimeState.DEGRADED and len(self._degraded_reasons) == 0:
            self._state = EmbeddedRuntimeState.RUNNING

        # 3. Serialize Output Payload
        return self._serialize_payload(runtime_result, presentation_cues, pres_ms, total_ms)

    def process_json_payload(self, json_str: str) -> str:
        """Process a JSON-formatted request string from Android/JNI and return a JSON result string."""
        try:
            req = json.loads(json_str)
            timestamp = float(req.get("timestamp", time.time()))
            raw_obs = req.get("observations", [])

            observations: list[PhysicalObservation] = []
            for i, item in enumerate(raw_obs):
                x = float(item["x"])
                z = float(item["z"])
                cov = float(item.get("covariance", 0.05))
                conf = float(item.get("confidence", 1.0))
                cid = str(item.get("class_name", "obstacle"))
                det_id = item.get("detection_id", i + 1)
                observations.append(
                    PhysicalObservation(
                        position=Vector2(x, z),
                        measurement_covariance=((cov, 0.0), (0.0, cov)),
                        confidence=conf,
                        timestamp=timestamp,
                        detection_id=det_id,
                        class_name=cid,
                    )
                )

            res_dict = self.process_observations_payload(observations, timestamp)
            return json.dumps(res_dict)
        except Exception as ex:
            err_dict = self._build_error_payload(str(ex), time.time())
            return json.dumps(err_dict)

    def _serialize_payload(
        self,
        res: RuntimeResult,
        cues: list[PresentationInstruction],
        pres_ms: float,
        total_ms: float,
    ) -> dict[str, Any]:
        cues_data = []
        for c in cues:
            cues_data.append({
                "channel": c.channel.value,
                "priority": c.priority.value,
                "message_id": c.message_id,
                "text": c.text,
                "direction": c.direction,
                "intensity": c.intensity,
                "duration": c.duration,
                "haptic_pattern": c.haptic_pattern.value if c.haptic_pattern else None,
            })

        dec = res.decision
        risk = res.risk_assessment

        return {
            "frame_id": res.frame_id,
            "timestamp": res.timestamp,
            "runtime_state": self._state.value,
            "decision": {
                "action": dec.action.value,
                "reason": dec.reason.value,
                "confidence": dec.confidence,
                "primary_threat_id": dec.primary_threat_id,
                "recommended_heading": dec.recommended_heading,
                "risk_level": dec.risk_level.value,
            },
            "risk": {
                "object_id": risk.object_id if risk else None,
                "risk_level": risk.risk_level.value if risk else "safe",
                "risk_score": risk.risk_score if risk else 0.0,
                "ttc": risk.ttc if risk else None,
                "cpa_distance": risk.cpa_distance if risk else None,
            },
            "presentation_cues": cues_data,
            "timing": {
                "pipeline_ms": res.timing.total_ms if res.timing else 0.0,
                "presentation_ms": pres_ms,
                "total_embedded_ms": total_ms,
                "effective_fps": 1000.0 / max(0.001, total_ms),
            },
            "health": {
                "status": res.runtime_status.value if res.runtime_status else "healthy",
                "active_tracks": res.active_track_count,
                "dropped_frames": self._dropped_frames_count,
                "processed_frames": self._processed_frames_count,
            },
        }

    def _build_error_payload(self, message: str, timestamp: float) -> dict[str, Any]:
        return {
            "frame_id": -1,
            "timestamp": timestamp,
            "runtime_state": self._state.value,
            "decision": {
                "action": NavigationAction.EMERGENCY_STOP.value,
                "reason": DecisionReason.EMERGENCY_COLLISION_RISK.value,
                "confidence": 1.0,
                "primary_threat_id": None,
                "recommended_heading": None,
                "risk_level": RiskLevel.CRITICAL.value,
            },
            "risk": {
                "object_id": None,
                "risk_level": RiskLevel.CRITICAL.value,
                "risk_score": 100.0,
                "ttc": 0.0,
                "cpa_distance": 0.0,
            },
            "presentation_cues": [
                {
                    "channel": PresentationChannel.SPEECH.value,
                    "priority": PresentationPriority.EMERGENCY.value,
                    "message_id": "error_emergency_stop",
                    "text": "Stop",
                    "direction": None,
                    "intensity": 1.0,
                    "duration": 0.8,
                    "haptic_pattern": HapticPattern.EMERGENCY.value,
                }
            ],
            "timing": {
                "pipeline_ms": 0.0,
                "presentation_ms": 0.0,
                "total_embedded_ms": 0.0,
                "effective_fps": 0.0,
            },
            "health": {
                "status": "error",
                "active_tracks": 0,
                "dropped_frames": self._dropped_frames_count,
                "processed_frames": self._processed_frames_count,
                "error_message": message,
            },
        }
