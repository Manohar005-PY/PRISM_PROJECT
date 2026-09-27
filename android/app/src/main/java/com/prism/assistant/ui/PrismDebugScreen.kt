package com.prism.assistant.ui

import android.graphics.Color
import android.widget.TextView
import com.prism.assistant.bridge.PrismFrameResult
import com.prism.assistant.camera.CameraFrameMeta
import com.prism.assistant.runtime.RuntimeState
import java.util.Locale

/**
 * Controller binding PRISM telemetry, camera diagnostics, and multi-entity state to the Android debug HUD.
 */
class PrismDebugScreen(
    private val tvCameraStatus: TextView,
    private val tvState: TextView,
    private val tvFpsLatency: TextView,
    private val tvDecision: TextView,
    private val tvRisk: TextView,
    private val tvObstacleDetails: TextView,
    private val tvServerUrl: TextView
) {
    fun updateState(state: RuntimeState) {
        tvState.text = "STATE: ${state.name}"
        when (state) {
            RuntimeState.RUNNING -> tvState.setTextColor(Color.parseColor("#00E676"))
            RuntimeState.DEGRADED -> tvState.setTextColor(Color.parseColor("#FF9100"))
            RuntimeState.ERROR -> tvState.setTextColor(Color.parseColor("#FF1744"))
            else -> tvState.setTextColor(Color.WHITE)
        }
    }

    fun updateServerUrl(ipAddress: String, port: Int) {
        tvServerUrl.text = "LAPTOP VIEW: http://$ipAddress:$port (or localhost:$port via adb)"
    }

    fun updateCameraDiagnostics(meta: CameraFrameMeta) {
        tvCameraStatus.text = String.format(
            "CAMERA: CONNECTED (%dx%d @ %.1f FPS) | FRAME: %d",
            meta.width,
            meta.height,
            meta.captureFps,
            meta.frameNumber
        )
    }

    fun updateResult(result: PrismFrameResult, isLive: Boolean = true) {
        val timing = result.timing
        val dec = result.decision
        val risk = result.risk

        tvFpsLatency.text = String.format(
            "FPS: %.1f | Pipe: %.2f ms | Total: %.2f ms | Drops: %d",
            timing.effectiveFps,
            timing.pipelineMs,
            timing.totalEmbeddedMs,
            result.droppedFrames
        )

        val headingStr = if (dec.recommendedHeading != null) {
            String.format(" (heading: %+.1f°)", dec.recommendedHeading * 180.0 / Math.PI)
        } else ""

        val threatStr = if (dec.primaryThreatId != null) " [Threat #ID${dec.primaryThreatId}]" else ""
        tvDecision.text = "ACTION: ${dec.action.uppercase()}$threatStr$headingStr"
        when (dec.action.lowercase()) {
            "continue" -> tvDecision.setTextColor(Color.parseColor("#00E676"))
            "caution", "slow_down" -> tvDecision.setTextColor(Color.parseColor("#FFD600"))
            "turn_left", "turn_right" -> tvDecision.setTextColor(Color.parseColor("#FF9100"))
            "stop", "emergency_stop" -> tvDecision.setTextColor(Color.parseColor("#FF1744"))
            else -> tvDecision.setTextColor(Color.WHITE)
        }

        val ttcStr = if (risk.ttc != null) String.format("%.1fs", risk.ttc) else "N/A"
        val cpaStr = if (risk.cpaDistance != null) String.format("%.2fm", risk.cpaDistance) else "N/A"

        tvRisk.text = String.format(
            "RISK: %s (Score: %.1f | TTC: %s | CPA: %s)",
            risk.riskLevel.uppercase(),
            risk.riskScore,
            ttcStr,
            cpaStr
        )

        val trackListStr = if (result.trackList.isNotEmpty()) {
            result.trackList.take(3).joinToString(" | ") {
                String.format("#%d: (X:%+.1fm, Z:%.1fm)", it.trackId, it.x, it.z)
            }
        } else {
            "Corridor Clear"
        }

        val primaryTrackInfo = result.trackList.find { it.trackId == dec.primaryThreatId } ?: result.trackList.firstOrNull()
        val motion = result.userMotion
        val motionStr = if (motion != null && motion.state.name != "UNKNOWN") {
            String.format(
                Locale.US,
                " | USER: %s (CONF: %.2f, STEPS: %d, LAST: %.2fs)",
                motion.state.name,
                motion.confidence,
                motion.stepCount,
                motion.timeSinceLastStepSeconds
            )
        } else ""

        val hazardStatusStr = if (primaryTrackInfo != null && primaryTrackInfo.hazardState != "NONE") {
            String.format(
                " | HAZARD: %s (%s)",
                primaryTrackInfo.hazardState,
                primaryTrackInfo.userMotionState
            )
        } else ""

        val spatialCue = result.cues.find { it.channel == "spatial_audio" }?.spatialCue
        val spatialStatusStr = if (spatialCue != null) {
            String.format(
                Locale.US,
                " | SPATIAL: %s (%s, pan: %.2f)",
                spatialCue.cueType.name,
                spatialCue.direction.name,
                spatialCue.pan
            )
        } else ""

        tvObstacleDetails.text = String.format(
            "ENTITIES (%d): %s%s%s%s",
            result.activeTracks,
            trackListStr,
            hazardStatusStr,
            motionStr,
            spatialStatusStr
        )
    }
}
