package com.prism.assistant.telemetry

import android.content.Context
import android.util.Log
import com.prism.assistant.bridge.PrismFrameResult
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * High-performance non-blocking structured telemetry logger for Android.
 * Formats full per-frame detection, TTC, CPA, velocity, risk, and decision JSON payloads.
 */
class AndroidTelemetrySink(
    private val context: Context
) {
    private val memoryBuffer = ConcurrentLinkedQueue<String>()
    private var sessionFile: File? = null
    private var fileWriter: PrintWriter? = null
    private val listeners = CopyOnWriteArrayList<(String) -> Unit>()
    private var latestJsonStr: String = "{}"

    init {
        initSessionFile()
    }

    private fun initSessionFile() {
        try {
            val dateStr = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val dir = context.getExternalFilesDir("telemetry") ?: context.filesDir
            if (!dir.exists()) dir.mkdirs()

            sessionFile = File(dir, "prism_session_$dateStr.jsonl")
            fileWriter = PrintWriter(FileWriter(sessionFile!!, true))
            Log.i("PRISM_TELEMETRY", "Session telemetry recording to: ${sessionFile!!.absolutePath}")
        } catch (ex: Exception) {
            Log.e("PRISM_TELEMETRY", "Failed to initialize session file: ${ex.message}")
            sessionFile = null
            fileWriter = null
        }
    }

    fun addListener(listener: (String) -> Unit) {
        listeners.add(listener)
    }

    fun removeListener(listener: (String) -> Unit) {
        listeners.remove(listener)
    }

    fun getLatestJson(): String = latestJsonStr

    fun getSessionFile(): File? = sessionFile

    fun recordFrame(
        result: PrismFrameResult,
        orientation: com.prism.assistant.sensor.CameraOrientation? = null
    ) {
        val json = JSONObject()

        // 1. Core Header & Timestamp
        json.put("frame_id", result.frameId)
        json.put("timestamp", result.timestamp)
        json.put("runtime_state", result.runtimeState)

        // 2. All Active Detected Entities
        val entitiesArray = JSONArray()
        for (trk in result.trackList) {
            val dist = sqrt(trk.x * trk.x + trk.z * trk.z)
            val ttc = if (trk.vz < -0.1) {
                kotlin.math.max(0.1, -trk.z / trk.vz)
            } else {
                kotlin.math.max(0.5, trk.z / 1.0)
            }
            val riskScore = kotlin.math.min(
                100.0,
                (kotlin.math.max(0.0, 1.0 - (kotlin.math.min(dist, 4.5) / 4.5)) * 60.0 +
                 kotlin.math.max(0.0, 1.0 - (kotlin.math.min(abs(trk.x), 1.5) / 1.5)) * 30.0) *
                 (if (trk.vz < -0.1) 1.3 else 1.0)
            )

            val entityObj = JSONObject().apply {
                put("id", trk.trackId)
                put("x", String.format(Locale.US, "%.3f", trk.x).toDouble())
                put("z", String.format(Locale.US, "%.3f", trk.z).toDouble())
                put("vx", String.format(Locale.US, "%.3f", trk.vx).toDouble())
                put("vz", String.format(Locale.US, "%.3f", trk.vz).toDouble())
                put("class", trk.className)
                put("distance_m", String.format(Locale.US, "%.2f", dist).toDouble())
                put("ttc_s", String.format(Locale.US, "%.2f", ttc).toDouble())
                put("risk_score", String.format(Locale.US, "%.1f", riskScore).toDouble())

                // Telemetry Observability
                put("status", trk.status)
                put("hits", trk.hits)
                put("consecutive_hits", trk.consecutiveHits)
                put("missed_frames", trk.missedFrames)
                put("age", trk.age)
                put("geometric_confidence", String.format(Locale.US, "%.2f", trk.geometricConfidence).toDouble())
                put("temporal_confidence", String.format(Locale.US, "%.2f", trk.temporalConfidence).toDouble())
                put("physical_confidence", String.format(Locale.US, "%.2f", trk.physicalConfidence).toDouble())
                put("width_m", String.format(Locale.US, "%.2f", trk.widthMeters).toDouble())
                put("height_m", String.format(Locale.US, "%.2f", trk.heightMeters).toDouble())
                put("aspect_ratio", String.format(Locale.US, "%.2f", trk.aspectRatio).toDouble())
                put("cluster_size", trk.clusterSize)
                put("geometry_valid", trk.geometryValid)
                put("geometry_score", String.format(Locale.US, "%.2f", trk.geometryScore).toDouble())
                trk.rejectReason?.let { put("geometry_reject_reason", it) }
                put("position_consistency", String.format(Locale.US, "%.2f", trk.positionConsistency).toDouble())
                put("depth_consistency", String.format(Locale.US, "%.2f", trk.depthConsistency).toDouble())
                put("height_consistency", String.format(Locale.US, "%.2f", trk.heightConsistency).toDouble())
                put("width_consistency", String.format(Locale.US, "%.2f", trk.widthConsistency).toDouble())
                put("aspect_ratio_consistency", String.format(Locale.US, "%.2f", trk.aspectRatioConsistency).toDouble())
                put("ground_contact_consistency", String.format(Locale.US, "%.2f", trk.groundContactConsistency).toDouble())
                put("confidence_update_reason", trk.confidenceUpdateReason)
                put("shape_class", trk.shapeClass)
                put("fill_ratio", String.format(Locale.US, "%.2f", trk.fillRatio).toDouble())
                put("floor_structure_likelihood", String.format(Locale.US, "%.2f", trk.floorStructureLikelihood).toDouble())
                put("ground_plane_coherence", String.format(Locale.US, "%.2f", trk.groundPlaneCoherence).toDouble())
                put("time_to_cpa", String.format(Locale.US, "%.2f", trk.timeToCpa).toDouble())
                put("cpa_distance", String.format(Locale.US, "%.2f", trk.cpaDistance).toDouble())
                put("collision_threat_confidence", String.format(Locale.US, "%.2f", trk.collisionThreatConfidence).toDouble())
                put("hazard_state", trk.hazardState)
                put("user_response_expected", trk.userResponseExpected)
                put("user_motion_state", trk.userMotionState)
                put("motion_confidence", String.format(Locale.US, "%.2f", trk.motionConfidence).toDouble())
                put("step_count", trk.stepCount)
                put("time_since_last_step", String.format(Locale.US, "%.2f", trk.timeSinceLastStep).toDouble())
                put("motion_energy", String.format(Locale.US, "%.3f", trk.motionEnergy).toDouble())
            }
            entitiesArray.put(entityObj)
        }
        json.put("entities", entitiesArray)

        // 3. Primary Threat
        val threatObj = JSONObject().apply {
            put("id", result.risk.objectId)
            put("risk_level", result.risk.riskLevel)
            put("risk_score", String.format(Locale.US, "%.1f", result.risk.riskScore).toDouble())
            put("ttc_s", result.risk.ttc?.let { String.format(Locale.US, "%.2f", it).toDouble() })
            put("cpa_m", result.risk.cpaDistance?.let { String.format(Locale.US, "%.2f", it).toDouble() })
        }
        json.put("primary_threat", threatObj)

        // 4. Navigation Decision
        val decObj = JSONObject().apply {
            put("action", result.decision.action)
            put("reason", result.decision.reason)
            put("confidence", result.decision.confidence)
            put("recommended_heading_rad", result.decision.recommendedHeading?.let { String.format(Locale.US, "%.3f", it).toDouble() })
            put("recommended_heading_deg", result.decision.recommendedHeading?.let { String.format(Locale.US, "%.1f", it * 180.0 / Math.PI).toDouble() })
        }
        json.put("decision", decObj)

        // 5. Multimodal Presentation Cues
        val cuesArray = JSONArray()
        for (cue in result.cues) {
            cuesArray.put(JSONObject().apply {
                put("channel", cue.channel)
                put("priority", cue.priority)
                put("message_id", cue.messageId)
                put("text", cue.text)
                put("direction", cue.direction)
                put("intensity", cue.intensity)
                put("duration", cue.duration)
                put("haptic_pattern", cue.hapticPattern)
                if (cue.spatialCue != null) {
                    put("spatial_pan", cue.spatialCue.pan)
                    put("spatial_cue_type", cue.spatialCue.cueType.name)
                    put("spatial_sector", cue.spatialCue.direction.name)
                }
            })
        }
        json.put("presentation_cues", cuesArray)

        // 6. Timing & Performance
        val timingObj = JSONObject().apply {
            put("pipeline_ms", String.format(Locale.US, "%.2f", result.timing.pipelineMs).toDouble())
            put("presentation_ms", String.format(Locale.US, "%.2f", result.timing.presentationMs).toDouble())
            put("total_ms", String.format(Locale.US, "%.2f", result.timing.totalEmbeddedMs).toDouble())
            put("fps", String.format(Locale.US, "%.1f", result.timing.effectiveFps).toDouble())
        }
        json.put("timing", timingObj)

        // 7. System Health
        val healthObj = JSONObject().apply {
            put("active_tracks", result.activeTracks)
            put("dropped_frames", result.droppedFrames)
            put("processed_frames", result.processedFrames)
        }
        json.put("health", healthObj)

        // 8. Dynamic Camera Orientation & IMU Fusion Telemetry (Phase G)
        val orientationObj = JSONObject().apply {
            put("pitch_deg", String.format(Locale.US, "%.2f", (orientation?.pitchRadians ?: 0.20944) * 180.0 / Math.PI).toDouble())
            put("roll_deg", String.format(Locale.US, "%.2f", (orientation?.rollRadians ?: 0.0) * 180.0 / Math.PI).toDouble())
            put("pitch_rad", String.format(Locale.US, "%.4f", orientation?.pitchRadians ?: 0.20944).toDouble())
            put("roll_rad", String.format(Locale.US, "%.4f", orientation?.rollRadians ?: 0.0).toDouble())
            put("confidence", String.format(Locale.US, "%.2f", orientation?.confidence ?: 1.0).toDouble())
            put("source_state", orientation?.sourceState?.name ?: "FALLBACK_CALIBRATED")
            put("gravity_magnitude", String.format(Locale.US, "%.2f", orientation?.gravityMagnitude ?: 9.81).toDouble())
            put("imu_timestamp_ns", orientation?.timestampNs ?: 0L)
        }
        json.put("orientation", orientationObj)

        // 9. Dynamic IMU Ego-Motion & Step Telemetry (Phase J)
        result.userMotion?.let {
            val motionObj = JSONObject().apply {
                put("user_motion_state", it.state.name)
                put("motion_confidence", String.format(Locale.US, "%.2f", it.confidence).toDouble())
                put("step_count", it.stepCount)
                put("time_since_last_step", String.format(Locale.US, "%.2f", it.timeSinceLastStepSeconds).toDouble())
                put("motion_energy", String.format(Locale.US, "%.3f", it.motionEnergy).toDouble())
                put("cadence_spm", String.format(Locale.US, "%.1f", it.cadenceStepsPerMinute).toDouble())
                put("sensor_timestamp", it.timestampNs)
            }
            json.put("user_motion", motionObj)
        }

        val jsonStr = json.toString()
        latestJsonStr = jsonStr
        memoryBuffer.add(jsonStr)

        // Logcat broadcast for ADB live streaming tools
        Log.i("PRISM_TELEMETRY", jsonStr)

        // Notify active socket/HTTP subscribers
        for (listener in listeners) {
            try {
                listener(jsonStr)
            } catch (ex: Exception) {
                // Ignore subscriber dropped connection
            }
        }

        // Flush to session file
        fileWriter?.let { writer ->
            try {
                writer.println(jsonStr)
                writer.flush()
            } catch (ex: Exception) {
                // Keep in memory buffer
            }
        }
    }

    fun getBufferedLines(): List<String> = memoryBuffer.toList()

    fun getBufferedJsonArrayString(): String {
        val lines = memoryBuffer.toList()
        return if (lines.isEmpty()) "[]" else "[" + lines.joinToString(",") + "]"
    }

    fun clear() {
        memoryBuffer.clear()
    }

    fun close() {
        try {
            fileWriter?.flush()
            fileWriter?.close()
        } catch (ex: Exception) {
            // Ignore close error
        }
        fileWriter = null
    }
}
