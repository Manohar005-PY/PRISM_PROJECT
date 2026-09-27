package com.prism.assistant.bridge

import com.prism.assistant.accessibility.AudioPriority
import com.prism.assistant.sensor.CameraOrientation
import com.prism.assistant.sensor.UserMotionEstimate
import com.prism.assistant.sensor.UserMotionInferenceMode
import com.prism.assistant.sensor.UserMotionState
import com.prism.assistant.spatial.MultisensoryMode
import com.prism.assistant.spatial.SpatialCue
import com.prism.assistant.spatial.SpatialCueType
import com.prism.assistant.warning.DistanceBand
import com.prism.assistant.warning.HazardLifecycleState
import com.prism.assistant.warning.MovementState
import com.prism.assistant.warning.SpatialDirection
import com.prism.assistant.warning.WarningContext
import com.prism.assistant.warning.WarningMessageGenerator
import com.prism.assistant.warning.WarningSeverity
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Geometric and spatial feature properties of an extracted visual candidate.
 */
data class CandidateGeometry(
    val widthMeters: Double = 0.0,
    val heightMeters: Double = 0.0,
    val aspectRatio: Double = 0.0,
    val groundContactV: Double = 0.0,
    val topV: Double = 0.0,
    val leftCol: Int = 0,
    val rightCol: Int = 0,
    val topRow: Int = 0,
    val bottomRow: Int = 0,
    val gradientEnergy: Double = 0.0,
    val localContrast: Double = 0.0,
    val clusterSize: Int = 0,
    val geometryValid: Boolean = true,
    val geometryScore: Double = 1.0,
    val rejectReason: String? = null,
    val shapeClass: String = "UNCERTAIN",
    val fillRatio: Double = 1.0,
    val floorStructureLikelihood: Double = 0.0,
    val groundPlaneCoherence: Double = 1.0
)

/**
 * Clean data transfer objects bridging platform-independent PRISM core to Android.
 */
data class PrismObservation(
    val x: Double,
    val z: Double,
    val covariance: Double = 0.05,
    val confidence: Double = 1.0,
    val className: String = "obstacle",
    val geometry: CandidateGeometry? = null,
    val geometricConfidence: Double = 1.0
)

data class PrismDecision(
    val action: String,
    val reason: String,
    val confidence: Double,
    val primaryThreatId: Int?,
    val recommendedHeading: Double?,
    val riskLevel: String
)

data class PrismRisk(
    val objectId: Int?,
    val riskLevel: String,
    val riskScore: Double,
    val ttc: Double?,
    val cpaDistance: Double?
)

data class PrismPresentationCue(
    val channel: String,
    val priority: String,
    val messageId: String,
    val text: String?,
    val direction: Double?,
    val intensity: Double,
    val duration: Double,
    val hapticPattern: String?,
    val spatialCue: SpatialCue? = null
)

data class PrismTiming(
    val pipelineMs: Double,
    val presentationMs: Double,
    val totalEmbeddedMs: Double,
    val effectiveFps: Double
)

/**
 * Historical snapshot of a single matched visual observation.
 */
data class TrackObservationSnapshot(
    val timestamp: Double,
    val x: Double,
    val z: Double,
    val heightMeters: Double,
    val widthMeters: Double,
    val aspectRatio: Double,
    val groundContactV: Double
)

data class TrackInfo(
    val trackId: Int,
    val x: Double,
    val z: Double,
    val vx: Double,
    val vz: Double,
    val className: String,
    val status: String = "CONFIRMED",
    val hits: Int = 1,
    val consecutiveHits: Int = 1,
    val missedFrames: Int = 0,
    val age: Int = 1,
    val geometricConfidence: Double = 1.0,
    val temporalConfidence: Double = 1.0,
    val physicalConfidence: Double = 1.0,
    val widthMeters: Double = 0.0,
    val heightMeters: Double = 0.0,
    val aspectRatio: Double = 0.0,
    val clusterSize: Int = 0,
    val geometryValid: Boolean = true,
    val geometryScore: Double = 1.0,
    val rejectReason: String? = null,
    val positionConsistency: Double = 1.0,
    val depthConsistency: Double = 1.0,
    val heightConsistency: Double = 1.0,
    val widthConsistency: Double = 1.0,
    val aspectRatioConsistency: Double = 1.0,
    val groundContactConsistency: Double = 1.0,
    val confidenceUpdateReason: String = "INITIAL_CREATION",
    val shapeClass: String = "UNCERTAIN",
    val fillRatio: Double = 1.0,
    val floorStructureLikelihood: Double = 0.0,
    val groundPlaneCoherence: Double = 1.0,
    val warningState: String = "NONE",
    val timeToCpa: Double = 99.0,
    val cpaDistance: Double = 99.0,
    val collisionThreatConfidence: Double = 1.0,
    val hazardState: String = "NONE",
    val hazardEncounterTimestamp: Double = 0.0,
    val hazardResolvedTimestamp: Double? = null,
    val userResponseExpected: Boolean = false,
    val hazardPersistenceDuration: Double = 0.0,
    val initialHazardDistance: Double = 0.0,
    val userMotionState: String = "MOVING",
    val motionConfidence: Double = 0.0,
    val stepCount: Int = 0,
    val timeSinceLastStep: Double = 99.0,
    val motionEnergy: Double = 0.0
)

data class PrismFrameResult(
    val frameId: Long,
    val timestamp: Double,
    val runtimeState: String,
    val decision: PrismDecision,
    val risk: PrismRisk,
    val cues: List<PrismPresentationCue>,
    val timing: PrismTiming,
    val activeTracks: Int,
    val trackList: List<TrackInfo> = emptyList(),
    val droppedFrames: Long,
    val processedFrames: Long,
    val errorMessage: String? = null,
    val userMotion: UserMotionEstimate? = null
)

/**
 * Interface implemented by PRISM embedded backends (e.g. Python Chaquopy / ONNX / Native JNI / Mock).
 */
interface IPrismCoreBridge {
    fun initialize(): Boolean
    fun processFrame(
        timestamp: Double,
        observations: List<PrismObservation>,
        motionEstimate: UserMotionEstimate? = null
    ): PrismFrameResult
    fun reset()
}

/**
 * Track Status enumeration mirroring Python PRISM TrackStatus.
 */
enum class TrackStatus {
    TENTATIVE,
    CONFIRMED,
    LOST,
    DELETED
}

/**
 * Warning severity state per track for deduplication and hysteresis.
 */
enum class TrackWarningState {
    NONE,
    ADVISORY,
    WARNING,
    EMERGENCY
}

/**
 * Persistent Track Record for multi-object state estimation with lightweight temporal validation.
 */
data class AndroidTrackRecord(
    val trackId: Int,
    var x: Double,
    var z: Double,
    var vx: Double = 0.0,
    var vz: Double = 0.0,
    var lastTimestamp: Double,
    var hits: Int = 1,
    var consecutiveHits: Int = 1,
    var missedFrames: Int = 0,
    var status: TrackStatus = TrackStatus.TENTATIVE,
    var className: String = "obstacle",
    var age: Int = 1,
    var geometricConfidence: Double = 1.0,
    var temporalConfidence: Double = 1.0,
    var physicalConfidence: Double = 1.0,
    var lastGeometry: CandidateGeometry? = null,
    val history: ArrayDeque<TrackObservationSnapshot> = ArrayDeque(),
    var positionConsistency: Double = 1.0,
    var depthConsistency: Double = 1.0,
    var heightConsistency: Double = 1.0,
    var widthConsistency: Double = 1.0,
    var aspectRatioConsistency: Double = 1.0,
    var groundContactConsistency: Double = 1.0,
    var confidenceUpdateReason: String = "INITIAL_CREATION",
    var warningState: String = "NONE",
    var lastWarningTimestamp: Double = 0.0,
    var lastWarnedAction: String = "NONE",
    var lastWarnedSeverity: String = "CLEAR",
    var lastWarnedDirection: String = "NONE",
    var lastWarnedDistanceBand: String = "NONE",
    var lastWarnedMovement: MovementState = MovementState.STATIONARY,
    var timeToCpa: Double = 99.0,
    var cpaDistance: Double = 99.0,
    var collisionThreatConfidence: Double = 1.0,
    var hazardState: HazardLifecycleState = HazardLifecycleState.NONE,
    var hazardEncounterTimestamp: Double = 0.0,
    var hazardResolvedTimestamp: Double? = null,
    var userResponseExpected: Boolean = false,
    var initialHazardDistance: Double = 0.0,
    var holdDistance: Double = 0.0,
    var userMotionState: UserMotionState = UserMotionState.MOVING
) {
    fun addSnapshot(obs: PrismObservation, timestamp: Double) {
        val geom = obs.geometry
        if (history.size >= 6) {
            history.removeFirst()
        }
        history.addLast(
            TrackObservationSnapshot(
                timestamp = timestamp,
                x = obs.x,
                z = obs.z,
                heightMeters = geom?.heightMeters ?: 1.0,
                widthMeters = geom?.widthMeters ?: 0.5,
                aspectRatio = geom?.aspectRatio ?: 1.0,
                groundContactV = geom?.groundContactV ?: 300.0
            )
        )
    }
}

/**
 * Autonomous Multi-Target Tracking and Predictive Decision Bridge.
 * Tracks multiple entities simultaneously, updates velocity via Kalman filtering,
 * evaluates TTC/CPA per track, and generates spatial evasion decisions.
 */
class DeterministicPrismBridge(
    private val confirmationHits: Int = 1
) : IPrismCoreBridge {
    private var frameCounter: Long = 0
    private var lastTimestamp: Double? = null
    private var nextTrackId: Int = 1
    private val tracks = mutableMapOf<Int, AndroidTrackRecord>()
    private var lastDecisionAction: String = "continue"
    private var preferredTurnDirection: String? = null
    private var lastActionChangeTimestamp: Double = 0.0
    private var lastPrimaryThreatId: Int? = null
    private var hadActiveHazardLastFrame: Boolean = false

    var userMotionMode: UserMotionInferenceMode = UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED
    var latestUserMotionEstimate: UserMotionEstimate = UserMotionEstimate()
    var multisensoryMode: MultisensoryMode = MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC

    override fun initialize(): Boolean {
        reset()
        return true
    }

    override fun reset() {
        frameCounter = 0
        lastTimestamp = null
        nextTrackId = 1
        tracks.clear()
        lastDecisionAction = "continue"
        preferredTurnDirection = null
        lastActionChangeTimestamp = 0.0
        lastPrimaryThreatId = null
        hadActiveHazardLastFrame = false
        latestUserMotionEstimate = UserMotionEstimate()
        multisensoryMode = MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC
    }

    fun updateMotionEstimate(estimate: UserMotionEstimate) {
        latestUserMotionEstimate = estimate
    }

    fun updateMotionMode(mode: UserMotionInferenceMode) {
        userMotionMode = mode
    }

    fun updateMultisensoryMode(mode: MultisensoryMode) {
        multisensoryMode = mode
    }

    override fun processFrame(
        timestamp: Double,
        observations: List<PrismObservation>,
        motionEstimate: UserMotionEstimate?
    ): PrismFrameResult {
        if (motionEstimate != null) {
            latestUserMotionEstimate = motionEstimate
        }
        val t0 = System.nanoTime()
        frameCounter++
        val dt = if (lastTimestamp != null) max(0.001, timestamp - lastTimestamp!!) else 0.033
        lastTimestamp = timestamp

        val imuReliable = (userMotionMode == UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED) &&
                (latestUserMotionEstimate.state != UserMotionState.UNKNOWN) &&
                (latestUserMotionEstimate.confidence >= 0.50)

        // 1. Multi-Target Data Association & Tracking (Stage 4)
        val matchedObs = mutableSetOf<Int>()
        val matchedTracks = mutableSetOf<Int>()

        // Velocity-aware distance gating & association (greedy matching with speed-adaptive gate)
        for ((obsIdx, obs) in observations.withIndex()) {
            var bestTrackId: Int? = null
            var bestDist = Double.MAX_VALUE

            for ((tId, track) in tracks) {
                if (tId in matchedTracks || track.status == TrackStatus.DELETED) continue
                // Predicted position based on current velocity
                val predX = track.x + track.vx * dt
                val predZ = track.z + track.vz * dt
                val dist = sqrt((obs.x - predX) * (obs.x - predX) + (obs.z - predZ) * (obs.z - predZ))
                val trackSpeed = sqrt(track.vx * track.vx + track.vz * track.vz)
                val maxGateDist = max(0.90, min(1.80, 1.00 + trackSpeed * dt * 2.0))

                if (dist < maxGateDist && dist < bestDist) {
                    bestDist = dist
                    bestTrackId = tId
                }
            }

            if (bestTrackId != null) {
                val trk = tracks[bestTrackId]!!
                // Physical velocity sanity check & filtering (Stage 4)
                val maxAllowedVx = 3.5 // max realistic lateral speed (m/s)
                val maxAllowedVz = 4.5 // max realistic closing speed (m/s)
                val rawVxUnclamped = (obs.x - trk.x) / dt
                val rawVzUnclamped = (obs.z - trk.z) / dt
                val rawVx = if (rawVxUnclamped.isNaN() || rawVxUnclamped.isInfinite()) 0.0 else rawVxUnclamped.coerceIn(-maxAllowedVx, maxAllowedVx)
                val rawVz = if (rawVzUnclamped.isNaN() || rawVzUnclamped.isInfinite()) 0.0 else rawVzUnclamped.coerceIn(-maxAllowedVz, maxAllowedVz)

                val alpha = 0.50 // Filter smoothing factor
                trk.vx = alpha * rawVx + (1 - alpha) * trk.vx
                trk.vz = alpha * rawVz + (1 - alpha) * trk.vz

                trk.hits++
                trk.consecutiveHits++
                trk.age++
                trk.missedFrames = 0

                updateTrackConsistency(trk, obs, dt, timestamp)

                trk.x = if (obs.x.isNaN() || obs.x.isInfinite()) trk.x else obs.x
                trk.z = if (obs.z.isNaN() || obs.z.isInfinite()) trk.z else obs.z
                trk.lastTimestamp = timestamp
                trk.className = obs.className
                trk.lastGeometry = obs.geometry

                // User motion inference
                if (imuReliable) {
                    val imuState = latestUserMotionEstimate.state
                    if (trk.hazardState in listOf(HazardLifecycleState.EMERGENCY, HazardLifecycleState.USER_RESPONDING, HazardLifecycleState.HAZARD_PERSISTING)) {
                        if (imuState.isStationary) {
                            trk.userMotionState = UserMotionState.STATIONARY
                        } else if (imuState.isMoving) {
                            trk.userMotionState = imuState
                        } else {
                            trk.userMotionState = UserMotionState.MOVING
                        }
                    } else {
                        trk.userMotionState = if (imuState.isStationary) UserMotionState.STATIONARY else imuState
                    }
                } else {
                    // Fallback / Relative-only mode (Phase I behavior preserved)
                    if (trk.hazardState in listOf(HazardLifecycleState.EMERGENCY, HazardLifecycleState.USER_RESPONDING, HazardLifecycleState.HAZARD_PERSISTING)) {
                        if (trk.vz > 0.18) {
                            trk.userMotionState = UserMotionState.BACKING_AWAY
                        } else if (abs(trk.vz) < 0.18) {
                            trk.userMotionState = UserMotionState.STOPPED
                        } else {
                            trk.userMotionState = UserMotionState.MOVING
                        }
                    } else {
                        trk.userMotionState = if (abs(trk.vz) < 0.10 && trk.z > 2.0) UserMotionState.STOPPED else UserMotionState.MOVING
                    }
                }

                // Check lifecycle transitions for persisting or responding hazards
                if (trk.hazardState == HazardLifecycleState.EMERGENCY) {
                    val userStopped = if (imuReliable) {
                        latestUserMotionEstimate.state.isStationary
                    } else {
                        trk.userMotionState == UserMotionState.STOPPED || abs(trk.vz) < 0.18
                    }
                    if (userStopped) {
                        trk.hazardState = HazardLifecycleState.HAZARD_PERSISTING
                        trk.holdDistance = trk.z
                        trk.userResponseExpected = false
                    }
                } else if (trk.hazardState == HazardLifecycleState.HAZARD_PERSISTING) {
                    // 1. Check for hazard resolution (user stepped back, obstacle moved away or left corridor)
                    val distIncreased = trk.z >= 1.40 || (trk.initialHazardDistance > 0.0 && trk.z >= trk.initialHazardDistance + 0.50)
                    val obstacleMovingAway = trk.vz > 0.20 && trk.z > 1.20
                    val leftCorridor = abs(trk.x) > 1.25

                    if (distIncreased || obstacleMovingAway || leftCorridor) {
                        trk.hazardState = HazardLifecycleState.HAZARD_CLEARED
                        trk.hazardResolvedTimestamp = timestamp
                    } else {
                        val userMovingTowardHazard = if (imuReliable) {
                            latestUserMotionEstimate.state.isMoving && (trk.vz < -0.15 || trk.z < (trk.holdDistance - 0.10))
                        } else {
                            (trk.userMotionState == UserMotionState.MOVING || trk.userMotionState == UserMotionState.WALKING) && (trk.vz < -0.15 || trk.z < (trk.holdDistance - 0.10))
                        }
                        if (userMovingTowardHazard) {
                            // 2. Re-escalate if user resumes walking toward persistent obstacle
                            trk.hazardState = HazardLifecycleState.EMERGENCY
                            trk.userResponseExpected = true
                            trk.holdDistance = trk.z
                            trk.lastWarnedSeverity = "HOLD"
                            trk.lastWarningTimestamp = 0.0
                        }
                    }
                }

                matchedTracks.add(bestTrackId)
                matchedObs.add(obsIdx)
            }
        }

        // 2. Age existing unmatched tracks & prune stale tracks
        val toRemove = mutableListOf<Int>()
        for ((tId, track) in tracks) {
            if (tId !in matchedTracks) {
                track.missedFrames++
                track.age++
                track.consecutiveHits = 0
                track.temporalConfidence = (track.temporalConfidence * 0.80).coerceIn(0.0, 1.0)
                val safeGeom = if (track.geometricConfidence.isNaN()) 0.0 else track.geometricConfidence
                val rawPhys = 0.45 * safeGeom + 0.55 * track.temporalConfidence
                track.physicalConfidence = if (rawPhys.isNaN() || rawPhys.isInfinite()) 0.0 else rawPhys.coerceIn(0.0, 1.0)
                track.confidenceUpdateReason = "MISSED_FRAME_DECAY"

                if (track.status == TrackStatus.TENTATIVE && track.missedFrames >= 1) {
                    track.status = TrackStatus.DELETED
                    toRemove.add(tId)
                } else if (track.missedFrames > 5 || track.physicalConfidence < 0.15) {
                    track.status = TrackStatus.DELETED
                    toRemove.add(tId)
                } else {
                    track.status = TrackStatus.LOST
                }
            }
        }
        toRemove.forEach { tracks.remove(it) }

        // 3. Spawn new tracks for unmatched observations
        for ((obsIdx, obs) in observations.withIndex()) {
            if (obsIdx !in matchedObs) {
                val newId = nextTrackId++
                val isCloseHazard = (!obs.z.isNaN() && obs.z < 1.0)
                val rawGeom = obs.geometricConfidence
                val safeGeom = if (rawGeom.isNaN() || rawGeom.isInfinite()) 0.0 else rawGeom.coerceIn(0.0, 1.0)
                val isGeometricallyValid = (safeGeom >= 0.50 && (obs.geometry?.heightMeters ?: 1.0) >= 0.22)
                val initialStatus = if (confirmationHits <= 1 || (isCloseHazard && isGeometricallyValid)) {
                    TrackStatus.CONFIRMED
                } else {
                    TrackStatus.TENTATIVE
                }

                val initTemporal = if (isCloseHazard && isGeometricallyValid) 0.85 else 0.50
                val rawPhys = 0.45 * safeGeom + 0.55 * initTemporal
                val initPhysical = if (rawPhys.isNaN() || rawPhys.isInfinite()) 0.0 else rawPhys.coerceIn(0.0, 1.0)
                val initReason = if (isCloseHazard && isGeometricallyValid) "CLOSE_HAZARD_INIT" else "NEW_TRACK_INIT"
                val safeX = if (obs.x.isNaN() || obs.x.isInfinite()) 0.0 else obs.x
                val safeZ = if (obs.z.isNaN() || obs.z.isInfinite()) 1.0 else obs.z

                val newTrack = AndroidTrackRecord(
                    trackId = newId,
                    x = safeX,
                    z = safeZ,
                    vx = 0.0,
                    vz = 0.0,
                    lastTimestamp = timestamp,
                    hits = 1,
                    consecutiveHits = 1,
                    missedFrames = 0,
                    status = initialStatus,
                    className = obs.className,
                    age = 1,
                    geometricConfidence = safeGeom,
                    temporalConfidence = initTemporal,
                    physicalConfidence = initPhysical,
                    lastGeometry = obs.geometry,
                    positionConsistency = 1.0,
                    depthConsistency = 1.0,
                    heightConsistency = 1.0,
                    widthConsistency = 1.0,
                    aspectRatioConsistency = 1.0,
                    groundContactConsistency = 1.0,
                    confidenceUpdateReason = initReason,
                    warningState = "NONE",
                    lastWarningTimestamp = 0.0,
                    lastWarnedAction = "NONE",
                    timeToCpa = 99.0,
                    cpaDistance = 99.0,
                    collisionThreatConfidence = 1.0,
                    hazardState = if (safeZ < 1.0 && isGeometricallyValid) HazardLifecycleState.DETECTED else HazardLifecycleState.NONE,
                    userMotionState = if (imuReliable) latestUserMotionEstimate.state else UserMotionState.MOVING
                )
                newTrack.addSnapshot(obs, timestamp)
                tracks[newId] = newTrack
            }
        }

        // Active tracks (excluding deleted)
        val activeTrackRecords = tracks.values.filter { it.status != TrackStatus.DELETED }

        // If no active tracks after pruning -> SAFE cruise
        if (activeTrackRecords.isEmpty()) {
            val tEnd = System.nanoTime()
            val totalMs = max(0.01, (tEnd - t0) / 1_000_000.0)
            lastDecisionAction = "continue"
            preferredTurnDirection = null
            lastPrimaryThreatId = null
            val cues = mutableListOf<PrismPresentationCue>()
            if (hadActiveHazardLastFrame) {
                cues.add(PrismPresentationCue("speech", "info", "warn:clear", "Path clear.", null, 0.70, 0.8, null))
                hadActiveHazardLastFrame = false
            }
            cues.add(
                PrismPresentationCue("visual", "info", "continue:safe", "[SAFE] CONTINUE", null, 0.2, 0.5, null)
            )
            return PrismFrameResult(
                frameId = frameCounter,
                timestamp = timestamp,
                runtimeState = "RUNNING",
                decision = PrismDecision("continue", "no_hazard", 1.0, null, 0.0, "safe"),
                risk = PrismRisk(null, "safe", 0.0, null, null),
                cues = cues,
                timing = PrismTiming(totalMs * 0.8, totalMs * 0.2, totalMs, 1000.0 / totalMs),
                activeTracks = 0,
                trackList = emptyList(),
                droppedFrames = 0,
                processedFrames = frameCounter,
                errorMessage = null,
                userMotion = latestUserMotionEstimate
            )
        }

        // 4. Vector 2D Collision Geometry, TTC, CPA & Threat Evaluation (Stage 5)
        data class TrackRiskEval(
            val track: AndroidTrackRecord,
            val distance: Double,
            val ttc: Double,
            val cpaDist: Double,
            val timeToCpa: Double,
            val riskScore: Double,
            val threatConfidence: Double,
            val inCorridor: Boolean,
            val isEgressing: Boolean
        )

        val riskEvals = mutableListOf<TrackRiskEval>()

        for (track in activeTrackRecords) {
            val dist = sqrt(track.x * track.x + track.z * track.z)
            val inCorridor = abs(track.x) <= 1.20
            val isEgressing = abs(track.x) > 0.45 && (track.x * track.vx > 0.12) // moving outward away from travel path

            val vSq = track.vx * track.vx + track.vz * track.vz
            val (timeToCpa, cpaDist) = if (vSq > 0.04) {
                val tCpa = -(track.x * track.vx + track.z * track.vz) / vSq
                if (tCpa > 0.0) {
                    val xCpa = track.x + track.vx * tCpa
                    val zCpa = track.z + track.vz * tCpa
                    Pair(tCpa, sqrt(xCpa * xCpa + zCpa * zCpa))
                } else {
                    Pair(0.0, dist)
                }
            } else {
                Pair(0.0, dist)
            }

            // Analytical TTC formulation
            val ttc = when {
                track.vz < -0.10 -> {
                    // Approaching longitudinally
                    max(0.1, -track.z / track.vz)
                }
                timeToCpa in 0.1..4.0 && cpaDist < 0.70 -> {
                    // Lateral crossing trajectory intersecting user corridor
                    max(0.2, timeToCpa)
                }
                track.vz > 0.15 -> {
                    // Receding obstacle moving away faster than approach
                    99.0
                }
                else -> {
                    // Stationary or slow drift
                    max(0.5, track.z / 1.0)
                }
            }

            track.timeToCpa = timeToCpa
            track.cpaDistance = cpaDist

            // Collision Threat Confidence
            val threatConfidence = when {
                track.vz > 0.20 -> 0.10 // Receding
                isEgressing -> 0.25      // Leaving corridor
                inCorridor && track.z < 2.5 -> 0.95
                inCorridor -> 0.80
                cpaDist < 0.65 && timeToCpa in 0.1..3.0 -> 0.85
                else -> 0.30
            }
            track.collisionThreatConfidence = threatConfidence

            // Spatial risk scoring
            val distFactor = max(0.0, 1.0 - (min(dist, 4.5) / 4.5))
            val latFactor = max(0.0, 1.0 - (min(abs(track.x), 1.2) / 1.2))
            val speedFactor = when {
                track.vz < -0.10 -> 1.3
                track.vz > 0.15 -> 0.20 // Substantially reduced risk for receding objects
                isEgressing -> 0.45
                else -> 1.0
            }
            val rawRisk = (distFactor * 60.0 + latFactor * 30.0) * speedFactor * threatConfidence

            // Scale by physical confidence: unconfirmed/low-confidence candidates cannot produce high risk
            val physWeight = if (track.z < 1.0 && track.geometricConfidence >= 0.50) 1.0 else track.physicalConfidence
            val riskScore = min(100.0, rawRisk * physWeight)

            riskEvals.add(
                TrackRiskEval(
                    track = track,
                    distance = dist,
                    ttc = ttc,
                    cpaDist = cpaDist,
                    timeToCpa = timeToCpa,
                    riskScore = riskScore,
                    threatConfidence = threatConfidence,
                    inCorridor = inCorridor,
                    isEgressing = isEgressing
                )
            )
        }

        // Select Primary Threat
        // If an unheld emergency exists (e.g. newly arrived Track 2), prioritize it so Track 1's hold doesn't suppress it
        val unheldEmergency = riskEvals.find {
            it.track.hazardState != HazardLifecycleState.HAZARD_PERSISTING &&
            it.track.hazardState != HazardLifecycleState.HAZARD_CLEARED &&
            ((it.track.z < 1.0 && (it.inCorridor || it.track.x in -0.6..0.6)) || it.riskScore >= 85.0)
        }
        val primary = unheldEmergency ?: (riskEvals.maxByOrNull { it.riskScore } ?: riskEvals.first())
        val primaryTrack = primary.track
        val distZ = primaryTrack.z
        val posX = primaryTrack.x

        // 5. Multi-Entity Spatial Avoidance & Clearance Evaluation
        var leftCorridorBlocked = false
        var rightCorridorBlocked = false

        for (eval in riskEvals) {
            if (eval.track.z < 2.8 && eval.inCorridor && eval.track.vz <= 0.15) {
                if (eval.track.x < -0.15) leftCorridorBlocked = true
                if (eval.track.x > 0.15) rightCorridorBlocked = true
                if (abs(eval.track.x) <= 0.15) {
                    leftCorridorBlocked = true
                    rightCorridorBlocked = true
                }
            }
        }

        // 6. Decision Arbitration with Directional Hysteresis & Emergency Priority (Stage 6)
        val action: String
        val riskLvl: String
        val riskScore: Double
        val ttc: Double?
        var recommendedHeading: Double? = null

        when {
            // Persistent hazard hold
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_PERSISTING -> {
                action = "emergency_stop"
                riskLvl = "critical"
                riskScore = max(85.0, primary.riskScore)
                ttc = primary.ttc
                preferredTurnDirection = null
            }
            // Cleared hazard (resolved)
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_CLEARED -> {
                action = "continue"
                riskLvl = "safe"
                riskScore = 0.0
                ttc = null
                preferredTurnDirection = null
            }
            // Absolute Emergency Stop Priority: close hazard (<1.0m) or critical collision threat
            (distZ < 1.0 && (primary.inCorridor || posX in -0.6..0.6)) || primary.riskScore >= 85.0 -> {
                action = "emergency_stop"
                riskLvl = "critical"
                riskScore = primary.riskScore
                ttc = primary.ttc
                preferredTurnDirection = null

                if (primaryTrack.hazardState != HazardLifecycleState.EMERGENCY) {
                    primaryTrack.hazardState = HazardLifecycleState.EMERGENCY
                    primaryTrack.hazardEncounterTimestamp = timestamp
                    primaryTrack.initialHazardDistance = distZ
                    primaryTrack.holdDistance = distZ
                    primaryTrack.userResponseExpected = true
                }
            }
            // Turn evasion with directional hysteresis (prevent rapid L/R chattering when posX ~ 0)
            distZ < 2.6 && (primary.inCorridor || primary.threatConfidence >= 0.70) && primaryTrack.vz <= 0.15 -> {
                val desiredAction = if (preferredTurnDirection != null && abs(posX) < 0.25) {
                    preferredTurnDirection!!
                } else if (leftCorridorBlocked && !rightCorridorBlocked) {
                    "turn_right"
                } else if (rightCorridorBlocked && !leftCorridorBlocked) {
                    "turn_left"
                } else if (leftCorridorBlocked && rightCorridorBlocked) {
                    "slow_down"
                } else {
                    if (posX >= 0) "turn_left" else "turn_right"
                }

                action = desiredAction
                preferredTurnDirection = if (action in listOf("turn_left", "turn_right")) action else null

                recommendedHeading = when (action) {
                    "turn_left" -> -0.45
                    "turn_right" -> 0.45
                    else -> null
                }

                riskLvl = "high"
                riskScore = primary.riskScore
                ttc = primary.ttc
                primaryTrack.hazardState = HazardLifecycleState.ACTION_REQUIRED
            }
            // Medium range slowdown
            distZ < 4.0 && (primary.inCorridor || primary.threatConfidence >= 0.70) && primaryTrack.vz <= 0.15 -> {
                action = "slow_down"
                riskLvl = "medium"
                riskScore = primary.riskScore
                ttc = primary.ttc
                preferredTurnDirection = null
                primaryTrack.hazardState = HazardLifecycleState.CAUTION
            }
            else -> {
                action = "continue"
                riskLvl = if (primary.riskScore > 20.0) "low" else "safe"
                riskScore = primary.riskScore
                ttc = primary.ttc
                preferredTurnDirection = null
                primaryTrack.hazardState = if (primary.riskScore > 20.0) HazardLifecycleState.AWARE else HazardLifecycleState.DETECTED
            }
        }

        // 7. Presentation Warning Layer (Spatial, Graduated, Deduplicated)
        val spatialDir = WarningMessageGenerator.defaultConfig.classify(posX)
        val distBand = WarningMessageGenerator.quantizeDistance(distZ)
        val movement = WarningMessageGenerator.classifyMovement(primaryTrack.vx, primaryTrack.vz, posX)
        val severity = when {
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_PERSISTING -> WarningSeverity.EMERGENCY
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_CLEARED -> WarningSeverity.CLEAR
            else -> WarningMessageGenerator.mapToSeverity(
                action = action,
                riskScore = riskScore,
                distZ = distZ,
                inCorridor = primary.inCorridor,
                vz = primaryTrack.vz,
                activeTracks = activeTrackRecords.size
            )
        }

        val anyActiveEmergencyOrHold = activeTrackRecords.any {
            it.hazardState in listOf(HazardLifecycleState.EMERGENCY, HazardLifecycleState.HAZARD_PERSISTING)
        }

        val isDeescalatingToClear = hadActiveHazardLastFrame && (severity == WarningSeverity.CLEAR || (!anyActiveEmergencyOrHold && primaryTrack.hazardState == HazardLifecycleState.HAZARD_CLEARED))
        if (severity != WarningSeverity.CLEAR && (anyActiveEmergencyOrHold || action != "continue")) {
            hadActiveHazardLastFrame = true
        } else if (isDeescalatingToClear) {
            hadActiveHazardLastFrame = false
        }

        val warnContext = WarningContext(
            severity = severity,
            className = primaryTrack.className,
            spatialDirection = spatialDir,
            distanceBand = distBand,
            movementState = movement,
            recommendedAction = action,
            recommendedHeading = recommendedHeading,
            leftCorridorBlocked = leftCorridorBlocked,
            rightCorridorBlocked = rightCorridorBlocked,
            distanceZ = distZ,
            lateralX = posX,
            riskScore = riskScore,
            ttc = ttc,
            cpaDistance = primary.cpaDist,
            threatConfidence = primary.threatConfidence,
            isDeescalatingToClear = isDeescalatingToClear
        )

        val warningMsg = WarningMessageGenerator.generate(warnContext)

        val isNewSeverity = (severity.name != primaryTrack.lastWarnedSeverity)
        val isNewDirection = (spatialDir.name != primaryTrack.lastWarnedDirection)
        val isNewDistanceBand = (distBand.name != primaryTrack.lastWarnedDistanceBand)
        val isMovementChanged = (movement != primaryTrack.lastWarnedMovement &&
                movement in listOf(MovementState.CROSSING_FROM_LEFT, MovementState.CROSSING_FROM_RIGHT))
        val isCooldownExpired = (timestamp - primaryTrack.lastWarningTimestamp) >= 1.5
        val isActionChanged = (action != primaryTrack.lastWarnedAction)

        val shouldEmitSpeech = when {
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_PERSISTING -> isCooldownExpired
            severity == WarningSeverity.EMERGENCY -> {
                primaryTrack.lastWarnedSeverity != "EMERGENCY" ||
                        isCooldownExpired ||
                        isActionChanged
            }
            severity in listOf(WarningSeverity.URGENT, WarningSeverity.DIRECTIONAL) -> {
                isNewSeverity || isMovementChanged || (isNewDirection && isCooldownExpired) || isActionChanged || isCooldownExpired
            }
            severity in listOf(WarningSeverity.CAUTION, WarningSeverity.AWARE) -> {
                isNewSeverity || isMovementChanged || ((timestamp - primaryTrack.lastWarningTimestamp) >= 2.0 && (isNewDirection || isNewDistanceBand))
            }
            severity == WarningSeverity.CLEAR -> {
                isDeescalatingToClear
            }
            else -> false
        }

        val cues = mutableListOf<PrismPresentationCue>()

        if (warningMsg.speech != null && shouldEmitSpeech) {
            val prioTag = when (warningMsg.priority) {
                AudioPriority.EMERGENCY -> "emergency"
                AudioPriority.SAFETY_WARNING -> "urgent"
                AudioPriority.DIRECTIONAL -> "urgent"
                AudioPriority.CAUTION -> "warning"
                AudioPriority.NAVIGATION -> "warning"
                AudioPriority.AWARENESS -> "info"
                AudioPriority.ADVISORY -> "info"
                AudioPriority.SYSTEM_INFO -> "info"
            }
            cues.add(
                PrismPresentationCue(
                    channel = "speech",
                    priority = prioTag,
                    messageId = "warn:${severity.name.lowercase()}",
                    text = warningMsg.speech,
                    direction = recommendedHeading,
                    intensity = if (severity == WarningSeverity.EMERGENCY) 1.0 else 0.75,
                    duration = if (severity == WarningSeverity.EMERGENCY) 0.8 else 1.0,
                    hapticPattern = null
                )
            )
        }

        val isSpatialCooldownExpired = (timestamp - primaryTrack.lastWarningTimestamp) >= 1.5

        val shouldGenerateSpatial = when {
            multisensoryMode != MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC -> false
            severity in listOf(WarningSeverity.CLEAR, WarningSeverity.AWARE) -> false
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_PERSISTING -> isSpatialCooldownExpired
            severity == WarningSeverity.EMERGENCY -> {
                primaryTrack.lastWarnedSeverity != "EMERGENCY" || isSpatialCooldownExpired || isActionChanged
            }
            severity in listOf(WarningSeverity.URGENT, WarningSeverity.DIRECTIONAL) -> {
                isNewSeverity || isActionChanged || isMovementChanged || (isNewDirection && isSpatialCooldownExpired) || isSpatialCooldownExpired
            }
            severity == WarningSeverity.CAUTION -> {
                isNewSeverity || (isNewDirection && isSpatialCooldownExpired) || isSpatialCooldownExpired
            }
            else -> false
        }

        val spatialCueObj: SpatialCue? = if (shouldGenerateSpatial) {
            when (severity) {
                WarningSeverity.EMERGENCY -> {
                    // Short urgent centered cue per Section 13 (pan = 0.0)
                    SpatialCue(
                        direction = SpatialDirection.CENTER,
                        pan = 0.0,
                        severity = WarningSeverity.EMERGENCY,
                        durationSeconds = 0.30,
                        volume = 1.0,
                        priority = AudioPriority.EMERGENCY,
                        cueType = SpatialCueType.HAZARD_LOCATION,
                        frequencyHz = 1100
                    )
                }
                WarningSeverity.URGENT, WarningSeverity.DIRECTIONAL -> {
                    when (action) {
                        "turn_left" -> {
                            // Reinforce recommended evasion direction (LEFT) per Section 10
                            SpatialCue(
                                direction = SpatialDirection.LEFT,
                                pan = -0.65,
                                severity = severity,
                                durationSeconds = 0.25,
                                volume = if (severity == WarningSeverity.URGENT) 0.90 else 0.80,
                                priority = if (severity == WarningSeverity.URGENT) AudioPriority.SAFETY_WARNING else AudioPriority.DIRECTIONAL,
                                cueType = SpatialCueType.RECOMMENDED_DIRECTION,
                                frequencyHz = 750
                            )
                        }
                        "turn_right" -> {
                            // Reinforce recommended evasion direction (RIGHT) per Section 10
                            SpatialCue(
                                direction = SpatialDirection.RIGHT,
                                pan = 0.65,
                                severity = severity,
                                durationSeconds = 0.25,
                                volume = if (severity == WarningSeverity.URGENT) 0.90 else 0.80,
                                priority = if (severity == WarningSeverity.URGENT) AudioPriority.SAFETY_WARNING else AudioPriority.DIRECTIONAL,
                                cueType = SpatialCueType.RECOMMENDED_DIRECTION,
                                frequencyHz = 750
                            )
                        }
                        else -> {
                            // Slow down / path narrowing -> centered recommended direction
                            SpatialCue(
                                direction = SpatialDirection.CENTER,
                                pan = 0.0,
                                severity = severity,
                                durationSeconds = 0.25,
                                volume = 0.80,
                                priority = if (severity == WarningSeverity.URGENT) AudioPriority.SAFETY_WARNING else AudioPriority.DIRECTIONAL,
                                cueType = SpatialCueType.RECOMMENDED_DIRECTION,
                                frequencyHz = 600
                            )
                        }
                    }
                }
                WarningSeverity.CAUTION -> {
                    // Hazard location beacon panned to obstacle sector
                    val panVal = SpatialCue.directionToPan(spatialDir)
                    SpatialCue(
                        direction = spatialDir,
                        pan = panVal,
                        severity = WarningSeverity.CAUTION,
                        durationSeconds = 0.20,
                        volume = 0.65,
                        priority = AudioPriority.CAUTION,
                        cueType = SpatialCueType.HAZARD_LOCATION,
                        frequencyHz = 650
                    )
                }
                else -> null
            }
        } else null

        if (spatialCueObj != null) {
            val prioTag = when (spatialCueObj.priority) {
                AudioPriority.EMERGENCY -> "emergency"
                AudioPriority.SAFETY_WARNING -> "urgent"
                AudioPriority.DIRECTIONAL -> "urgent"
                AudioPriority.CAUTION -> "warning"
                else -> "info"
            }
            val cueTypeStr = spatialCueObj.cueType.name.lowercase()
            val dirStr = spatialCueObj.direction.name.lowercase()
            cues.add(
                PrismPresentationCue(
                    channel = "spatial_audio",
                    priority = prioTag,
                    messageId = "spatial:${cueTypeStr}:${dirStr}",
                    text = null,
                    direction = spatialCueObj.pan,
                    intensity = spatialCueObj.volume,
                    duration = spatialCueObj.durationSeconds,
                    hapticPattern = null,
                    spatialCue = spatialCueObj
                )
            )
        }

        if (shouldEmitSpeech || shouldGenerateSpatial) {
            primaryTrack.lastWarningTimestamp = timestamp
            primaryTrack.lastWarnedSeverity = severity.name
            primaryTrack.lastWarnedDirection = spatialDir.name
            primaryTrack.lastWarnedDistanceBand = distBand.name
            primaryTrack.lastWarnedAction = action
            primaryTrack.lastWarnedMovement = movement
            primaryTrack.warningState = severity.name
        }

        val shouldEmitHaptic = (multisensoryMode != MultisensoryMode.SPEECH_ONLY) && (warningMsg.hapticPattern != null)
        if (shouldEmitHaptic && warningMsg.hapticPattern != null) {
            val prioTag = when (warningMsg.priority) {
                AudioPriority.EMERGENCY -> "emergency"
                AudioPriority.SAFETY_WARNING -> "urgent"
                AudioPriority.DIRECTIONAL -> "urgent"
                AudioPriority.CAUTION -> "warning"
                else -> "info"
            }
            cues.add(
                PrismPresentationCue(
                    channel = "haptic",
                    priority = prioTag,
                    messageId = "haptic:${warningMsg.hapticPattern}",
                    text = null,
                    direction = recommendedHeading,
                    intensity = if (severity == WarningSeverity.EMERGENCY) 1.0 else 0.70,
                    duration = 0.8,
                    hapticPattern = warningMsg.hapticPattern
                )
            )
        }

        val visPriority = when (severity) {
            WarningSeverity.EMERGENCY -> "emergency"
            WarningSeverity.URGENT, WarningSeverity.DIRECTIONAL -> "urgent"
            WarningSeverity.CAUTION -> "warning"
            else -> "info"
        }
        val visText = when {
            primaryTrack.hazardState == HazardLifecycleState.HAZARD_PERSISTING -> "[HOLD] EMERGENCY_STOP"
            else -> warningMsg.visualText ?: "[${riskLvl.uppercase()}] ${action.uppercase()}"
        }
        cues.add(
            PrismPresentationCue(
                channel = "visual",
                priority = visPriority,
                messageId = "vis:${severity.name.lowercase()}",
                text = visText,
                direction = recommendedHeading,
                intensity = if (severity == WarningSeverity.EMERGENCY) 1.0 else 0.70,
                duration = 0.5,
                hapticPattern = null
            )
        )

        lastDecisionAction = action
        lastPrimaryThreatId = primaryTrack.trackId

        val trackList = activeTrackRecords.map {
            TrackInfo(
                trackId = it.trackId,
                x = it.x,
                z = it.z,
                vx = it.vx,
                vz = it.vz,
                className = it.className,
                status = it.status.name,
                hits = it.hits,
                consecutiveHits = it.consecutiveHits,
                missedFrames = it.missedFrames,
                age = it.age,
                geometricConfidence = it.geometricConfidence,
                temporalConfidence = it.temporalConfidence,
                physicalConfidence = it.physicalConfidence,
                widthMeters = it.lastGeometry?.widthMeters ?: 0.0,
                heightMeters = it.lastGeometry?.heightMeters ?: 0.0,
                aspectRatio = it.lastGeometry?.aspectRatio ?: 0.0,
                clusterSize = it.lastGeometry?.clusterSize ?: 0,
                geometryValid = it.lastGeometry?.geometryValid ?: true,
                geometryScore = it.lastGeometry?.geometryScore ?: it.geometricConfidence,
                rejectReason = it.lastGeometry?.rejectReason,
                positionConsistency = it.positionConsistency,
                depthConsistency = it.depthConsistency,
                heightConsistency = it.heightConsistency,
                widthConsistency = it.widthConsistency,
                aspectRatioConsistency = it.aspectRatioConsistency,
                groundContactConsistency = it.groundContactConsistency,
                confidenceUpdateReason = it.confidenceUpdateReason,
                shapeClass = it.lastGeometry?.shapeClass ?: "UNCERTAIN",
                fillRatio = it.lastGeometry?.fillRatio ?: 1.0,
                floorStructureLikelihood = it.lastGeometry?.floorStructureLikelihood ?: 0.0,
                groundPlaneCoherence = it.lastGeometry?.groundPlaneCoherence ?: 1.0,
                warningState = it.warningState,
                timeToCpa = it.timeToCpa,
                cpaDistance = it.cpaDistance,
                collisionThreatConfidence = it.collisionThreatConfidence,
                hazardState = it.hazardState.name,
                hazardEncounterTimestamp = it.hazardEncounterTimestamp,
                hazardResolvedTimestamp = it.hazardResolvedTimestamp,
                userResponseExpected = it.userResponseExpected,
                hazardPersistenceDuration = if (it.hazardEncounterTimestamp > 0.0) max(0.0, timestamp - it.hazardEncounterTimestamp) else 0.0,
                initialHazardDistance = it.initialHazardDistance,
                userMotionState = it.userMotionState.name,
                motionConfidence = latestUserMotionEstimate.confidence,
                stepCount = latestUserMotionEstimate.stepCount,
                timeSinceLastStep = latestUserMotionEstimate.timeSinceLastStepSeconds,
                motionEnergy = latestUserMotionEstimate.motionEnergy
            )
        }

        val tEnd = System.nanoTime()
        val totalMs = max(0.01, (tEnd - t0) / 1_000_000.0)

        return PrismFrameResult(
            frameId = frameCounter,
            timestamp = timestamp,
            runtimeState = "RUNNING",
            decision = PrismDecision(action, "collision_geometry", 0.95, primaryTrack.trackId, recommendedHeading, riskLvl),
            risk = PrismRisk(primaryTrack.trackId, riskLvl, riskScore, ttc, abs(posX)),
            cues = cues,
            timing = PrismTiming(totalMs * 0.8, totalMs * 0.2, totalMs, 1000.0 / totalMs),
            activeTracks = activeTrackRecords.size,
            trackList = trackList,
            droppedFrames = 0,
            processedFrames = frameCounter,
            errorMessage = null,
            userMotion = latestUserMotionEstimate
        )
    }

    /**
     * Evaluates lightweight frame-to-frame temporal consistency and updates track confidence.
     */
    private fun updateTrackConsistency(
        trk: AndroidTrackRecord,
        obs: PrismObservation,
        dt: Double,
        timestamp: Double
    ) {
        val predX = trk.x + trk.vx * dt
        val predZ = trk.z + trk.vz * dt
        val geom = obs.geometry

        // 1. Position Consistency (comparing observed vs predicted position)
        val predDistSq = (obs.x - predX) * (obs.x - predX) + (obs.z - predZ) * (obs.z - predZ)
        val predError = sqrt(predDistSq)
        val posConsistency = if (predError.isNaN()) 0.0 else max(0.0, 1.0 - (predError / 1.5))

        // 2. Depth Consistency (comparing delta Z with expected velocity movement)
        val expectedDz = trk.vz * dt
        val actualDz = obs.z - trk.z
        val depthJumpError = abs(actualDz - expectedDz)
        val depthConsistency = if (depthJumpError.isNaN()) 0.0 else max(0.0, 1.0 - (depthJumpError / 1.2))

        // 3. Height Consistency (physical upright 3D height stability across frames)
        val prevHeight = trk.lastGeometry?.heightMeters ?: trk.history.lastOrNull()?.heightMeters ?: (geom?.heightMeters ?: 1.0)
        val currHeight = geom?.heightMeters ?: prevHeight
        val deltaHeight = abs(currHeight - prevHeight)
        val heightConsistency = if (deltaHeight.isNaN()) 0.0 else max(0.0, 1.0 - (deltaHeight / 0.50))

        // 4. Width Consistency
        val prevWidth = trk.lastGeometry?.widthMeters ?: trk.history.lastOrNull()?.widthMeters ?: (geom?.widthMeters ?: 0.5)
        val currWidth = geom?.widthMeters ?: prevWidth
        val deltaWidth = abs(currWidth - prevWidth)
        val widthConsistency = if (deltaWidth.isNaN()) 0.0 else max(0.0, 1.0 - (deltaWidth / 0.60))

        // 5. Aspect Ratio Consistency
        val prevAR = trk.lastGeometry?.aspectRatio ?: (prevHeight / max(0.01, prevWidth))
        val currAR = geom?.aspectRatio ?: (currHeight / max(0.01, currWidth))
        val deltaAR = abs(currAR - prevAR)
        val arConsistency = if (deltaAR.isNaN()) 0.0 else max(0.0, 1.0 - (deltaAR / 2.0))

        // 6. Ground-Contact Consistency
        val prevGroundV = trk.lastGeometry?.groundContactV ?: trk.history.lastOrNull()?.groundContactV ?: (geom?.groundContactV ?: 300.0)
        val currGroundV = geom?.groundContactV ?: prevGroundV
        val deltaGroundV = abs(currGroundV - prevGroundV)
        val groundConsistency = if (deltaGroundV.isNaN()) 0.0 else max(0.0, 1.0 - (deltaGroundV / 120.0))

        trk.positionConsistency = posConsistency.coerceIn(0.0, 1.0)
        trk.depthConsistency = depthConsistency.coerceIn(0.0, 1.0)
        trk.heightConsistency = heightConsistency.coerceIn(0.0, 1.0)
        trk.widthConsistency = widthConsistency.coerceIn(0.0, 1.0)
        trk.aspectRatioConsistency = arConsistency.coerceIn(0.0, 1.0)
        trk.groundContactConsistency = groundConsistency.coerceIn(0.0, 1.0)

        // Instantaneous Temporal Consistency Score
        val instTemporalScore = (
            0.25 * trk.positionConsistency +
            0.25 * trk.depthConsistency +
            0.25 * trk.heightConsistency +
            0.15 * trk.widthConsistency +
            0.10 * trk.groundContactConsistency
        ).coerceIn(0.0, 1.0)

        // Exponential smoothing accumulation
        val alphaTemp = 0.35
        val safeInstTemp = if (instTemporalScore.isNaN() || instTemporalScore.isInfinite()) 0.0 else instTemporalScore
        val safePrevTemp = if (trk.temporalConfidence.isNaN() || trk.temporalConfidence.isInfinite()) 0.50 else trk.temporalConfidence
        val rawTemp = alphaTemp * safeInstTemp + (1.0 - alphaTemp) * safePrevTemp
        trk.temporalConfidence = if (rawTemp.isNaN() || rawTemp.isInfinite()) 0.0 else rawTemp.coerceIn(0.0, 1.0)

        val rawGeom = obs.geometricConfidence
        val safeGeom = if (rawGeom.isNaN() || rawGeom.isInfinite()) 0.0 else rawGeom.coerceIn(0.0, 1.0)
        trk.geometricConfidence = safeGeom

        val rawPhys = 0.45 * trk.geometricConfidence + 0.55 * trk.temporalConfidence
        trk.physicalConfidence = if (rawPhys.isNaN() || rawPhys.isInfinite()) 0.0 else rawPhys.coerceIn(0.0, 1.0)

        // Add observation to bounded snapshot history
        trk.addSnapshot(obs, timestamp)

        // Lifecycle transition logic
        val isCloseHazard = (obs.z < 1.0)
        val isGeometricallyValid = (trk.geometricConfidence >= 0.50 && (geom?.heightMeters ?: 1.0) >= 0.32)

        if (trk.status == TrackStatus.TENTATIVE) {
            if (isCloseHazard && isGeometricallyValid) {
                trk.status = TrackStatus.CONFIRMED
                trk.confidenceUpdateReason = "CLOSE_HAZARD_FAST_CONFIRM"
            } else if (trk.consecutiveHits >= confirmationHits && trk.physicalConfidence >= 0.40) {
                trk.status = TrackStatus.CONFIRMED
                trk.confidenceUpdateReason = "TEMPORAL_CONFIRMED"
            } else {
                trk.confidenceUpdateReason = "TENTATIVE_EVAL"
            }
        } else if (trk.status == TrackStatus.LOST) {
            if (trk.physicalConfidence >= 0.40) {
                trk.status = TrackStatus.CONFIRMED
                trk.confidenceUpdateReason = "REACQUIRED_CONFIRMED"
            } else {
                trk.status = TrackStatus.TENTATIVE
                trk.confidenceUpdateReason = "REACQUIRED_TENTATIVE"
            }
        } else {
            trk.confidenceUpdateReason = "TRACK_UPDATED"
        }
    }
}
