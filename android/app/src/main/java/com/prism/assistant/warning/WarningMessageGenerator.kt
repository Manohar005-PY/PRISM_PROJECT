package com.prism.assistant.warning

import com.prism.assistant.accessibility.AudioPriority
import kotlin.math.abs

/**
 * Conceptual warning severity levels for graduated collision assistance.
 */
enum class WarningSeverity(val level: Int) {
    CLEAR(0),        // No relevant immediate obstacle threat
    AWARE(1),        // Low-risk object exists (awareness, no immediate action)
    CAUTION(2),      // Relevant to user path, moderate risk, no emergency response yet
    DIRECTIONAL(3),  // Explicit steering recommendation with verified corridor clearance
    URGENT(4),       // High collision probability, immediate corrective movement required
    EMERGENCY(5)     // Critical collision threat, immediate stopping required
}

/**
 * Conceptual spatial regions relative to forward trajectory.
 */
enum class SpatialDirection(val spokenName: String) {
    FAR_LEFT("far left"),
    LEFT("left"),
    SLIGHTLY_LEFT("slightly left"),
    CENTER("directly ahead"),
    SLIGHTLY_RIGHT("slightly right"),
    RIGHT("right"),
    FAR_RIGHT("far right")
}

/**
 * Configurable boundaries for spatial lateral partitioning.
 */
data class SpatialMappingConfig(
    val centerThreshold: Double = 0.15,
    val slightThreshold: Double = 0.40,
    val regularThreshold: Double = 0.85
) {
    fun classify(x: Double, previousDirection: SpatialDirection? = null): SpatialDirection {
        if (previousDirection == null) {
            return when {
                x < -regularThreshold -> SpatialDirection.FAR_LEFT
                x < -slightThreshold -> SpatialDirection.LEFT
                x < -centerThreshold -> SpatialDirection.SLIGHTLY_LEFT
                x <= centerThreshold -> SpatialDirection.CENTER
                x <= slightThreshold -> SpatialDirection.SLIGHTLY_RIGHT
                x <= regularThreshold -> SpatialDirection.RIGHT
                else -> SpatialDirection.FAR_RIGHT
            }
        }
        val h = 0.05 // Hysteresis margin
        return when (previousDirection) {
            SpatialDirection.CENTER -> when {
                x < -(centerThreshold + h) -> SpatialDirection.SLIGHTLY_LEFT
                x > (centerThreshold + h) -> SpatialDirection.SLIGHTLY_RIGHT
                else -> SpatialDirection.CENTER
            }
            SpatialDirection.SLIGHTLY_LEFT -> when {
                x > -centerThreshold + h -> SpatialDirection.CENTER
                x < -(slightThreshold + h) -> SpatialDirection.LEFT
                else -> SpatialDirection.SLIGHTLY_LEFT
            }
            SpatialDirection.SLIGHTLY_RIGHT -> when {
                x < centerThreshold - h -> SpatialDirection.CENTER
                x > (slightThreshold + h) -> SpatialDirection.RIGHT
                else -> SpatialDirection.SLIGHTLY_RIGHT
            }
            SpatialDirection.LEFT -> when {
                x > -slightThreshold + h -> SpatialDirection.SLIGHTLY_LEFT
                x < -(regularThreshold + h) -> SpatialDirection.FAR_LEFT
                else -> SpatialDirection.LEFT
            }
            SpatialDirection.RIGHT -> when {
                x < slightThreshold - h -> SpatialDirection.SLIGHTLY_RIGHT
                x > (regularThreshold + h) -> SpatialDirection.FAR_RIGHT
                else -> SpatialDirection.RIGHT
            }
            SpatialDirection.FAR_LEFT -> {
                if (x > -regularThreshold + h) SpatialDirection.LEFT else SpatialDirection.FAR_LEFT
            }
            SpatialDirection.FAR_RIGHT -> {
                if (x < regularThreshold - h) SpatialDirection.RIGHT else SpatialDirection.FAR_RIGHT
            }
        }
    }
}

/**
 * Quantized distance ranges to avoid noisy continuous centimeter readouts.
 */
enum class DistanceBand(val spokenDescription: String) {
    CRITICAL("close"),                   // < 1.0m
    CLOSE("about one meter"),            // 1.0m - 1.8m
    NEAR("about two meters"),            // 1.8m - 2.8m
    MEDIUM("about three meters"),        // 2.8m - 3.8m
    FAR("about four meters"),            // 3.8m - 4.8m
    VERY_FAR("ahead")                    // >= 4.8m
}

/**
 * Movement classification inferred from Kalman tracking velocity.
 */
enum class MovementState {
    STATIONARY,
    APPROACHING,
    CROSSING_FROM_LEFT,
    CROSSING_FROM_RIGHT,
    MOVING_AWAY
}

/**
 * Presentation context aggregating decision, spatial, kinematic, and clearance inputs.
 */
data class WarningContext(
    val severity: WarningSeverity,
    val className: String = "obstacle",
    val spatialDirection: SpatialDirection = SpatialDirection.CENTER,
    val distanceBand: DistanceBand = DistanceBand.NEAR,
    val movementState: MovementState = MovementState.STATIONARY,
    val recommendedAction: String = "continue",
    val recommendedHeading: Double? = null,
    val leftCorridorBlocked: Boolean = false,
    val rightCorridorBlocked: Boolean = false,
    val distanceZ: Double = 3.0,
    val lateralX: Double = 0.0,
    val riskScore: Double = 0.0,
    val ttc: Double? = null,
    val cpaDistance: Double? = null,
    val threatConfidence: Double = 1.0,
    val isDeescalatingToClear: Boolean = false,
    val isInitialClear: Boolean = false
)

/**
 * Output generated by WarningMessageGenerator for downstream audio, haptic, and visual channels.
 */
data class WarningMessage(
    val speech: String?,
    val hapticPattern: String?,
    val severity: WarningSeverity,
    val priority: AudioPriority,
    val visualText: String? = null
)

/**
 * Deterministic presentation-layer warning message generator translating system decision and
 * tracking state into spatial, graduated, non-visual communication.
 */
object WarningMessageGenerator {

    val defaultConfig = SpatialMappingConfig()

    fun quantizeDistance(z: Double, previousBand: DistanceBand? = null): DistanceBand {
        if (previousBand == null) {
            return when {
                z < 1.0 -> DistanceBand.CRITICAL
                z < 1.8 -> DistanceBand.CLOSE
                z < 2.8 -> DistanceBand.NEAR
                z < 3.8 -> DistanceBand.MEDIUM
                z < 4.8 -> DistanceBand.FAR
                else -> DistanceBand.VERY_FAR
            }
        }
        val margin = 0.12 // Schmitt trigger deadband
        return when (previousBand) {
            DistanceBand.CRITICAL -> if (z >= 1.0 + margin) DistanceBand.CLOSE else DistanceBand.CRITICAL
            DistanceBand.CLOSE -> when {
                z < 1.0 - margin -> DistanceBand.CRITICAL
                z >= 1.8 + margin -> DistanceBand.NEAR
                else -> DistanceBand.CLOSE
            }
            DistanceBand.NEAR -> when {
                z < 1.8 - margin -> DistanceBand.CLOSE
                z >= 2.8 + margin -> DistanceBand.MEDIUM
                else -> DistanceBand.NEAR
            }
            DistanceBand.MEDIUM -> when {
                z < 2.8 - margin -> DistanceBand.NEAR
                z >= 3.8 + margin -> DistanceBand.FAR
                else -> DistanceBand.MEDIUM
            }
            DistanceBand.FAR -> when {
                z < 3.8 - margin -> DistanceBand.MEDIUM
                z >= 4.8 + margin -> DistanceBand.VERY_FAR
                else -> DistanceBand.FAR
            }
            DistanceBand.VERY_FAR -> if (z < 4.8 - margin) DistanceBand.FAR else DistanceBand.VERY_FAR
        }
    }

    fun classifyMovement(vx: Double, vz: Double, x: Double): MovementState {
        return when {
            vz > 0.15 -> MovementState.MOVING_AWAY
            vx > 0.10 && x < 0.60 -> MovementState.CROSSING_FROM_LEFT
            vx < -0.10 && x > -0.60 -> MovementState.CROSSING_FROM_RIGHT
            vz < -0.15 && abs(vx) <= 0.25 -> MovementState.APPROACHING
            else -> MovementState.STATIONARY
        }
    }

    fun mapToSeverity(
        action: String,
        riskScore: Double,
        distZ: Double,
        inCorridor: Boolean,
        vz: Double,
        activeTracks: Int,
        previousSeverity: WarningSeverity? = null
    ): WarningSeverity {
        return when {
            action == "emergency_stop" || riskScore >= 85.0 || (distZ < 1.0 && (inCorridor || abs(distZ) < 1.0)) -> {
                WarningSeverity.EMERGENCY
            }
            action in listOf("turn_left", "turn_right") -> {
                val urgentThreshold = if (previousSeverity == WarningSeverity.URGENT) 2.05 else 1.8
                val scoreThreshold = if (previousSeverity == WarningSeverity.URGENT) 60.0 else 65.0
                if (distZ < urgentThreshold || riskScore >= scoreThreshold) WarningSeverity.URGENT else WarningSeverity.DIRECTIONAL
            }
            action == "slow_down" -> {
                val urgentThreshold = if (previousSeverity == WarningSeverity.URGENT) 2.25 else 2.0
                if (distZ < urgentThreshold) WarningSeverity.URGENT else WarningSeverity.CAUTION
            }
            activeTracks > 0 -> {
                if (!inCorridor && (riskScore <= 25.0 || action == "continue")) {
                    WarningSeverity.CLEAR
                } else if (vz > 0.15 || distZ >= 4.0 || riskScore <= 20.0) {
                    WarningSeverity.AWARE
                } else {
                    val cautionDist = if (previousSeverity == WarningSeverity.CAUTION) 3.8 else 3.5
                    if (distZ < cautionDist && (inCorridor || riskScore > 20.0)) {
                        WarningSeverity.CAUTION
                    } else {
                        WarningSeverity.AWARE
                    }
                }
            }
            else -> WarningSeverity.CLEAR
        }
    }

    fun generate(context: WarningContext): WarningMessage {
        val isPerson = context.className.equals("person", ignoreCase = true)
        val objType = if (isPerson) "Person" else "Obstacle"

        return when (context.severity) {
            WarningSeverity.CLEAR -> {
                val speech = if (context.isDeescalatingToClear || context.isInitialClear) "Path clear." else null
                WarningMessage(
                    speech = speech,
                    hapticPattern = null,
                    severity = WarningSeverity.CLEAR,
                    priority = AudioPriority.SYSTEM_INFO,
                    visualText = "[CLEAR] PATH_CLEAR"
                )
            }

            WarningSeverity.AWARE -> {
                val speech = when {
                    context.movementState == MovementState.MOVING_AWAY -> {
                        if (isPerson) "Person ahead, moving away." else "Object ahead, moving away."
                    }
                    context.spatialDirection == SpatialDirection.LEFT || context.spatialDirection == SpatialDirection.FAR_LEFT -> {
                        "$objType ahead, left."
                    }
                    context.spatialDirection == SpatialDirection.RIGHT || context.spatialDirection == SpatialDirection.FAR_RIGHT -> {
                        "$objType ahead, right."
                    }
                    else -> {
                        if (isPerson) "Person ahead." else "Object ahead."
                    }
                }
                WarningMessage(
                    speech = speech,
                    hapticPattern = null,
                    severity = WarningSeverity.AWARE,
                    priority = AudioPriority.AWARENESS,
                    visualText = "[AWARE] $speech"
                )
            }

            WarningSeverity.CAUTION -> {
                val speech = when {
                    context.movementState == MovementState.CROSSING_FROM_LEFT -> "Person crossing from left."
                    context.movementState == MovementState.CROSSING_FROM_RIGHT -> "Person crossing from right."
                    context.spatialDirection == SpatialDirection.CENTER -> {
                        when (context.distanceBand) {
                            DistanceBand.NEAR -> "$objType directly ahead, about two meters."
                            DistanceBand.MEDIUM -> "$objType ahead, about three meters."
                            DistanceBand.FAR -> "$objType ahead, about four meters."
                            else -> "Obstacle directly ahead."
                        }
                    }
                    context.spatialDirection in listOf(SpatialDirection.SLIGHTLY_LEFT, SpatialDirection.LEFT) -> {
                        "$objType ahead, left."
                    }
                    context.spatialDirection in listOf(SpatialDirection.SLIGHTLY_RIGHT, SpatialDirection.RIGHT) -> {
                        "$objType ahead, right."
                    }
                    context.distanceBand == DistanceBand.NEAR -> "$objType ahead, about two meters."
                    context.distanceBand == DistanceBand.MEDIUM -> "$objType ahead, about three meters."
                    context.distanceBand == DistanceBand.FAR -> "$objType ahead, about four meters."
                    else -> "$objType ahead."
                }
                WarningMessage(
                    speech = speech,
                    hapticPattern = "warning",
                    severity = WarningSeverity.CAUTION,
                    priority = AudioPriority.CAUTION,
                    visualText = "[CAUTION] $speech"
                )
            }

            WarningSeverity.DIRECTIONAL -> {
                val speech = when {
                    context.movementState == MovementState.CROSSING_FROM_LEFT -> {
                        if (!context.leftCorridorBlocked) "Person crossing from left. Move left."
                        else if (!context.rightCorridorBlocked) "Person crossing from left. Move right."
                        else "Person crossing from left. Slow down."
                    }
                    context.movementState == MovementState.CROSSING_FROM_RIGHT -> {
                        if (!context.rightCorridorBlocked) "Person crossing from right. Move right."
                        else if (!context.leftCorridorBlocked) "Person crossing from right. Move left."
                        else "Person crossing from right. Slow down."
                    }
                    context.recommendedAction == "turn_left" -> {
                        if (!context.leftCorridorBlocked) "Obstacle ahead. Move left." else "Obstacle ahead. Slow down."
                    }
                    context.recommendedAction == "turn_right" -> {
                        if (!context.rightCorridorBlocked) "Obstacle ahead. Move right." else "Obstacle ahead. Slow down."
                    }
                    context.recommendedAction == "slow_down" && context.leftCorridorBlocked && context.rightCorridorBlocked -> {
                        "Path narrowing. Slow down."
                    }
                    else -> {
                        if (context.spatialDirection == SpatialDirection.CENTER) "Obstacle directly ahead."
                        else "Obstacle ahead. Slow down."
                    }
                }
                val haptic = when (context.recommendedAction) {
                    "turn_left" -> "left"
                    "turn_right" -> "right"
                    else -> "slow"
                }
                WarningMessage(
                    speech = speech,
                    hapticPattern = haptic,
                    severity = WarningSeverity.DIRECTIONAL,
                    priority = AudioPriority.DIRECTIONAL,
                    visualText = "[DIRECTIONAL] $speech"
                )
            }

            WarningSeverity.URGENT -> {
                val speech = when {
                    context.leftCorridorBlocked && context.rightCorridorBlocked -> {
                        "Path blocked. Stop now."
                    }
                    context.recommendedAction == "turn_left" -> {
                        if (context.distanceBand in listOf(DistanceBand.CRITICAL, DistanceBand.CLOSE)) {
                            "Obstacle close. Move left."
                        } else {
                            "Path blocked. Move left now."
                        }
                    }
                    context.recommendedAction == "turn_right" -> {
                        if (context.distanceBand in listOf(DistanceBand.CRITICAL, DistanceBand.CLOSE)) {
                            "Obstacle close. Move right."
                        } else {
                            "Path blocked. Move right now."
                        }
                    }
                    else -> {
                        if (context.spatialDirection == SpatialDirection.CENTER) "Obstacle close. Slow down."
                        else "Obstacle close."
                    }
                }
                val haptic = when (context.recommendedAction) {
                    "turn_left" -> "left"
                    "turn_right" -> "right"
                    else -> "slow"
                }
                WarningMessage(
                    speech = speech,
                    hapticPattern = haptic,
                    severity = WarningSeverity.URGENT,
                    priority = AudioPriority.SAFETY_WARNING,
                    visualText = "[URGENT] $speech"
                )
            }

            WarningSeverity.EMERGENCY -> {
                val speech = if (context.leftCorridorBlocked && context.rightCorridorBlocked && abs(context.lateralX) > 0.25) {
                    "Path blocked. Stop."
                } else {
                    "Stop. Obstacle directly ahead."
                }
                WarningMessage(
                    speech = speech,
                    hapticPattern = "emergency",
                    severity = WarningSeverity.EMERGENCY,
                    priority = AudioPriority.EMERGENCY,
                    visualText = "[EMERGENCY] $speech"
                )
            }
        }
    }
}
