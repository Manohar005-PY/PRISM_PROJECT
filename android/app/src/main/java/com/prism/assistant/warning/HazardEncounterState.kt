package com.prism.assistant.warning

import com.prism.assistant.sensor.UserMotionState
import kotlin.math.max

/**
 * Lifecycle states representing the evolution of a physical obstacle encounter relative to the user.
 */
enum class HazardLifecycleState {
    NONE,
    DETECTED,
    AWARE,
    CAUTION,
    ACTION_REQUIRED,
    EMERGENCY,
    USER_RESPONDING,
    HAZARD_PERSISTING,
    STOPPED_FOR_HAZARD,
    HAZARD_CLEARED
}

typealias UserMotionState = com.prism.assistant.sensor.UserMotionState

/**
 * Structured state tracking per-track hazard encounter lifecycle and user response modeling.
 */
data class HazardEncounterRecord(
    val trackId: Int,
    var hazardState: HazardLifecycleState = HazardLifecycleState.NONE,
    var userMotionState: UserMotionState = UserMotionState.MOVING,
    var userResponseExpected: Boolean = false,
    var encounterTimestamp: Double = 0.0,
    var lastWarningSeverity: String = "CLEAR",
    var lastWarningTimestamp: Double = 0.0,
    var lastDecision: String = "continue",
    var initialHazardDistance: Double = 0.0,
    var holdDistance: Double = 0.0,
    var resolutionTimestamp: Double? = null,
    var isSafetyHoldActive: Boolean = false
) {
    fun hazardAge(currentTimestamp: Double): Double {
        return if (encounterTimestamp > 0.0) max(0.0, currentTimestamp - encounterTimestamp) else 0.0
    }
}
