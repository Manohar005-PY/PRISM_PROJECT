package com.prism.assistant.sensor

/**
 * Dedicated user motion state representing device-level ego-motion inferred from IMU sensors.
 */
enum class UserMotionState {
    UNKNOWN,
    STATIONARY,
    WALKING,
    MOVING,
    STOPPING,
    STARTING,
    STOPPED,       // Backward compatibility with Phase I tests
    BACKING_AWAY;  // Backward compatibility with Phase I tests

    val isStationary: Boolean get() = this == STATIONARY || this == STOPPING || this == STOPPED
    val isMoving: Boolean get() = this == WALKING || this == MOVING || this == STARTING
}

/**
 * Operating mode for user motion inference comparison (A/B testing).
 */
enum class UserMotionInferenceMode {
    USER_MOTION_MODE_RELATIVE_ONLY,
    USER_MOTION_MODE_IMU_FUSED
}

/**
 * Snapshot of device-level ego-motion evidence estimated from IMU accelerometer and gyroscope.
 */
data class UserMotionEstimate(
    val state: UserMotionState = UserMotionState.UNKNOWN,
    val confidence: Double = 0.0,
    val stepCount: Int = 0,
    val timeSinceLastStepSeconds: Double = 99.0,
    val motionEnergy: Double = 0.0,
    val cadenceStepsPerMinute: Double = 0.0,
    val timestampNs: Long = 0L,
    val isStepCandidate: Boolean = false
)
