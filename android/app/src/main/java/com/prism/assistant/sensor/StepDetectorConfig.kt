package com.prism.assistant.sensor

/**
 * Configurable thresholds and timing constraints for IMU step detection and ego-motion estimation.
 */
data class StepDetectorConfig(
    val stepThresholdMps2: Double = 1.20,          // Peak dynamic acceleration threshold (m/s^2)
    val refractoryPeriodSeconds: Double = 0.28,    // Min interval between steps (refractory period)
    val maxStepIntervalSeconds: Double = 1.80,     // Max interval between steps before cadence breaks
    val stationaryEnergyThreshold: Double = 0.25,  // Motion energy (RMS) threshold for stationary state
    val activeEnergyThreshold: Double = 0.60,      // Motion energy (RMS) threshold for active movement
    val gyroStationaryThreshold: Double = 0.20,    // Max angular velocity (rad/s) for stationary state
    val minStepsForWalking: Int = 2,               // Steps required to promote STARTING to WALKING
    val stoppingTimeoutSeconds: Double = 1.00,     // Timeout without steps to enter STOPPING
    val stationaryTimeoutSeconds: Double = 1.50    // Timeout without steps to enter STATIONARY
)
