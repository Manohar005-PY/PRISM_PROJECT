package com.prism.assistant.sensor

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Interface defining user motion and step detection operations.
 */
interface IUserMotionEstimator {
    val config: StepDetectorConfig
    fun processAccelerometerSample(ax: Double, ay: Double, az: Double, timestampNs: Long, gx: Double? = null, gy: Double? = null, gz: Double? = null)
    fun processGyroSample(wx: Double, wy: Double, wz: Double, timestampNs: Long)
    fun getMotionEstimate(): UserMotionEstimate
    fun reset()
}

/**
 * Lightweight, deterministic IMU Ego-Motion Estimator and Step Detector.
 *
 * Implements orientation-invariant dynamic acceleration magnitude extraction,
 * low-pass motion filtering, peak detection with refractory period constraints,
 * motion energy integration, and 6-state user motion lifecycle modeling.
 */
class UserMotionEstimator(
    override val config: StepDetectorConfig = StepDetectorConfig()
) : IUserMotionEstimator {

    private var currentState: UserMotionState = UserMotionState.UNKNOWN
    private var confidence: Double = 0.0
    private var totalStepCount: Int = 0
    private var consecutiveSteps: Int = 0
    private var lastStepTimestampNs: Long = 0L
    private var lastAccelTimestampNs: Long = 0L
    private var lastGyroTimestampNs: Long = 0L

    // Internal filter state
    private var smoothedDynAcc: Double = 0.0
    private var prevSmoothed1: Double = 0.0
    private var prevSmoothed2: Double = 0.0
    private var motionEnergyEma: Double = 0.0
    private var gyroMagEma: Double = 0.0
    private var currentCadence: Double = 0.0
    private var isLastSampleStep: Boolean = false

    // Internal gravity filter fallback when external gravity is not provided
    private var estGx: Double = 0.0
    private var estGy: Double = 9.80665
    private var estGz: Double = 0.0
    private var hasGravityInit: Boolean = false

    override fun reset() {
        currentState = UserMotionState.UNKNOWN
        confidence = 0.0
        totalStepCount = 0
        consecutiveSteps = 0
        lastStepTimestampNs = 0L
        lastAccelTimestampNs = 0L
        lastGyroTimestampNs = 0L
        smoothedDynAcc = 0.0
        prevSmoothed1 = 0.0
        prevSmoothed2 = 0.0
        motionEnergyEma = 0.0
        gyroMagEma = 0.0
        currentCadence = 0.0
        isLastSampleStep = false
        hasGravityInit = false
    }

    override fun processAccelerometerSample(
        ax: Double,
        ay: Double,
        az: Double,
        timestampNs: Long,
        gx: Double?,
        gy: Double?,
        gz: Double?
    ) {
        isLastSampleStep = false

        // Guard against NaN / Infinite / corrupt sensor readings
        if (ax.isNaN() || ay.isNaN() || az.isNaN() ||
            ax.isInfinite() || ay.isInfinite() || az.isInfinite()) {
            return
        }

        // Timing validation
        if (lastAccelTimestampNs > 0L) {
            val dtNs = timestampNs - lastAccelTimestampNs
            if (dtNs <= 0L) {
                // Ignore zero dt or out-of-order samples
                return
            }
            val dtSec = dtNs * 1e-9
            if (dtSec > 2.0) {
                // Large timestamp gap (sensor dropout or paused app) -> reset filters safely
                smoothedDynAcc = 0.0
                prevSmoothed1 = 0.0
                prevSmoothed2 = 0.0
                motionEnergyEma = 0.0
                consecutiveSteps = 0
            }
        }
        val dtSec = if (lastAccelTimestampNs > 0L) {
            min(0.50, max(0.001, (timestampNs - lastAccelTimestampNs) * 1e-9))
        } else {
            0.02
        }
        lastAccelTimestampNs = timestampNs

        // 1. Gravity Removal
        val effGx: Double
        val effGy: Double
        val effGz: Double

        if (gx != null && gy != null && gz != null &&
            !gx.isNaN() && !gy.isNaN() && !gz.isNaN() &&
            !gx.isInfinite() && !gy.isInfinite() && !gz.isInfinite()) {
            effGx = gx
            effGy = gy
            effGz = gz
        } else {
            // Self-contained exponential gravity filter
            if (!hasGravityInit) {
                estGx = ax
                estGy = ay
                estGz = az
                hasGravityInit = true
            } else {
                val alphaG = min(0.20, dtSec * 2.0)
                estGx = alphaG * ax + (1.0 - alphaG) * estGx
                estGy = alphaG * ay + (1.0 - alphaG) * estGy
                estGz = alphaG * az + (1.0 - alphaG) * estGz
            }
            effGx = estGx
            effGy = estGy
            effGz = estGz
        }

        // Linear dynamic acceleration
        val lax = ax - effGx
        val lay = ay - effGy
        val laz = az - effGz
        val dynAccMag = sqrt(lax * lax + lay * lay + laz * laz)

        // 2. Low-Pass Smoothing of Dynamic Acceleration
        // Time constant tau ~ 0.05s
        val alphaLp = 1.0 - exp(-dtSec / 0.05)
        prevSmoothed2 = prevSmoothed1
        prevSmoothed1 = smoothedDynAcc
        smoothedDynAcc = alphaLp * dynAccMag + (1.0 - alphaLp) * smoothedDynAcc

        // 3. Motion Energy Integration (RMS energy over window)
        val alphaEnergy = 1.0 - exp(-dtSec / 0.35)
        motionEnergyEma = alphaEnergy * (dynAccMag * dynAccMag) + (1.0 - alphaEnergy) * motionEnergyEma
        val currentMotionEnergy = sqrt(motionEnergyEma)

        // 4. Peak Detection with Refractory Constraint
        // A peak occurs when prevSmoothed1 is higher than prevSmoothed2 and smoothedDynAcc
        val isLocalPeak = (prevSmoothed1 > prevSmoothed2) && (prevSmoothed1 > smoothedDynAcc)
        val peakValue = prevSmoothed1

        val timeSinceLastStepSec = if (lastStepTimestampNs > 0L) {
            (timestampNs - lastStepTimestampNs) * 1e-9
        } else {
            99.0
        }

        if (isLocalPeak &&
            peakValue >= config.stepThresholdMps2 &&
            timeSinceLastStepSec >= config.refractoryPeriodSeconds
        ) {
            // Valid step candidate detected!
            totalStepCount++
            isLastSampleStep = true

            if (timeSinceLastStepSec <= config.maxStepIntervalSeconds) {
                consecutiveSteps++
                currentCadence = (60.0 / timeSinceLastStepSec).coerceIn(40.0, 220.0)
            } else {
                consecutiveSteps = 1
                currentCadence = 60.0
            }
            lastStepTimestampNs = timestampNs
        }

        // 5. State Machine Transition & Confidence Estimation
        updateMotionState(currentMotionEnergy, timeSinceLastStepSec, dtSec)
    }

    override fun processGyroSample(wx: Double, wy: Double, wz: Double, timestampNs: Long) {
        if (wx.isNaN() || wy.isNaN() || wz.isNaN() ||
            wx.isInfinite() || wy.isInfinite() || wz.isInfinite()) {
            return
        }

        val dtSec = if (lastGyroTimestampNs > 0L) {
            min(0.50, max(0.001, (timestampNs - lastGyroTimestampNs) * 1e-9))
        } else {
            0.02
        }
        lastGyroTimestampNs = timestampNs

        val gyroMag = sqrt(wx * wx + wy * wy + wz * wz)
        val alphaGyro = 1.0 - exp(-dtSec / 0.30)
        gyroMagEma = alphaGyro * gyroMag + (1.0 - alphaGyro) * gyroMagEma
    }

    private fun updateMotionState(motionEnergy: Double, timeSinceLastStepSec: Double, dtSec: Double) {
        if (lastAccelTimestampNs == 0L) {
            currentState = UserMotionState.UNKNOWN
            confidence = 0.0
            return
        }

        when (currentState) {
            UserMotionState.UNKNOWN -> {
                if (isLastSampleStep) {
                    currentState = UserMotionState.STARTING
                    confidence = 0.65
                } else if (motionEnergy < config.stationaryEnergyThreshold) {
                    currentState = UserMotionState.STATIONARY
                    confidence = 0.85
                } else {
                    currentState = UserMotionState.MOVING
                    confidence = 0.50
                }
            }

            UserMotionState.STATIONARY, UserMotionState.STOPPED -> {
                if (isLastSampleStep) {
                    // First valid step triggers STARTING
                    currentState = UserMotionState.STARTING
                    confidence = 0.70
                } else if (motionEnergy > config.activeEnergyThreshold && gyroMagEma > config.gyroStationaryThreshold) {
                    currentState = UserMotionState.MOVING
                    confidence = 0.55
                } else {
                    // High confidence stationary
                    val energyFactor = max(0.0, 1.0 - (motionEnergy / config.stationaryEnergyThreshold))
                    confidence = min(0.98, 0.75 + 0.23 * energyFactor)
                }
            }

            UserMotionState.STARTING -> {
                if (consecutiveSteps >= config.minStepsForWalking) {
                    // Confirmed periodic walking
                    currentState = UserMotionState.WALKING
                    confidence = 0.85
                } else if (timeSinceLastStepSec > config.stoppingTimeoutSeconds) {
                    if (motionEnergy < config.stationaryEnergyThreshold) {
                        currentState = UserMotionState.STATIONARY
                        confidence = 0.85
                        consecutiveSteps = 0
                    } else {
                        currentState = UserMotionState.MOVING
                        confidence = 0.50
                    }
                } else {
                    confidence = 0.70
                }
            }

            UserMotionState.WALKING -> {
                if (isLastSampleStep) {
                    // Step reinforces walking confidence
                    confidence = min(0.98, 0.85 + 0.03 * min(5, consecutiveSteps))
                } else if (timeSinceLastStepSec > config.stoppingTimeoutSeconds) {
                    // Steps ceased -> transition to STOPPING
                    currentState = UserMotionState.STOPPING
                    confidence = 0.70
                } else {
                    // Confidence decays slightly during interval between steps
                    confidence = max(0.70, confidence - 0.02 * dtSec)
                }
            }

            UserMotionState.STOPPING -> {
                if (isLastSampleStep) {
                    // User resumed walking
                    currentState = UserMotionState.WALKING
                    confidence = 0.85
                } else if (timeSinceLastStepSec > config.stationaryTimeoutSeconds && motionEnergy < config.activeEnergyThreshold) {
                    // Confirmed stop
                    currentState = UserMotionState.STATIONARY
                    confidence = 0.90
                    consecutiveSteps = 0
                } else if (motionEnergy > config.activeEnergyThreshold && gyroMagEma > config.gyroStationaryThreshold) {
                    currentState = UserMotionState.MOVING
                    confidence = 0.55
                } else {
                    confidence = 0.75
                }
            }

            UserMotionState.MOVING, UserMotionState.BACKING_AWAY -> {
                if (consecutiveSteps >= config.minStepsForWalking) {
                    currentState = UserMotionState.WALKING
                    confidence = 0.85
                } else if (isLastSampleStep) {
                    currentState = UserMotionState.STARTING
                    confidence = 0.70
                } else if (motionEnergy < config.stationaryEnergyThreshold && gyroMagEma < config.gyroStationaryThreshold) {
                    currentState = UserMotionState.STATIONARY
                    confidence = 0.85
                    consecutiveSteps = 0
                } else {
                    confidence = 0.60
                }
            }
        }
    }

    override fun getMotionEstimate(): UserMotionEstimate {
        val nowNs = lastAccelTimestampNs
        val timeSinceLastStep = if (lastStepTimestampNs > 0L) {
            max(0.0, (nowNs - lastStepTimestampNs) * 1e-9)
        } else {
            99.0
        }

        return UserMotionEstimate(
            state = currentState,
            confidence = confidence,
            stepCount = totalStepCount,
            timeSinceLastStepSeconds = timeSinceLastStep,
            motionEnergy = sqrt(motionEnergyEma),
            cadenceStepsPerMinute = currentCadence,
            timestampNs = nowNs,
            isStepCandidate = isLastSampleStep
        )
    }
}
