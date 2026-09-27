package com.prism.assistant.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Operating source state of the camera orientation estimator.
 */
enum class OrientationSourceState {
    IMU_FUSED,
    GRAVITY_ONLY,
    FALLBACK_CALIBRATED,
    UNAVAILABLE
}

/**
 * Operating mode for A/B testing: Fixed nominal vs Dynamic IMU-fused orientation.
 */
enum class OrientationMode {
    FIXED_ORIENTATION,
    IMU_FUSED_ORIENTATION
}

/**
 * Configurable physical camera mount calibration offsets.
 */
data class CameraMountCalibration(
    val pitchOffsetRadians: Double = 0.20944, // Nominal ~12.0 degrees downward tilt
    val rollOffsetRadians: Double = 0.0
)

/**
 * Estimated dynamic 3D camera orientation relative to local gravity horizontal plane.
 */
data class CameraOrientation(
    val pitchRadians: Double,
    val rollRadians: Double,
    val timestampNs: Long = 0L,
    val confidence: Double = 1.0,
    val sourceState: OrientationSourceState = OrientationSourceState.FALLBACK_CALIBRATED,
    val gyroRates: DoubleArray = doubleArrayOf(0.0, 0.0, 0.0),
    val gravityVector: DoubleArray = doubleArrayOf(0.0, 0.0, 9.81),
    val gravityMagnitude: Double = 9.81
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as CameraOrientation
        return pitchRadians == other.pitchRadians &&
                rollRadians == other.rollRadians &&
                timestampNs == other.timestampNs &&
                confidence == other.confidence &&
                sourceState == other.sourceState
    }

    override fun hashCode(): Int {
        var result = pitchRadians.hashCode()
        result = 31 * result + rollRadians.hashCode()
        result = 31 * result + timestampNs.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + sourceState.hashCode()
        return result
    }
}

/**
 * Interface for providing dynamic device and camera orientation.
 */
interface IDeviceOrientationProvider {
    var calibration: CameraMountCalibration
    var mode: OrientationMode
    fun start(): Boolean
    fun stop()
    fun getOrientation(timestampNs: Long = 0L): CameraOrientation
    fun getMotionEstimate(): UserMotionEstimate = UserMotionEstimate()
    fun updateMode(mode: OrientationMode) { this.mode = mode }
    fun updateCalibration(calibration: CameraMountCalibration) { this.calibration = calibration }
}

/**
 * Lightweight, deterministic IMU Complementary Sensor Fusion Engine.
 *
 * Fuses high-rate Gyroscope angular velocities for responsive short-term tilt tracking
 * with low-rate Gravity vectors for drift-free long-term reference, with graceful
 * fallback to nominal camera calibration when sensors are uncalibrated or unavailable.
 */
class OrientationFusion(
    private val sensorManager: SensorManager? = null,
    override var calibration: CameraMountCalibration = CameraMountCalibration(),
    override var mode: OrientationMode = OrientationMode.IMU_FUSED_ORIENTATION
) : IDeviceOrientationProvider, SensorEventListener {

    // Secondary constructor taking Android Context
    constructor(
        context: Context,
        calibration: CameraMountCalibration = CameraMountCalibration(),
        mode: OrientationMode = OrientationMode.IMU_FUSED_ORIENTATION
    ) : this(
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager,
        calibration = calibration,
        mode = mode
    )

    private var gravitySensor: Sensor? = null
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    val motionEstimator: UserMotionEstimator = UserMotionEstimator()

    // Internal filter state (device frame angles in radians)
    private var fusedPitch: Double = calibration.pitchOffsetRadians
    private var fusedRoll: Double = calibration.rollOffsetRadians
    private var lastSensorTimestampNs: Long = 0L
    private var lastGravityTimestampNs: Long = 0L
    private var lastGyroTimestampNs: Long = 0L

    // Current raw measurements
    private var rawGravity = doubleArrayOf(0.0, 9.81, 0.0)
    private var rawGyro = doubleArrayOf(0.0, 0.0, 0.0)
    private var gravityMag: Double = 9.81
    private var hasValidGravity: Boolean = false
    private var hasValidGyro: Boolean = false
    private var isRunning: Boolean = false

    init {
        gravitySensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        accelSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }

    override fun start(): Boolean {
        if (isRunning || sensorManager == null) return false
        var registeredAny = false

        gravitySensor?.let {
            val reg = sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            if (reg) registeredAny = true
        }
        if (accelSensor != null && accelSensor != gravitySensor) {
            val reg = sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_GAME)
            if (reg) registeredAny = true
        }
        gyroSensor?.let {
            val reg = sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            if (reg) registeredAny = true
        }

        isRunning = registeredAny
        return isRunning
    }

    override fun stop() {
        if (!isRunning) return
        sensorManager?.unregisterListener(this)
        isRunning = false
    }

    override fun getMotionEstimate(): UserMotionEstimate = motionEstimator.getMotionEstimate()

    /**
     * Direct update method for synthetic / playback unit testing.
     */
    fun processAccelerometerSample(ax: Double, ay: Double, az: Double, timestampNs: Long) {
        val gx = if (hasValidGravity) rawGravity[0] else null
        val gy = if (hasValidGravity) rawGravity[1] else null
        val gz = if (hasValidGravity) rawGravity[2] else null
        motionEstimator.processAccelerometerSample(ax, ay, az, timestampNs, gx, gy, gz)
    }

    /**
     * Direct update method for synthetic / playback unit testing.
     */
    fun processGravitySample(gx: Double, gy: Double, gz: Double, timestampNs: Long) {
        if (gx.isNaN() || gy.isNaN() || gz.isNaN() || gx.isInfinite() || gy.isInfinite() || gz.isInfinite()) return
        val mag = sqrt(gx * gx + gy * gy + gz * gz)
        if (mag < 1.0 || mag > 25.0) return // Reject wild abnormal vectors

        rawGravity[0] = gx
        rawGravity[1] = gy
        rawGravity[2] = gz
        gravityMag = mag
        hasValidGravity = true
        lastGravityTimestampNs = timestampNs

        // Calculate gravity pitch & roll in device portrait orientation
        // Portrait device: Y points up, X points right, Z points out
        // Downward tilt (pitch forward) causes Z to become positive (or Y to tilt forward)
        val gravPitch = atan2(gz, max(0.01, sqrt(gx * gx + gy * gy)))
        val gravRoll = atan2(-gx, max(0.01, gy))

        if (!hasValidGyro) {
            // Gravity only mode: apply exponential smoothing
            val alpha = 0.20
            fusedPitch = alpha * (calibration.pitchOffsetRadians + gravPitch) + (1.0 - alpha) * fusedPitch
            fusedRoll = alpha * (calibration.rollOffsetRadians + gravRoll) + (1.0 - alpha) * fusedRoll
        } else {
            // Slow gravity drift correction onto existing fused state
            val alpha = if (abs(mag - 9.81) < 1.2) 0.05 else 0.01
            val targetPitch = calibration.pitchOffsetRadians + gravPitch
            val targetRoll = calibration.rollOffsetRadians + gravRoll
            fusedPitch = (1.0 - alpha) * fusedPitch + alpha * targetPitch
            fusedRoll = (1.0 - alpha) * fusedRoll + alpha * targetRoll
        }

        lastSensorTimestampNs = timestampNs
    }

    /**
     * Direct update method for synthetic / playback unit testing.
     */
    fun processGyroSample(wx: Double, wy: Double, wz: Double, timestampNs: Long) {
        if (wx.isNaN() || wy.isNaN() || wz.isNaN() || wx.isInfinite() || wy.isInfinite() || wz.isInfinite()) return
        rawGyro[0] = wx
        rawGyro[1] = wy
        rawGyro[2] = wz

        if (lastGyroTimestampNs > 0L) {
            val dt = (timestampNs - lastGyroTimestampNs) * 1e-9
            if (dt in 0.0005..0.50) { // Valid time step between 0.5ms and 500ms
                // Portrait device: wx is pitch rate (tilt around X-axis), wy is roll rate (tilt around Y-axis)
                val dPitch = wx * dt
                val dRoll = wy * dt

                // Integrate high-frequency gyro motion
                fusedPitch += dPitch
                fusedRoll += dRoll
                hasValidGyro = true
            }
        }
        lastGyroTimestampNs = timestampNs
        lastSensorTimestampNs = timestampNs
        motionEstimator.processGyroSample(wx, wy, wz, timestampNs)
    }

    override fun getOrientation(timestampNs: Long): CameraOrientation {
        if (mode == OrientationMode.FIXED_ORIENTATION) {
            return CameraOrientation(
                pitchRadians = calibration.pitchOffsetRadians,
                rollRadians = calibration.rollOffsetRadians,
                timestampNs = timestampNs,
                confidence = 1.0,
                sourceState = OrientationSourceState.FALLBACK_CALIBRATED,
                gyroRates = rawGyro.clone(),
                gravityVector = rawGravity.clone(),
                gravityMagnitude = gravityMag
            )
        }

        val sourceState = when {
            hasValidGravity && hasValidGyro -> OrientationSourceState.IMU_FUSED
            hasValidGravity -> OrientationSourceState.GRAVITY_ONLY
            else -> OrientationSourceState.FALLBACK_CALIBRATED
        }

        // Calculate dynamic orientation confidence
        val magDev = abs(gravityMag - 9.81)
        val magConfidence = max(0.20, 1.0 - (magDev / 5.0))
        val sensorConfidence = when (sourceState) {
            OrientationSourceState.IMU_FUSED -> min(0.98, 0.80 + 0.18 * magConfidence)
            OrientationSourceState.GRAVITY_ONLY -> min(0.80, 0.50 + 0.30 * magConfidence)
            OrientationSourceState.FALLBACK_CALIBRATED -> 0.70
            OrientationSourceState.UNAVAILABLE -> 0.0
        }

        // Bound final pitch and roll to physical human walking envelopes
        val boundedPitch = fusedPitch.coerceIn(-0.35, 0.85) // -20 deg to +48 deg
        val boundedRoll = fusedRoll.coerceIn(-0.60, 0.60)   // -34 deg to +34 deg

        return CameraOrientation(
            pitchRadians = boundedPitch,
            rollRadians = boundedRoll,
            timestampNs = if (lastSensorTimestampNs > 0L) lastSensorTimestampNs else timestampNs,
            confidence = sensorConfidence,
            sourceState = sourceState,
            gyroRates = rawGyro.clone(),
            gravityVector = rawGravity.clone(),
            gravityMagnitude = gravityMag
        )
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return
        val ts = event.timestamp

        when (event.sensor.type) {
            Sensor.TYPE_GRAVITY -> {
                processGravitySample(
                    gx = event.values[0].toDouble(),
                    gy = event.values[1].toDouble(),
                    gz = event.values[2].toDouble(),
                    timestampNs = ts
                )
            }
            Sensor.TYPE_ACCELEROMETER -> {
                if (gravitySensor?.type == Sensor.TYPE_ACCELEROMETER) {
                    processGravitySample(
                        gx = event.values[0].toDouble(),
                        gy = event.values[1].toDouble(),
                        gz = event.values[2].toDouble(),
                        timestampNs = ts
                    )
                }
                processAccelerometerSample(
                    ax = event.values[0].toDouble(),
                    ay = event.values[1].toDouble(),
                    az = event.values[2].toDouble(),
                    timestampNs = ts
                )
            }
            Sensor.TYPE_GYROSCOPE -> {
                processGyroSample(
                    wx = event.values[0].toDouble(),
                    wy = event.values[1].toDouble(),
                    wz = event.values[2].toDouble(),
                    timestampNs = ts
                )
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No-op
    }
}
