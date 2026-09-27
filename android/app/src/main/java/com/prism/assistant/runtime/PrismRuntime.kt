package com.prism.assistant.runtime

import android.content.Context
import com.prism.assistant.bridge.DeterministicPrismBridge
import com.prism.assistant.bridge.IPrismCoreBridge
import com.prism.assistant.bridge.PrismFrameResult
import com.prism.assistant.bridge.PrismObservation
import com.prism.assistant.presentation.AndroidHapticChannel
import com.prism.assistant.presentation.AndroidSpatialAudioChannel
import com.prism.assistant.presentation.AndroidSpeechChannel
import com.prism.assistant.presentation.AndroidVisualChannel
import com.prism.assistant.telemetry.AndroidTelemetryServer
import com.prism.assistant.telemetry.AndroidTelemetrySink
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Main Android host runtime managing lifecycle, threading, PRISM execution, and channel dispatch.
 */
class PrismRuntime(
    private val context: Context,
    private val bridge: IPrismCoreBridge = DeterministicPrismBridge(confirmationHits = 2)
) {
    private val scope = CoroutineScope(Dispatchers.Default)
    private var loopJob: Job? = null
    private val isRunningFlag = AtomicBoolean(false)

    var state: RuntimeState = RuntimeState.STOPPED
        private set

    var executionMode: ExecutionMode = ExecutionMode.LIVE
        private set

    // Presentation Channels & Telemetry
    val speechChannel = AndroidSpeechChannel(context)
    val spatialAudioChannel = AndroidSpatialAudioChannel()
    val hapticChannel = AndroidHapticChannel(context)
    val visualChannel = AndroidVisualChannel { banner, prio ->
        onVisualUpdateListener?.invoke(banner, prio)
    }
    val telemetrySink = AndroidTelemetrySink(context)
    val telemetryServer = AndroidTelemetryServer(context, telemetrySink, port = 8080)
    val accessibilityController = com.prism.assistant.accessibility.AccessibilityController(speechChannel, hapticChannel)
    val orientationFusion = com.prism.assistant.sensor.OrientationFusion(context)

    var latestFrameResult: PrismFrameResult? = null
        private set

    // Callbacks & Observers
    var onFrameResultListener: ((PrismFrameResult) -> Unit)? = null
    var onVisualUpdateListener: ((banner: String, priority: String) -> Unit)? = null
    var onStateChangeListener: ((RuntimeState) -> Unit)? = null

    fun setMode(mode: ExecutionMode) {
        if (state == RuntimeState.RUNNING) return
        executionMode = mode
    }

    fun start() {
        if (isRunningFlag.getAndSet(true)) return

        updateState(RuntimeState.INITIALIZING)
        speechChannel.initialize()
        spatialAudioChannel.initialize()
        orientationFusion.start()
        val initSuccess = bridge.initialize()
        if (!initSuccess) {
            updateState(RuntimeState.ERROR)
            isRunningFlag.set(false)
            return
        }

        updateState(RuntimeState.RUNNING)
        telemetryServer.start()

        if (executionMode == ExecutionMode.MOCK || executionMode == ExecutionMode.REPLAY) {
            startMockLoop()
        }
    }

    fun stop() {
        if (!isRunningFlag.getAndSet(false)) return

        updateState(RuntimeState.STOPPING)
        telemetryServer.stop()
        orientationFusion.stop()
        loopJob?.cancel()
        loopJob = null

        speechChannel.shutdown()
        spatialAudioChannel.shutdown()
        hapticChannel.cancel()
        bridge.reset()

        updateState(RuntimeState.STOPPED)
    }

    /**
     * Process an incoming frame of observations.
     * Can be invoked from CameraX analyzer or mock/replay generator.
     */
    fun processFrame(timestamp: Double, observations: List<PrismObservation>) {
        if (state != RuntimeState.RUNNING && state != RuntimeState.DEGRADED) return

        try {
            // 1. Process through PRISM Core Bridge with dynamic orientation & motion
            val orientation = orientationFusion.getOrientation()
            val motionEstimate = orientationFusion.getMotionEstimate()
            val result = bridge.processFrame(timestamp, observations, motionEstimate)
            latestFrameResult = result

            // 2. Dispatch Presentation Cues with complete Failure Isolation
            for (cue in result.cues) {
                try {
                    when (cue.channel) {
                        "speech" -> speechChannel.deliver(cue)
                        "spatial_audio" -> spatialAudioChannel.deliver(cue)
                        "haptic" -> hapticChannel.deliver(cue)
                        "visual" -> visualChannel.deliver(cue)
                    }
                } catch (chEx: Exception) {
                    // Isolated channel failure
                }
            }

            // 3. Record Telemetry
            telemetrySink.recordFrame(result, orientation)

            // 4. Notify UI / Observers
            onFrameResultListener?.invoke(result)

        } catch (ex: Exception) {
            updateState(RuntimeState.DEGRADED)
        }
    }

    private fun startMockLoop() {
        loopJob = scope.launch {
            var simTime = 0.0
            while (isActive && isRunningFlag.get()) {
                val cycleTime = simTime % 8.0
                val obsList = mutableListOf<PrismObservation>()

                if (cycleTime < 6.0) {
                    // Entity 1: Center approaching pedestrian
                    val z1 = 5.2 - (cycleTime * 0.75)
                    if (z1 > 0.6) {
                        obsList.add(PrismObservation(x = 0.1, z = z1, covariance = 0.05, className = "person"))
                    }

                    // Entity 2: Right side stationary / slow obstacle
                    if (cycleTime < 5.0) {
                        obsList.add(PrismObservation(x = 1.2, z = 3.5, covariance = 0.06, className = "obstacle"))
                    }

                    // Entity 3: Crossing pedestrian moving from left (-2.0m) to right (+1.5m)
                    if (cycleTime in 1.0..5.5) {
                        val crossX = -2.0 + ((cycleTime - 1.0) * 0.7)
                        obsList.add(PrismObservation(x = crossX, z = 2.8, covariance = 0.05, className = "person"))
                    }
                }

                processFrame(simTime, obsList)
                simTime += 0.033 // 30 FPS
                delay(33)
            }
        }
    }

    private fun updateState(newState: RuntimeState) {
        state = newState
        onStateChangeListener?.invoke(newState)
    }

    fun getHealth(): RuntimeHealth {
        return RuntimeHealth(
            status = state.name,
            captureFps = 30.0,
            processedFps = 30.0,
            droppedFramesCount = 0,
            averageLatencyMs = 0.5,
            activeTrackCount = 1,
            isCameraHealthy = true,
            isPresentationHealthy = speechChannel.isAvailable()
        )
    }
}
