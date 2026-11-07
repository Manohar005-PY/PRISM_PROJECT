package com.prism.assistant

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.prism.assistant.accessibility.AccessibilityController
import com.prism.assistant.accessibility.ReleaseMode
import com.prism.assistant.camera.AndroidCameraFrameSource
import com.prism.assistant.camera.AndroidPerceptionAnalyzer
import com.prism.assistant.runtime.ExecutionMode
import com.prism.assistant.runtime.PrismRuntime
import com.prism.assistant.runtime.RuntimeState
import com.prism.assistant.ui.PrismDebugScreen

class MainActivity : AppCompatActivity() {

    private lateinit var runtime: PrismRuntime
    private lateinit var cameraSource: AndroidCameraFrameSource
    private lateinit var perceptionAnalyzer: AndroidPerceptionAnalyzer
    private lateinit var debugScreen: PrismDebugScreen
    private lateinit var previewView: PreviewView

    private lateinit var participantCardLayout: LinearLayout
    private lateinit var debugHudLayout: LinearLayout
    private lateinit var tvParticipantGuidance: TextView
    private lateinit var tvParticipantStatus: TextView

    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    private var userStoppedManually: Boolean = false

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCameraAndRuntime()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        runtime = (application as PrismApplication).runtime
        cameraSource = AndroidCameraFrameSource(this)
        perceptionAnalyzer = AndroidPerceptionAnalyzer()

        previewView = findViewById(R.id.cameraPreview)
        participantCardLayout = findViewById(R.id.participantCardLayout)
        debugHudLayout = findViewById(R.id.debugHudLayout)

        tvParticipantGuidance = findViewById(R.id.tvParticipantGuidance)
        tvParticipantStatus = findViewById(R.id.tvParticipantStatus)

        val tvCameraStatus = findViewById<TextView>(R.id.tvCameraStatus)
        val tvState = findViewById<TextView>(R.id.tvRuntimeState)
        val tvFps = findViewById<TextView>(R.id.tvFpsAndLatency)
        val tvDecision = findViewById<TextView>(R.id.tvDecision)
        val tvRisk = findViewById<TextView>(R.id.tvRisk)
        val tvObstacleDetails = findViewById<TextView>(R.id.tvObstacleDetails)
        val tvServerUrl = findViewById<TextView>(R.id.tvServerUrl)

        debugScreen = PrismDebugScreen(
            tvCameraStatus,
            tvState,
            tvFps,
            tvDecision,
            tvRisk,
            tvObstacleDetails,
            tvServerUrl
        )

        // Show live telemetry server IP & port
        val ip = runtime.telemetryServer.getDeviceIpAddress()
        debugScreen.updateServerUrl(ip, runtime.telemetryServer.port)

        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)

        val accessibilityController = runtime.accessibilityController

        // Initialize UI mode
        updateReleaseModeUI(accessibilityController.releaseMode)

        // START button handler
        btnStart.setOnClickListener {
            userStoppedManually = false
            if (runtime.state != RuntimeState.RUNNING) {
                startCameraAndRuntime()
            }
        }

        // STOP button handler
        btnStop.setOnClickListener {
            userStoppedManually = true
            if (runtime.state == RuntimeState.RUNNING) {
                cameraSource.stopCamera()
                runtime.stop()
            }
        }

        runtime.onFrameResultListener = { result ->
            runOnUiThread {
                debugScreen.updateResult(result, runtime.executionMode == ExecutionMode.LIVE)

                // Update accessible participant card guidance
                val action = result.decision.action.lowercase()
                val guidanceText = when (action) {
                    "stop", "emergency_stop" -> "STOP! Obstacle directly in path."
                    "slow_down", "caution" -> "SLOW DOWN: Obstacle nearby."
                    "turn_left" -> "VEER LEFT: Clear path on left."
                    "turn_right" -> "VEER RIGHT: Clear path on right."
                    else -> "Proceed forward along corridor."
                }
                tvParticipantGuidance.text = guidanceText
                tvParticipantGuidance.contentDescription = "Live guidance: $guidanceText"

                val color = when (action) {
                    "stop", "emergency_stop" -> Color.parseColor("#FF1744")
                    "slow_down", "caution", "turn_left", "turn_right" -> Color.parseColor("#FFD600")
                    else -> Color.parseColor("#00E676")
                }
                tvParticipantGuidance.setTextColor(color)
            }
        }

        runtime.onStateChangeListener = { state ->
            runOnUiThread {
                debugScreen.updateState(state)
                val isRunning = (state == RuntimeState.RUNNING)
                btnStart.isEnabled = !isRunning
                btnStop.isEnabled = isRunning
                btnStart.alpha = if (isRunning) 0.5f else 1.0f
                btnStop.alpha = if (isRunning) 1.0f else 0.5f
                tvParticipantStatus.text = "Status: PRISM ${state.name} | Sensors Operational"
                tvParticipantStatus.contentDescription = "System status: PRISM ${state.name}, Sensors Operational."
            }
        }

        checkAndRequestPermissions()
    }

    private fun updateReleaseModeUI(mode: ReleaseMode) {
        if (mode == ReleaseMode.PARTICIPANT_STUDY) {
            debugHudLayout.visibility = View.GONE
            participantCardLayout.visibility = View.VISIBLE
        } else {
            debugHudLayout.visibility = View.VISIBLE
            participantCardLayout.visibility = View.VISIBLE
        }
    }

    private fun checkAndRequestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCameraAndRuntime()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCameraAndRuntime() {
        runtime.start()
        val ip = runtime.telemetryServer.getDeviceIpAddress()
        debugScreen.updateServerUrl(ip, runtime.telemetryServer.port)

        if (runtime.executionMode == ExecutionMode.LIVE) {
            cameraSource.startCamera(this, previewView) { imageProxy, meta ->
                runOnUiThread {
                    debugScreen.updateCameraDiagnostics(meta)
                }
                val observations = perceptionAnalyzer.analyzeFrame(imageProxy, meta.timestamp)
                runtime.processFrame(meta.timestamp, observations)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (!userStoppedManually && runtime.state != RuntimeState.RUNNING && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCameraAndRuntime()
        }
    }

    override fun onPause() {
        super.onPause()
        cameraSource.stopCamera()
        runtime.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraSource.stopCamera()
        runtime.stop()
    }
}
