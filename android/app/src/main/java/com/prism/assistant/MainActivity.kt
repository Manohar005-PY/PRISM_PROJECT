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
    private lateinit var tvParticipantDestination: TextView
    private lateinit var tvParticipantGuidance: TextView
    private lateinit var tvParticipantStatus: TextView

    private lateinit var btnWhereAmI: Button
    private lateinit var btnWhatIsAhead: Button
    private lateinit var btnRepeatInstruction: Button
    private lateinit var btnToggleRuntime: Button
    private lateinit var btnSelectDestination: Button
    private lateinit var btnReleaseMode: Button

    private var destinationIndex: Int = 0

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

        tvParticipantDestination = findViewById(R.id.tvParticipantDestination)
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

        btnWhereAmI = findViewById(R.id.btnWhereAmI)
        btnWhatIsAhead = findViewById(R.id.btnWhatIsAhead)
        btnRepeatInstruction = findViewById(R.id.btnRepeatInstruction)
        btnToggleRuntime = findViewById(R.id.btnToggleRuntime)
        btnSelectDestination = findViewById(R.id.btnSelectDestination)
        btnReleaseMode = findViewById(R.id.btnReleaseMode)

        val accessibilityController = runtime.accessibilityController

        // Initialize UI mode
        updateReleaseModeUI(accessibilityController.releaseMode)

        // Initialize Destination text
        accessibilityController.selectedDestination?.let { dest ->
            tvParticipantDestination.text = "DESTINATION: ${dest.name}"
            tvParticipantDestination.contentDescription = "Current destination: ${dest.name}"
        }

        // Accessible query button handlers
        btnWhereAmI.setOnClickListener {
            accessibilityController.queryWhereAmI()
        }

        btnWhatIsAhead.setOnClickListener {
            accessibilityController.queryWhatIsAhead(runtime.latestFrameResult)
        }

        btnRepeatInstruction.setOnClickListener {
            accessibilityController.repeatLastInstruction()
        }

        // Destination selection flow
        btnSelectDestination.setOnClickListener {
            val list = accessibilityController.destinations
            if (list.isNotEmpty()) {
                destinationIndex = (destinationIndex + 1) % list.size
                val chosen = list[destinationIndex]
                accessibilityController.selectDestination(chosen.id)
                tvParticipantDestination.text = "DESTINATION: ${chosen.name}"
                tvParticipantDestination.contentDescription = "Current destination: ${chosen.name}"
            }
        }

        // Release mode toggle
        btnReleaseMode.setOnClickListener {
            val nextMode = if (accessibilityController.releaseMode == ReleaseMode.PARTICIPANT_STUDY) {
                ReleaseMode.RESEARCH_DEBUG
            } else {
                ReleaseMode.PARTICIPANT_STUDY
            }
            accessibilityController.releaseMode = nextMode
            updateReleaseModeUI(nextMode)
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
                btnToggleRuntime.text = if (state == RuntimeState.RUNNING) "PAUSE" else "START"
                tvParticipantStatus.text = "Status: PRISM ${state.name} | Sensors Operational"
                tvParticipantStatus.contentDescription = "System status: PRISM ${state.name}, Sensors Operational."
            }
        }

        btnToggleRuntime.setOnClickListener {
            if (runtime.state == RuntimeState.RUNNING) {
                cameraSource.stopCamera()
                runtime.stop()
            } else {
                startCameraAndRuntime()
            }
        }

        checkAndRequestPermissions()
    }

    private fun updateReleaseModeUI(mode: ReleaseMode) {
        if (mode == ReleaseMode.PARTICIPANT_STUDY) {
            debugHudLayout.visibility = View.GONE
            participantCardLayout.visibility = View.VISIBLE
            btnReleaseMode.text = "MODE: STUDY"
        } else {
            debugHudLayout.visibility = View.VISIBLE
            participantCardLayout.visibility = View.VISIBLE
            btnReleaseMode.text = "MODE: DEBUG"
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
        if (runtime.state != RuntimeState.RUNNING && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
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
