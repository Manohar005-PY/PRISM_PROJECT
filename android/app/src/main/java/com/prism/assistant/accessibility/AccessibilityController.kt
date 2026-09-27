package com.prism.assistant.accessibility

import com.prism.assistant.bridge.PrismFrameResult
import com.prism.assistant.bridge.PrismPresentationCue
import com.prism.assistant.presentation.AndroidHapticChannel
import com.prism.assistant.presentation.AndroidSpatialAudioChannel
import com.prism.assistant.presentation.AndroidSpeechChannel
import com.prism.assistant.sensor.OrientationSourceState
import com.prism.assistant.spatial.MultisensoryMode
import com.prism.assistant.spatial.SpatialCue

/**
 * Release operating modes distinguishing engineering diagnostics from participant study interaction.
 */
enum class ReleaseMode {
    RESEARCH_DEBUG,
    PARTICIPANT_STUDY
}

/**
 * Hierarchical audio priority categories ensuring safety critical alerts preempt routine guidance.
 */
enum class AudioPriority(val rank: Int) {
    SYSTEM_INFO(1),      // Informational status / queries
    ADVISORY(2),         // Low-risk environmental advisories
    NAVIGATION(3),       // Routine turn-by-turn routing instructions
    AWARENESS(4),        // Low-risk object awareness
    CAUTION(5),          // Moderate caution advisories
    DIRECTIONAL(6),      // Directional evasion recommendations
    SAFETY_WARNING(7),   // Urgent evasion / slow down warnings
    EMERGENCY(8)         // Immediate safety stop (highest rank, preempts everything)
}

/**
 * Recognizable navigation instruction lifecycle states for blind and visually impaired users.
 */
enum class NavigationState {
    STARTING,
    NAVIGATING,
    UPCOMING_TURN,
    TURN_NOW,
    OFF_ROUTE,
    RECALCULATING,
    DESTINATION_REACHED,
    PAUSED,
    STOPPED_FOR_HAZARD,
    ERROR
}

/**
 * User-facing safe degradation states communicated in human-readable language.
 */
enum class SafeDegradationState {
    FULL,
    DEGRADED_ORIENTATION,
    DEGRADED_SENSOR,
    DEGRADED_PERCEPTION,
    PAUSED,
    UNAVAILABLE
}

/**
 * Structured accessible destination for transit and indoor environments.
 */
data class NavigationDestination(
    val id: String,
    val name: String,
    val building: String,
    val floor: String,
    val description: String
)

/**
 * Frozen baseline configuration metadata for PRISM-H1.
 */
object PrismBaselineConfig {
    const val RELEASE_VERSION = "PRISM-H1.0.0-RELEASE"
    const val CAMERA_HEIGHT_METERS = 1.30
    const val NOMINAL_PITCH_RADIANS = 0.20944 // 12.0 deg
    const val MIN_PHYSICAL_HEIGHT_M = 0.32
    const val MIN_VOLUMETRIC_HEIGHT_M = 0.22
    const val MAX_PHYSICAL_HEIGHT_M = 2.40
    const val MIN_DEPTH_M = 0.40
    const val MAX_DEPTH_M = 6.50
    const val MAX_LATERAL_EXTENT_M = 1.25
    const val CONFIRMATION_HITS = 2
    const val EMERGENCY_STOP_THRESHOLD_M = 1.00
    const val AUDIO_DUCKING_ENABLED = true
}

/**
 * End-to-End Accessibility & Speech/Haptic Priority Controller.
 * Manages TalkBack announcements, audio priority arbitration, concise speech generation,
 * destination selection flow, and non-visual status queries.
 */
class AccessibilityController(
    private val speechChannel: AndroidSpeechChannel? = null,
    private val hapticChannel: AndroidHapticChannel? = null,
    private val spatialAudioChannel: AndroidSpatialAudioChannel? = null
) {
    var releaseMode: ReleaseMode = ReleaseMode.PARTICIPANT_STUDY
    var navigationState: NavigationState = NavigationState.NAVIGATING
    var safeDegradationState: SafeDegradationState = SafeDegradationState.FULL
    var selectedDestination: NavigationDestination? = null
    var multisensoryMode: MultisensoryMode = MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC

    private var lastGuidanceInstruction: String = "Proceed forward along corridor."
    private var lastSpokenPriority: AudioPriority = AudioPriority.SYSTEM_INFO
    private var lastSpeechTimestampMs: Long = 0L

    // Accessible destination catalog for transit/academic test environments
    val destinations = listOf(
        NavigationDestination("D1", "North Concourse Exit", "Transit Station", "Ground Floor", "Street level access and bus connections"),
        NavigationDestination("D2", "Platform 1 & 2 (Westbound)", "Transit Station", "Sub-Level 1", "Train platforms via central elevator"),
        NavigationDestination("D3", "Elevator to All Platforms", "Transit Station", "Ground Floor", "East corridor near ticket machines"),
        NavigationDestination("D4", "Customer Assistance Desk", "Transit Station", "Ground Floor", "Central hall next to turnstiles"),
        NavigationDestination("D5", "Accessible Restroom", "Transit Station", "Ground Floor", "South concourse near information board")
    )

    init {
        selectedDestination = destinations.first()
    }

    /**
     * Priority-based speech dispatcher with emergency preemption and audio ducking.
     */
    fun dispatchSpeech(
        text: String,
        priority: AudioPriority,
        forceImmediate: Boolean = false
    ): Boolean {
        val now = System.currentTimeMillis()

        // 1. Interruption & Priority Arbitration Policy
        // If an EMERGENCY announcement is active or recent (<1.0s), never allow lower priority to interrupt
        if (lastSpokenPriority == AudioPriority.EMERGENCY && priority.rank < AudioPriority.EMERGENCY.rank) {
            if (now - lastSpeechTimestampMs < 1000L) {
                return false
            }
        }

        // Lower priority cannot interrupt higher priority
        if (priority.rank < lastSpokenPriority.rank && (now - lastSpeechTimestampMs < 1500L) && !forceImmediate) {
            return false
        }

        val prioTag = when (priority) {
            AudioPriority.EMERGENCY -> "emergency"
            AudioPriority.SAFETY_WARNING -> "urgent"
            AudioPriority.DIRECTIONAL -> "urgent"
            AudioPriority.CAUTION -> "warning"
            AudioPriority.NAVIGATION -> "warning"
            AudioPriority.AWARENESS -> "info"
            AudioPriority.ADVISORY -> "info"
            AudioPriority.SYSTEM_INFO -> "info"
        }

        val cue = PrismPresentationCue(
            channel = "speech",
            priority = prioTag,
            messageId = "acc:${priority.name.lowercase()}",
            text = text,
            direction = null,
            intensity = if (priority == AudioPriority.EMERGENCY) 1.0 else 0.75,
            duration = 1.0,
            hapticPattern = null
        )

        val delivered = speechChannel?.deliver(cue) ?: true
        if (delivered) {
            lastSpokenPriority = priority
            lastSpeechTimestampMs = now
            if (priority == AudioPriority.NAVIGATION) {
                lastGuidanceInstruction = text
            }
        }
        return delivered
    }

    /**
     * Dispatches distinct, recognizable haptic vibration patterns.
     */
    fun dispatchHaptic(pattern: String, priority: String = "info"): Boolean {
        val cue = PrismPresentationCue(
            channel = "haptic",
            priority = priority,
            messageId = "haptic:$pattern",
            text = null,
            direction = null,
            intensity = if (priority == "emergency") 1.0 else 0.7,
            duration = 0.5,
            hapticPattern = pattern
        )
        return hapticChannel?.deliver(cue) ?: true
    }

    /**
     * Changes the multisensory output mode with accessible confirmation speech.
     */
    fun updateMultisensoryMode(mode: MultisensoryMode) {
        this.multisensoryMode = mode
        val desc = when (mode) {
            MultisensoryMode.SPEECH_ONLY -> "Audio mode: Speech only."
            MultisensoryMode.MODE_A_SPEECH_HAPTIC -> "Audio mode: Speech and haptics."
            MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC -> "Audio mode: Speech, spatial audio, and haptics."
        }
        dispatchSpeech(desc, AudioPriority.SYSTEM_INFO, forceImmediate = true)
    }

    /**
     * Dispatches spatial audio cues respecting multisensory mode and emergency priority.
     */
    fun dispatchSpatialCue(cue: SpatialCue): Boolean {
        if (multisensoryMode != MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC) return false
        val now = System.currentTimeMillis()
        if (lastSpokenPriority == AudioPriority.EMERGENCY && cue.priority.rank < AudioPriority.EMERGENCY.rank) {
            if (now - lastSpeechTimestampMs < 1000L) {
                return false
            }
        }
        val prioTag = when (cue.priority) {
            AudioPriority.EMERGENCY -> "emergency"
            AudioPriority.SAFETY_WARNING, AudioPriority.DIRECTIONAL -> "urgent"
            AudioPriority.CAUTION -> "warning"
            else -> "info"
        }
        val presCue = PrismPresentationCue(
            channel = "spatial_audio",
            priority = prioTag,
            messageId = "spatial:${cue.cueType.name.lowercase()}:${cue.direction.name.lowercase()}",
            text = null,
            direction = cue.pan,
            intensity = cue.volume,
            duration = cue.durationSeconds,
            hapticPattern = null,
            spatialCue = cue
        )
        return spatialAudioChannel?.deliver(presCue) ?: true
    }

    /**
     * Accessible destination selection flow with audio/haptic confirmation.
     */
    fun selectDestination(destId: String): String {
        val target = destinations.find { it.id == destId } ?: destinations.first()
        selectedDestination = target
        navigationState = NavigationState.STARTING

        val msg = "Destination set to ${target.name}, ${target.floor}. Starting navigation."
        dispatchSpeech(msg, AudioPriority.NAVIGATION, forceImmediate = true)
        dispatchHaptic("slow", priority = "warning")
        return msg
    }

    /**
     * Query: "Where am I?" — concise landmark localization query for BVI users.
     */
    fun queryWhereAmI(): String {
        val destName = selectedDestination?.name ?: "Destination"
        val statusText = when (navigationState) {
            NavigationState.NAVIGATING -> "Main corridor heading toward $destName. 25 meters remaining."
            NavigationState.UPCOMING_TURN -> "Approaching corridor junction. Turn coming up."
            NavigationState.TURN_NOW -> "At the intersection. Turn now."
            NavigationState.OFF_ROUTE -> "Off route. Please pause while recalculating."
            NavigationState.DESTINATION_REACHED -> "Arrived at $destName."
            NavigationState.PAUSED -> "Navigation paused in main corridor."
            NavigationState.STOPPED_FOR_HAZARD -> "Stopped for obstacle. Path currently obstructed."
            else -> "In transit concourse."
        }
        dispatchSpeech(statusText, AudioPriority.SYSTEM_INFO, forceImmediate = true)
        return statusText
    }

    /**
     * Updates navigation state in response to safety emergency stops and hazard clearance.
     * Distinct from user-initiated pauses.
     */
    fun setSafetyStop(active: Boolean) {
        if (active) {
            if (navigationState != NavigationState.PAUSED) {
                navigationState = NavigationState.STOPPED_FOR_HAZARD
            }
        } else {
            if (navigationState == NavigationState.STOPPED_FOR_HAZARD) {
                navigationState = NavigationState.NAVIGATING
            }
        }
    }

    /**
     * Query: "What is ahead?" — concise threat and spatial clearance query.
     */
    fun queryWhatIsAhead(latestResult: PrismFrameResult?): String {
        val response = if (latestResult == null || latestResult.activeTracks == 0) {
            "Path ahead is clear."
        } else {
            val dec = latestResult.decision
            when (dec.action.lowercase()) {
                "emergency_stop" -> "Obstacle directly ahead. Stop."
                "turn_left" -> "Obstacle ahead on right. Clear to turn left."
                "turn_right" -> "Obstacle ahead on left. Clear to turn right."
                "slow_down" -> "Obstacle ahead. Slow down."
                else -> {
                    val threat = latestResult.risk
                    if (threat.riskScore > 30.0) {
                        "Obstacle detected at ${String.format("%.1f", threat.cpaDistance ?: 2.0)} meters."
                    } else {
                        "Corridor clear."
                    }
                }
            }
        }
        dispatchSpeech(response, AudioPriority.SYSTEM_INFO, forceImmediate = true)
        return response
    }

    /**
     * Query: Repeat last navigation instruction.
     */
    fun repeatLastInstruction(): String {
        dispatchSpeech(lastGuidanceInstruction, AudioPriority.NAVIGATION, forceImmediate = true)
        return lastGuidanceInstruction
    }

    /**
     * Translates system sensor and perception health into human-understandable guidance.
     */
    fun updateSystemHealthState(
        sourceState: OrientationSourceState,
        cameraHealthy: Boolean,
        pipelineRunning: Boolean
    ): SafeDegradationState {
        val newState = when {
            !pipelineRunning -> SafeDegradationState.PAUSED
            !cameraHealthy -> SafeDegradationState.UNAVAILABLE
            sourceState == OrientationSourceState.UNAVAILABLE -> SafeDegradationState.DEGRADED_SENSOR
            sourceState == OrientationSourceState.FALLBACK_CALIBRATED -> SafeDegradationState.DEGRADED_ORIENTATION
            else -> SafeDegradationState.FULL
        }

        if (newState != safeDegradationState) {
            safeDegradationState = newState
            val msg = when (newState) {
                SafeDegradationState.FULL -> "PRISM fully operational."
                SafeDegradationState.DEGRADED_ORIENTATION -> "Motion sensor unavailable. Continuing with calibrated orientation."
                SafeDegradationState.DEGRADED_SENSOR -> "Camera view reduced. Please keep phone upright."
                SafeDegradationState.DEGRADED_PERCEPTION -> "Floor detection uncertain. Proceed with caution."
                SafeDegradationState.PAUSED -> "Navigation paused."
                SafeDegradationState.UNAVAILABLE -> "Sensor error. System unavailable."
            }
            dispatchSpeech(msg, AudioPriority.SYSTEM_INFO)
        }

        return newState
    }
}
