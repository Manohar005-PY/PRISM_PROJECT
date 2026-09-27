package com.prism.assistant.spatial

import com.prism.assistant.accessibility.AudioPriority
import com.prism.assistant.warning.SpatialDirection
import com.prism.assistant.warning.WarningSeverity

/**
 * Functional category of spatial audio cue distinguishing hazard localization from evasion guidance.
 */
enum class SpatialCueType {
    HAZARD_LOCATION,          // Identifies where the obstacle is physically located
    RECOMMENDED_DIRECTION     // Reinforces the recommended evasion/steering direction
}

/**
 * Modes for multisensory output configuration and A/B experimental testing.
 */
enum class MultisensoryMode {
    SPEECH_ONLY,                      // Speech only (lowest sensory footprint)
    MODE_A_SPEECH_HAPTIC,             // Baseline: Speech + Haptic (Phase I/J)
    MODE_B_SPEECH_SPATIAL_HAPTIC      // Experimental: Speech + Spatial Audio + Haptic (Phase K)
}

/**
 * Dedicated spatial audio cue abstraction capturing direction, stereo panning,
 * urgency, severity, and functional cue type.
 */
data class SpatialCue(
    val direction: SpatialDirection,
    val pan: Double,                          // Stereo pan in [-1.0, 1.0] (-1.0 = full left, 0.0 = center, +1.0 = full right)
    val severity: WarningSeverity,
    val durationSeconds: Double = 0.25,        // Brief duration (150-300ms) to prevent masking ambient sounds
    val volume: Double = 0.80,                // Normalized amplitude in [0.0, 1.0]
    val priority: AudioPriority = AudioPriority.CAUTION,
    val cueType: SpatialCueType = SpatialCueType.HAZARD_LOCATION,
    val frequencyHz: Int = 800                // Audio tone frequency in Hz
) {
    companion object {
        /**
         * Maps a SpatialDirection sector to a normalized stereo pan value in [-1.0, 1.0].
         * Enforces exact center (0.0) for CENTER within deadband.
         */
        fun directionToPan(dir: SpatialDirection): Double {
            return when (dir) {
                SpatialDirection.FAR_LEFT -> -1.0
                SpatialDirection.LEFT -> -0.65
                SpatialDirection.SLIGHTLY_LEFT -> -0.35
                SpatialDirection.CENTER -> 0.0
                SpatialDirection.SLIGHTLY_RIGHT -> 0.35
                SpatialDirection.RIGHT -> 0.65
                SpatialDirection.FAR_RIGHT -> 1.0
            }
        }

        /**
         * Computes stereo channel gains using equal-power / linear panning law.
         * pan: -1.0 (left only) to +1.0 (right only).
         * returns Pair(leftGain, rightGain) with values in [0.0, 1.0].
         */
        fun computeStereoGains(pan: Double, volume: Double = 1.0): Pair<Double, Double> {
            val clampedPan = pan.coerceIn(-1.0, 1.0)
            val leftGain = ((1.0 - clampedPan) / 2.0).coerceIn(0.0, 1.0) * volume.coerceIn(0.0, 1.0)
            val rightGain = ((1.0 + clampedPan) / 2.0).coerceIn(0.0, 1.0) * volume.coerceIn(0.0, 1.0)
            return Pair(leftGain, rightGain)
        }
    }
}
