package com.prism.assistant.presentation

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.prism.assistant.bridge.PrismPresentationCue

/**
 * Haptic presentation channel adapting Android Vibrator hardware with pulse cooldown.
 */
class AndroidHapticChannel(context: Context) {

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
        manager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    private var lastVibrateTimeMs: Long = 0L

    fun isAvailable(): Boolean = vibrator?.hasVibrator() == true

    fun deliver(cue: PrismPresentationCue): Boolean {
        if (cue.channel != "haptic" || cue.hapticPattern == null) return false
        if (vibrator == null || !vibrator.hasVibrator()) return false

        val now = System.currentTimeMillis()
        val minIntervalMs = if (cue.priority == "emergency") 500L else 1000L
        if (now - lastVibrateTimeMs < minIntervalMs) return false

        lastVibrateTimeMs = now

        val pattern = cue.hapticPattern
        val timings: LongArray = when (pattern.lowercase()) {
            "warning" -> longArrayOf(0, 150)
            "slow" -> longArrayOf(0, 200, 100, 200)
            "left" -> longArrayOf(0, 100, 50, 250)
            "right" -> longArrayOf(0, 250, 50, 100)
            "stop" -> longArrayOf(0, 400)
            "emergency" -> longArrayOf(0, 350, 100, 350)
            else -> longArrayOf(0, 100)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                val effect = VibrationEffect.createWaveform(timings, -1)
                vibrator.vibrate(effect)
                return true
            } catch (_: Exception) {
                // Fallback
            }
        }

        @Suppress("DEPRECATION")
        vibrator.vibrate(timings, -1)
        return true
    }

    fun cancel() {
        vibrator?.cancel()
    }
}
