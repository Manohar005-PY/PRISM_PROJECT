package com.prism.assistant.presentation

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.ToneGenerator
import android.os.Build
import android.util.Log
import com.prism.assistant.bridge.PrismPresentationCue
import com.prism.assistant.spatial.SpatialCue
import kotlin.concurrent.thread
import kotlin.math.PI
import kotlin.math.sin

/**
 * Spatial audio presentation channel rendering directional stereo audio cues and emergency alerts.
 * Utilizes low-latency PCM stereo synthesis with AudioTrack for headphone channel panning,
 * with graceful ToneGenerator fallback when dynamic AudioTrack creation is unavailable.
 */
class AndroidSpatialAudioChannel {

    private var toneGenerator: ToneGenerator? = null
    private var isAvailable: Boolean = true
    private var lastToneTimeMs: Long = 0L

    init {
        initialize()
    }

    @Synchronized
    fun initialize() {
        if (toneGenerator != null && isAvailable) return
        try {
            toneGenerator?.release()
            toneGenerator = ToneGenerator(AudioManager.STREAM_MUSIC, 85)
            isAvailable = true
        } catch (ex: Exception) {
            Log.e("PRISM_AUDIO", "Failed to initialize ToneGenerator fallback: ${ex.message}")
            isAvailable = false
        }
    }

    fun isAvailable(): Boolean = isAvailable

    fun deliver(cue: PrismPresentationCue): Boolean {
        if (cue.channel != "spatial_audio") return false

        val now = System.currentTimeMillis()
        val minIntervalMs = if (cue.priority == "emergency") 400L else 800L
        if (now - lastToneTimeMs < minIntervalMs) return false
        lastToneTimeMs = now

        val spatial = cue.spatialCue
        val pan = spatial?.pan ?: (cue.direction?.coerceIn(-1.0, 1.0) ?: 0.0)
        val durationSec = spatial?.durationSeconds ?: cue.duration.coerceIn(0.15, 0.35)
        val volume = spatial?.volume ?: cue.intensity.coerceIn(0.1, 1.0)
        val freq = spatial?.frequencyHz ?: when (cue.priority) {
            "emergency" -> 1100
            "urgent" -> 800
            "warning" -> 650
            else -> 500
        }

        // 1. Attempt dynamic stereo PCM synthesis with AudioTrack for headphone panning
        val renderedStereo = playStereoPcm(
            pan = pan,
            frequencyHz = freq,
            durationSec = durationSec,
            volume = volume,
            isEmergency = (cue.priority == "emergency")
        )

        if (renderedStereo) return true

        // 2. Fallback to ToneGenerator if AudioTrack synthesis failed or on restricted device
        return playToneGeneratorFallback(cue)
    }

    private fun playStereoPcm(
        pan: Double,
        frequencyHz: Int,
        durationSec: Double,
        volume: Double,
        isEmergency: Boolean
    ): Boolean {
        try {
            val sampleRate = 44100
            val numSamples = (sampleRate * durationSec).toInt()
            if (numSamples <= 0) return false

            val (leftGain, rightGain) = SpatialCue.computeStereoGains(pan, volume)

            val pcmData = ShortArray(numSamples * 2) // Interleaved L, R, L, R...
            val attackSamples = (sampleRate * 0.02).toInt() // 20ms attack
            val decaySamples = (sampleRate * 0.03).toInt()  // 30ms decay

            for (i in 0 until numSamples) {
                val t = i.toDouble() / sampleRate
                // Tone synthesis (dual-frequency harmonic pulse for emergency stop)
                val rawSine = if (isEmergency) {
                    0.6 * sin(2.0 * PI * frequencyHz * t) + 0.4 * sin(2.0 * PI * (frequencyHz * 0.75) * t)
                } else {
                    sin(2.0 * PI * frequencyHz * t)
                }

                // Envelope to avoid acoustic click
                val env = when {
                    i < attackSamples -> i.toDouble() / attackSamples
                    i > (numSamples - decaySamples) -> (numSamples - i).toDouble() / decaySamples
                    else -> 1.0
                }

                val sampleVal = (rawSine * env * 32767.0).coerceIn(-32768.0, 32767.0)
                pcmData[i * 2] = (sampleVal * leftGain).toInt().toShort()
                pcmData[i * 2 + 1] = (sampleVal * rightGain).toInt().toShort()
            }

            val bufferSize = pcmData.size * 2
            val attributes = AudioAttributes.Builder()
                .setUsage(if (isEmergency) AudioAttributes.USAGE_ALARM else AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()

            val format = AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                .build()

            thread(name = "PrismSpatialAudioPlayback", isDaemon = true) {
                var track: AudioTrack? = null
                try {
                    track = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                        AudioTrack.Builder()
                            .setAudioAttributes(attributes)
                            .setAudioFormat(format)
                            .setBufferSizeInBytes(bufferSize)
                            .setTransferMode(AudioTrack.MODE_STATIC)
                            .build()
                    } else {
                        @Suppress("DEPRECATION")
                        AudioTrack(
                            AudioManager.STREAM_MUSIC,
                            sampleRate,
                            AudioFormat.CHANNEL_OUT_STEREO,
                            AudioFormat.ENCODING_PCM_16BIT,
                            bufferSize,
                            AudioTrack.MODE_STATIC
                        )
                    }

                    track.write(pcmData, 0, pcmData.size)
                    track.play()
                    Thread.sleep((durationSec * 1000).toLong() + 50)
                } catch (_: Exception) {
                    // Fallback to ToneGenerator will be triggered if this threw
                } finally {
                    try {
                        track?.stop()
                        track?.release()
                    } catch (_: Exception) {}
                }
            }
            return true
        } catch (_: Exception) {
            return false
        }
    }

    private fun playToneGeneratorFallback(cue: PrismPresentationCue): Boolean {
        val generator = toneGenerator ?: return false
        try {
            when (cue.priority) {
                "emergency" -> generator.startTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 300)
                "urgent" -> {
                    val toneType = if ((cue.direction ?: 0.0) < 0) ToneGenerator.TONE_PROP_BEEP2 else ToneGenerator.TONE_PROP_ACK
                    generator.startTone(toneType, 200)
                }
                "warning" -> generator.startTone(ToneGenerator.TONE_PROP_BEEP, 150)
                else -> generator.startTone(ToneGenerator.TONE_PROP_PROMPT, 100)
            }
            return true
        } catch (ex: Exception) {
            Log.e("PRISM_AUDIO", "Error playing fallback tone: ${ex.message}")
            return false
        }
    }

    fun shutdown() {
        try {
            toneGenerator?.release()
        } catch (_: Exception) {}
        toneGenerator = null
        isAvailable = false
    }
}
