package com.prism.assistant.presentation

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.util.Log
import com.prism.assistant.bridge.PrismPresentationCue
import java.util.Locale

/**
 * Speech presentation channel adapting Android TextToSpeech with emergency interrupt capability,
 * multi-tier locale fallback, lifecycle re-initialization, and speech rate-limiting hysteresis.
 */
class AndroidSpeechChannel(
    context: Context,
    private val onInitComplete: ((Boolean) -> Unit)? = null
) : TextToSpeech.OnInitListener {

    private val appContext: Context = context.applicationContext
    private var tts: TextToSpeech? = null
    private var isReady: Boolean = false
    private var isInitializing: Boolean = false
    private var pendingCue: PrismPresentationCue? = null
    private var lastSpokenText: String? = null
    private var lastSpeechTimeMs: Long = 0L

    init {
        initialize()
    }

    /**
     * Initializes or re-initializes the TextToSpeech engine safely across lifecycle transitions.
     */
    @Synchronized
    fun initialize() {
        if (tts != null && isReady) {
            return
        }
        if (isInitializing) {
            return
        }

        try {
            isInitializing = true
            isReady = false
            tts?.shutdown()
            tts = TextToSpeech(appContext, this)
            Log.i("PRISM_SPEECH", "Initializing TextToSpeech engine...")
        } catch (e: Exception) {
            isInitializing = false
            isReady = false
            Log.e("PRISM_SPEECH", "Failed to create TextToSpeech instance: ${e.message}", e)
            onInitComplete?.invoke(false)
        }
    }

    private val audioManager = appContext.getSystemService(Context.AUDIO_SERVICE) as? android.media.AudioManager
    private var isDuckingActive: Boolean = false

    private fun requestAudioDucking() {
        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                val focusRequest = android.media.AudioFocusRequest.Builder(android.media.AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .build()
                audioManager?.requestAudioFocus(focusRequest)
            } else {
                @Suppress("DEPRECATION")
                audioManager?.requestAudioFocus(null, android.media.AudioManager.STREAM_MUSIC, android.media.AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            }
            isDuckingActive = true
        } catch (_: Exception) {}
    }

    private fun abandonAudioDucking() {
        try {
            if (isDuckingActive) {
                @Suppress("DEPRECATION")
                audioManager?.abandonAudioFocus(null)
                isDuckingActive = false
            }
        } catch (_: Exception) {}
    }

    override fun onInit(status: Int) {
        isInitializing = false
        if (status == TextToSpeech.SUCCESS) {
            try {
                // 1. Configure AudioAttributes for navigation guidance audio stream
                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
                tts?.setAudioAttributes(audioAttributes)

                // 2. Set utterance progress listener for audio ducking lifecycle
                tts?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        requestAudioDucking()
                    }
                    override fun onDone(utteranceId: String?) {
                        abandonAudioDucking()
                    }
                    override fun onError(utteranceId: String?) {
                        abandonAudioDucking()
                    }
                })

                // 3. Multi-tier locale fallback: US -> Device Default -> English -> Available
                var langResult = tts?.setLanguage(Locale.US)
                if (langResult == TextToSpeech.LANG_MISSING_DATA || langResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                    val defaultLocale = Locale.getDefault()
                    Log.w("PRISM_SPEECH", "Locale.US not supported/missing. Falling back to default: $defaultLocale")
                    langResult = tts?.setLanguage(defaultLocale)
                }
                if (langResult == TextToSpeech.LANG_MISSING_DATA || langResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("PRISM_SPEECH", "Default locale not supported. Falling back to Locale.ENGLISH")
                    langResult = tts?.setLanguage(Locale.ENGLISH)
                }

                isReady = (langResult != TextToSpeech.LANG_MISSING_DATA && langResult != TextToSpeech.LANG_NOT_SUPPORTED)
                tts?.setSpeechRate(1.1f) // Slightly faster for prompt navigation guidance
                Log.i("PRISM_SPEECH", "TTS Engine initialized successfully. Ready=$isReady (langResult=$langResult)")
                onInitComplete?.invoke(isReady)

                // 4. Deliver any pending cue captured during initialization
                pendingCue?.let { cue ->
                    pendingCue = null
                    deliver(cue)
                }
            } catch (ex: Exception) {
                Log.e("PRISM_SPEECH", "Error during TTS post-init configuration: ${ex.message}", ex)
                isReady = false
                onInitComplete?.invoke(false)
            }
        } else {
            isReady = false
            Log.e("PRISM_SPEECH", "TTS Engine initialization failed with status $status")
            onInitComplete?.invoke(false)
        }
    }

    fun isAvailable(): Boolean = isReady

    fun deliver(cue: PrismPresentationCue): Boolean {
        if (cue.channel != "speech" || cue.text.isNullOrBlank()) return false

        // If TTS is not ready or null, attempt re-initialization and buffer urgent cue
        if (!isReady || tts == null) {
            if (cue.priority == "emergency" || cue.priority == "urgent") {
                pendingCue = cue
            }
            if (tts == null && !isInitializing) {
                initialize()
            }
            return false
        }

        val text = cue.text.trim()
        val now = System.currentTimeMillis()

        // Hysteresis cooldown: Prevent speech from cancelling itself 30 times a second
        val minIntervalMs = when (cue.priority) {
            "emergency" -> 1000L  // Minimum 1.0s between emergency announcements
            "urgent" -> 1600L     // Minimum 1.6s between urgent turns
            else -> 2500L         // Minimum 2.5s between warnings/cautions
        }

        val isDifferentUtterance = (text != lastSpokenText)
        val cooldownPassed = (now - lastSpeechTimeMs >= minIntervalMs)

        // For urgent/emergency cues, speak if different or cooldown passed; for low priority, require cooldown
        if (!cooldownPassed && !isDifferentUtterance && cue.priority != "emergency") {
            return false
        }

        if (!cooldownPassed && text == lastSpokenText) {
            return false
        }

        lastSpokenText = text
        lastSpeechTimeMs = now

        // Emergency or urgent cues pre-empt queued navigation chatter
        val queueMode = if (cue.priority == "emergency" || cue.priority == "urgent") {
            TextToSpeech.QUEUE_FLUSH  // Immediate preemption for emergency stop & urgent evasion
        } else {
            TextToSpeech.QUEUE_ADD
        }

        Log.i("PRISM_SPEECH", "Speaking: '$text' [Priority: ${cue.priority}]")
        val res = tts?.speak(text, queueMode, null, cue.messageId)
        return res == TextToSpeech.SUCCESS
    }

    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Exception) {}
        tts = null
        isReady = false
        isInitializing = false
    }
}
