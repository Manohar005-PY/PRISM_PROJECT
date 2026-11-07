package com.prism.assistant.presentation

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.util.Log
import com.prism.assistant.bridge.PrismPresentationCue
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Speech presentation channel adapting Android TextToSpeech with emergency interrupt capability,
 * sentence completion protection, multi-tier locale fallback, lifecycle re-initialization,
 * and speech rate-limiting hysteresis.
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
    private var lastSpokenPriority: String? = null
    private var activeUtteranceStartTimeMs: Long = 0L

    private val isSpeakingActive = AtomicBoolean(false)

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

                // 2. Set utterance progress listener for audio ducking lifecycle and speech state tracking
                tts?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        isSpeakingActive.set(true)
                        activeUtteranceStartTimeMs = System.currentTimeMillis()
                        requestAudioDucking()
                    }
                    override fun onDone(utteranceId: String?) {
                        isSpeakingActive.set(false)
                        abandonAudioDucking()
                    }
                    override fun onError(utteranceId: String?) {
                        isSpeakingActive.set(false)
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

    /**
     * Checks if the TTS engine is actively speaking an utterance.
     */
    fun isSpeaking(): Boolean {
        return isSpeakingActive.get() || (tts?.isSpeaking == true)
    }

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
        val isEmergency = (cue.priority == "emergency")
        val currentlySpeaking = isSpeaking()

        // 1. Completion Protection: NEVER interrupt ongoing speech unless it is an EMERGENCY stop!
        if (currentlySpeaking) {
            if (!isEmergency) {
                // Non-emergency instructions must not cut off previous decision before it finishes speaking
                Log.d("PRISM_SPEECH", "Speech in progress; suppressing non-emergency cue to allow sentence completion: '$text'")
                return false
            } else {
                // If an emergency announcement is ALREADY speaking and started recently (<1.2s), let it finish saying "Stop!"
                if (lastSpokenPriority == "emergency" && (now - activeUtteranceStartTimeMs < 1200L)) {
                    Log.d("PRISM_SPEECH", "Emergency announcement already actively speaking; suppressing self-interruption: '$text'")
                    return false
                }
            }
        }

        // 2. Minimum Cooldown Intervals between utterances
        val minIntervalMs = when (cue.priority) {
            "emergency" -> 1000L  // Minimum 1.0s between emergency announcements
            "urgent" -> 1800L     // Minimum 1.8s between urgent directional guidance
            else -> 2500L         // Minimum 2.5s between warnings/cautions/info
        }

        val elapsed = now - lastSpeechTimeMs

        // Non-emergency cues ALWAYS require the cooldown interval to pass since speech was started
        if (!isEmergency && elapsed < minIntervalMs) {
            Log.d("PRISM_SPEECH", "Speech cooldown active ($elapsed ms < $minIntervalMs ms); suppressing: '$text'")
            return false
        }

        // Repetitive identical text suppression: require extended interval (at least 3.0s)
        if (text == lastSpokenText && elapsed < 3000L && !isEmergency) {
            return false
        }

        lastSpokenText = text
        lastSpeechTimeMs = now
        lastSpokenPriority = cue.priority
        activeUtteranceStartTimeMs = now
        isSpeakingActive.set(true)

        // Only true emergencies use QUEUE_FLUSH to preempt immediately;
        // Routine or urgent directional guidance uses QUEUE_ADD (speaks cleanly since not currentlySpeaking)
        val queueMode = if (isEmergency) {
            TextToSpeech.QUEUE_FLUSH  // Immediate preemption for emergency safety stops
        } else {
            TextToSpeech.QUEUE_ADD
        }

        val utteranceId = cue.messageId.ifBlank { "speech_${now}" }
        Log.i("PRISM_SPEECH", "Speaking: '$text' [Priority: ${cue.priority}, QueueMode: $queueMode]")
        val res = tts?.speak(text, queueMode, null, utteranceId)
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
        isSpeakingActive.set(false)
    }
}
