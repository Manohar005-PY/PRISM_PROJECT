package com.prism.assistant.presentation

import com.prism.assistant.bridge.PrismPresentationCue

/**
 * Visual display channel updating debug screen overlay and HUD banner.
 */
class AndroidVisualChannel(
    private val onHudUpdate: ((banner: String, priority: String) -> Unit)? = null
) {
    var latestBanner: String? = null
        private set

    fun isAvailable(): Boolean = true

    fun deliver(cue: PrismPresentationCue): Boolean {
        if (cue.channel != "visual" || cue.text.isNullOrBlank()) return false
        latestBanner = cue.text
        onHudUpdate?.invoke(cue.text, cue.priority)
        return true
    }
}
