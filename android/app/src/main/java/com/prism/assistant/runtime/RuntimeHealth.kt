package com.prism.assistant.runtime

/**
 * Health and diagnostics state of the PRISM embedded runtime.
 */
data class RuntimeHealth(
    val status: String,
    val captureFps: Double,
    val processedFps: Double,
    val droppedFramesCount: Long,
    val averageLatencyMs: Double,
    val activeTrackCount: Int,
    val lastError: String? = null,
    val isCameraHealthy: Boolean = true,
    val isPresentationHealthy: Boolean = true
)
