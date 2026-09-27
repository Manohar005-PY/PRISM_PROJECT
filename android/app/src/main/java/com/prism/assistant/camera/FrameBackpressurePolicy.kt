package com.prism.assistant.camera

import java.util.concurrent.atomic.AtomicLong

/**
 * Bounded queue policy that keeps only the latest frames and drops stale frames.
 */
class FrameBackpressurePolicy(
    val maxQueueDepth: Int = 1,
    val maxStaleDeltaSeconds: Double = 0.25
) {
    private val totalCaptured = AtomicLong(0)
    private val totalProcessed = AtomicLong(0)
    private val totalDropped = AtomicLong(0)

    fun onFrameCaptured(timestamp: Double): Boolean {
        totalCaptured.incrementAndGet()
        return true
    }

    fun recordDrop() {
        totalDropped.incrementAndGet()
    }

    fun recordProcessed() {
        totalProcessed.incrementAndGet()
    }

    fun isStale(frameTimestamp: Double, currentTimestamp: Double): Boolean {
        return (currentTimestamp - frameTimestamp) > maxStaleDeltaSeconds
    }

    fun getCapturedCount(): Long = totalCaptured.get()
    fun getProcessedCount(): Long = totalProcessed.get()
    fun getDroppedCount(): Long = totalDropped.get()

    fun reset() {
        totalCaptured.set(0)
        totalProcessed.set(0)
        totalDropped.set(0)
    }
}
