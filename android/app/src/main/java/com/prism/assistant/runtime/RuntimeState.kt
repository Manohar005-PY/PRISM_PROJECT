package com.prism.assistant.runtime

/**
 * Explicit lifecycle states for PRISM Android embedded runtime.
 */
enum class RuntimeState {
    STOPPED,
    INITIALIZING,
    RUNNING,
    DEGRADED,
    ERROR,
    STOPPING
}

/**
 * Runtime execution mode.
 */
enum class ExecutionMode {
    LIVE,
    MOCK,
    REPLAY
}
