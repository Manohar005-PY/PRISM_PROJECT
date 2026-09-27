package com.prism.assistant

import android.app.Application
import com.prism.assistant.runtime.PrismRuntime

/**
 * Application singleton hosting the PRISM runtime instance.
 */
class PrismApplication : Application() {

    lateinit var runtime: PrismRuntime
        private set

    override fun onCreate() {
        super.onCreate()
        runtime = PrismRuntime(this)
    }
}
