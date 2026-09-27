package com.prism.assistant.camera

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

data class CameraFrameMeta(
    val frameNumber: Long,
    val width: Int,
    val height: Int,
    val captureFps: Double,
    val timestamp: Double
)

/**
 * Camera source adapting CameraX image stream with preview binding and bounded backpressure.
 */
class AndroidCameraFrameSource(
    private val context: Context,
    val backpressurePolicy: FrameBackpressurePolicy = FrameBackpressurePolicy(maxQueueDepth = 1)
) {
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var previewUseCase: Preview? = null
    private var isStreaming = false

    private val frameCounter = AtomicLong(0)
    private var lastFpsTimestamp = System.currentTimeMillis()
    private var framesInSecond = 0
    private var currentFps = 0.0

    fun startCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        onFrameAvailable: (ImageProxy, CameraFrameMeta) -> Unit
    ) {
        cameraExecutor = Executors.newSingleThreadExecutor()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                // 1. Setup Camera Preview
                previewUseCase = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                // 2. Setup Image Analysis (30 FPS YUV_420_888 stream)
                imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor!!) { imageProxy ->
                            val currentFrameNum = frameCounter.incrementAndGet()
                            val nowMs = System.currentTimeMillis()
                            framesInSecond++
                            if (nowMs - lastFpsTimestamp >= 1000) {
                                currentFps = (framesInSecond * 1000.0) / (nowMs - lastFpsTimestamp)
                                framesInSecond = 0
                                lastFpsTimestamp = nowMs
                            }

                            val timestamp = System.nanoTime() / 1_000_000_000.0
                            backpressurePolicy.onFrameCaptured(timestamp)

                            val meta = CameraFrameMeta(
                                frameNumber = currentFrameNum,
                                width = imageProxy.width,
                                height = imageProxy.height,
                                captureFps = currentFps,
                                timestamp = timestamp
                            )

                            try {
                                onFrameAvailable(imageProxy, meta)
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    previewUseCase,
                    imageAnalysis
                )
                isStreaming = true
            } catch (ex: Exception) {
                ex.printStackTrace()
                isStreaming = false
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stopCamera() {
        isStreaming = false
        imageAnalysis?.clearAnalyzer()
        cameraExecutor?.shutdown()
        cameraExecutor = null
    }

    fun isStreaming(): Boolean = isStreaming
}
