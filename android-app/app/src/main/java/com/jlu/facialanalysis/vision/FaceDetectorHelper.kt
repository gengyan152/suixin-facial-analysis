package com.jlu.facialanalysis.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import java.io.Closeable

data class PreviewDetectedFace(
    val bboxPx: RectF,
    val confidence: Float,
    val keypointsNorm: List<Pair<Float, Float>> // 6 个基础定位点（眼、鼻、嘴、耳屏），归一化 [0,1]
)

data class LivePreviewDetectionState(
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val faces: List<PreviewDetectedFace> = emptyList(),
    val inferenceTimeMs: Long = 0L,
    val fps: Float = 0f,
    val detectorReady: Boolean = false,
    val hintMessage: String = "正在初始化端侧离线检测模型…"
)

/**
 * 端侧离线人脸检测器：固定加载随 APK assets 内置打包的 blaze_face_short_range.tflite
 * 支持飞行模式/断网冷启动；配合 CameraX STRATEGY_KEEP_ONLY_LATEST 在 finally 中严格释放 ImageProxy
 */
class FaceDetectorHelper(
    context: Context,
    minDetectionConfidence: Float = 0.65f
) : Closeable {

    companion object {
        const val BUNDLED_MODEL_ASSET = "blaze_face_short_range.tflite"
    }

    private var faceDetector: FaceDetector? = null
    @Volatile
    private var isClosed: Boolean = false
    private var lastFrameTimestampNs: Long = 0L
    private var smoothedFps: Float = 0f

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(BUNDLED_MODEL_ASSET)
            .build()
        val options = FaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMinDetectionConfidence(minDetectionConfidence)
            .setRunningMode(RunningMode.IMAGE)
            .build()
        faceDetector = FaceDetector.createFromOptions(context.applicationContext, options)
    }

    @Synchronized
    fun analyzeCameraFrame(imageProxy: ImageProxy): LivePreviewDetectionState {
        try {
            val detector = if (isClosed) null else faceDetector
            if (detector == null) {
                return LivePreviewDetectionState(
                    detectorReady = false,
                    hintMessage = "端侧人脸检测器未就绪"
                )
            }

            val startMs = SystemClock.elapsedRealtime()
            val rawBitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val maxRawSide = maxOf(rawBitmap.width, rawBitmap.height)
            val previewScale = if (maxRawSide > 640) 640f / maxRawSide.toFloat() else 1f
            val uprightBitmap = if (rotation != 0 || previewScale < 1f) {
                val matrix = Matrix().apply {
                    if (previewScale < 1f) postScale(previewScale, previewScale)
                    if (rotation != 0) postRotate(rotation.toFloat())
                }
                val transformed = Bitmap.createBitmap(
                    rawBitmap,
                    0,
                    0,
                    rawBitmap.width,
                    rawBitmap.height,
                    matrix,
                    true
                )
                if (transformed !== rawBitmap) rawBitmap.recycle()
                transformed
            } else {
                rawBitmap
            }

            val frameW = uprightBitmap.width
            val frameH = uprightBitmap.height
            val mpImage = BitmapImageBuilder(uprightBitmap).build()
            val result = detector.detect(mpImage)
            val costMs = SystemClock.elapsedRealtime() - startMs

            val nowNs = SystemClock.elapsedRealtimeNanos()
            if (lastFrameTimestampNs > 0L) {
                val instantFps = 1_000_000_000f / (nowNs - lastFrameTimestampNs).coerceAtLeast(1L)
                smoothedFps = if (smoothedFps == 0f) instantFps else (smoothedFps * 0.8f + instantFps * 0.2f)
            }
            lastFrameTimestampNs = nowNs

            val detectedFaces = result.detections().map { det ->
                val box = det.boundingBox()
                val score = det.categories().firstOrNull()?.score() ?: 0f
                val kps = det.keypoints().orElse(emptyList()).map { kp ->
                    kp.x() to kp.y()
                }
                PreviewDetectedFace(
                    bboxPx = RectF(box.left, box.top, box.right, box.bottom),
                    confidence = score,
                    keypointsNorm = kps
                )
            }

            val hint = when {
                detectedFaces.isEmpty() -> "未检测到人脸，请将正脸移入取景框"
                detectedFaces.size > 1 -> "检测到 ${detectedFaces.size} 个目标，拍照后可点击框选分析对象"
                else -> "已锁定单人面部（置信度 ${(detectedFaces[0].confidence * 100).toInt()}%），可点击快门分析"
            }

            return LivePreviewDetectionState(
                frameWidth = frameW,
                frameHeight = frameH,
                faces = detectedFaces,
                inferenceTimeMs = costMs,
                fps = smoothedFps,
                detectorReady = true,
                hintMessage = hint
            )
        } catch (_: Throwable) {
            return LivePreviewDetectionState(
                detectorReady = !isClosed,
                hintMessage = "正在同步预览帧…"
            )
        } finally {
            // 无论检测成功或异常，确保释放 ImageProxy，杜绝 CameraX 缓冲队列卡死
            runCatching { imageProxy.close() }
        }
    }

    @Synchronized
    override fun close() {
        if (isClosed) return
        isClosed = true
        runCatching { faceDetector?.close() }
        faceDetector = null
    }
}
