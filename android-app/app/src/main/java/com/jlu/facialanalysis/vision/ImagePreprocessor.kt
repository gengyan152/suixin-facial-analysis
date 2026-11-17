package com.jlu.facialanalysis.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import androidx.camera.core.ImageProxy
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 纯数学坐标变换器：将转正后分析帧中的像素坐标映射到预览视图（兼容前置摄像头水平镜像与 FILL_CENTER 裁剪）
 */
object PreviewCoordinateMapper {

    data class NormalizedPoint(val x: Float, val y: Float)

    fun mapRectToView(
        bboxPx: RectF,
        frameWidth: Int,
        frameHeight: Int,
        viewWidth: Float,
        viewHeight: Float,
        isFrontCamera: Boolean
    ): RectF {
        if (frameWidth <= 0 || frameHeight <= 0 || viewWidth <= 0f || viewHeight <= 0f) {
            return RectF()
        }
        val scale = max(viewWidth / frameWidth.toFloat(), viewHeight / frameHeight.toFloat())
        val scaledW = frameWidth * scale
        val scaledH = frameHeight * scale
        val offsetX = (viewWidth - scaledW) / 2f
        val offsetY = (viewHeight - scaledH) / 2f

        val rawLeft = bboxPx.left * scale + offsetX
        val rawRight = bboxPx.right * scale + offsetX
        val top = bboxPx.top * scale + offsetY
        val bottom = bboxPx.bottom * scale + offsetY

        return if (isFrontCamera) {
            RectF(viewWidth - rawRight, top, viewWidth - rawLeft, bottom)
        } else {
            RectF(rawLeft, top, rawRight, bottom)
        }
    }

    fun mapNormalizedPointToView(
        normX: Float,
        normY: Float,
        frameWidth: Int,
        frameHeight: Int,
        viewWidth: Float,
        viewHeight: Float,
        isFrontCamera: Boolean
    ): NormalizedPoint {
        if (frameWidth <= 0 || frameHeight <= 0 || viewWidth <= 0f || viewHeight <= 0f) {
            return NormalizedPoint(0f, 0f)
        }
        val scale = max(viewWidth / frameWidth.toFloat(), viewHeight / frameHeight.toFloat())
        val scaledW = frameWidth * scale
        val scaledH = frameHeight * scale
        val offsetX = (viewWidth - scaledW) / 2f
        val offsetY = (viewHeight - scaledH) / 2f

        val pxX = normX * frameWidth * scale + offsetX
        val pxY = normY * frameHeight * scale + offsetY
        val finalX = if (isFrontCamera) viewWidth - pxX else pxX
        return NormalizedPoint(finalX, pxY)
    }
}

/**
 * 内存级图像预处理器：读取 EXIF 方向转正，按长边 <= 1280 px 等比缩放，输出内存 JPEG 字节数组（不写任何磁盘临时文件）
 */
object ImagePreprocessor {

    const val MAX_DIMENSION_PX = 1280
    private const val JPEG_QUALITY = 90

    fun loadAndNormalizeFromUri(context: Context, uri: Uri): ByteArray {
        val rawBytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw IllegalArgumentException("无法读取所选相册图片")
        return normalizeImageBytes(rawBytes)
    }

    fun normalizeImageBytes(rawBytes: ByteArray): ByteArray {
        val exif = ExifInterface(ByteArrayInputStream(rawBytes))
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(rawBytes, 0, rawBytes.size, boundsOpts)
        val srcW = boundsOpts.outWidth
        val srcH = boundsOpts.outHeight
        require(srcW > 0 && srcH > 0) { "无法解码图像尺寸，请选择有效的图片格式" }

        var sampleSize = 1
        while (max(srcW / sampleSize, srcH / sampleSize) > MAX_DIMENSION_PX * 2) {
            sampleSize *= 2
        }

        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = BitmapFactory.decodeByteArray(rawBytes, 0, rawBytes.size, decodeOpts)
            ?: throw IllegalArgumentException("图像解码失败")

        val matrix = buildExifMatrix(orientation)
        val swapsAxes = orientation in setOf(
            ExifInterface.ORIENTATION_ROTATE_90,
            ExifInterface.ORIENTATION_ROTATE_270,
            ExifInterface.ORIENTATION_TRANSPOSE,
            ExifInterface.ORIENTATION_TRANSVERSE
        )
        val uprightW = if (swapsAxes) decoded.height else decoded.width
        val uprightH = if (swapsAxes) decoded.width else decoded.height
        val maxSide = max(uprightW, uprightH)
        if (maxSide > MAX_DIMENSION_PX) {
            val scale = MAX_DIMENSION_PX.toFloat() / maxSide.toFloat()
            matrix.postScale(scale, scale)
        }

        val finalBmp = if (!matrix.isIdentity) {
            val transformed = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
            if (transformed !== decoded) decoded.recycle()
            transformed
        } else {
            decoded
        }

        val out = ByteArrayOutputStream()
        finalBmp.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        finalBmp.recycle()
        return out.toByteArray()
    }

    /**
     * 相机拍摄帧专用低内存归一化：
     * - 若为 JPEG 格式（高像素传感器如 50MP/60MP），直接从 ByteBuffer 按 inSampleSize 降采样解码，避免整帧 ARGB_8888 撑爆堆内存；
     * - 将缩放、旋转与前置水平镜像合并为单次 Matrix 变换，峰值内存 < 15 MB。
     */
    fun encodeImageProxyNormalized(image: ImageProxy, mirrorHorizontally: Boolean): ByteArray {
        val rotationDegrees = image.imageInfo.rotationDegrees
        if (image.format == ImageFormat.JPEG && image.planes.isNotEmpty()) {
            val buffer = image.planes[0].buffer.duplicate()
            buffer.rewind()
            val jpegBytes = ByteArray(buffer.remaining())
            buffer.get(jpegBytes)

            val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, boundsOpts)
            val srcW = boundsOpts.outWidth
            val srcH = boundsOpts.outHeight
            if (srcW > 0 && srcH > 0) {
                var sampleSize = 1
                while (max(srcW / sampleSize, srcH / sampleSize) > MAX_DIMENSION_PX * 2) {
                    sampleSize *= 2
                }
                val decodeOpts = BitmapFactory.Options().apply {
                    inSampleSize = sampleSize
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }
                val decoded = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, decodeOpts)
                if (decoded != null) {
                    try {
                        return encodeBitmapNormalized(decoded, rotationDegrees, mirrorHorizontally)
                    } finally {
                        if (!decoded.isRecycled) decoded.recycle()
                    }
                }
            }
        }

        val bmp = image.toBitmap()
        try {
            return encodeBitmapNormalized(bmp, rotationDegrees, mirrorHorizontally)
        } finally {
            if (!bmp.isRecycled) bmp.recycle()
        }
    }

    fun encodeBitmapNormalized(bitmap: Bitmap, rotationDegrees: Int, mirrorHorizontally: Boolean): ByteArray {
        val normRot = ((rotationDegrees % 360) + 360) % 360
        val swapsAxes = normRot == 90 || normRot == 270
        val uprightW = if (swapsAxes) bitmap.height else bitmap.width
        val uprightH = if (swapsAxes) bitmap.width else bitmap.height
        val maxSide = max(uprightW, uprightH)

        val matrix = Matrix()
        if (maxSide > MAX_DIMENSION_PX) {
            val scale = MAX_DIMENSION_PX.toFloat() / maxSide.toFloat()
            matrix.postScale(scale, scale)
        }
        if (normRot != 0) {
            matrix.postRotate(normRot.toFloat())
        }
        if (mirrorHorizontally) {
            matrix.postScale(-1f, 1f)
        }

        val finalBmp = if (!matrix.isIdentity) {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }

        val out = ByteArrayOutputStream()
        finalBmp.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        if (finalBmp !== bitmap && !finalBmp.isRecycled) {
            finalBmp.recycle()
        }
        return out.toByteArray()
    }

    private fun buildExifMatrix(orientation: Int): Matrix {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.preScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.preScale(-1f, 1f)
            }
        }
        return matrix
    }
}
