package com.jlu.facialanalysis.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.util.Size as AndroidSize
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Cameraswitch
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.jlu.facialanalysis.ui.theme.BotanicalPaperBackground
import com.jlu.facialanalysis.ui.theme.CardCream
import com.jlu.facialanalysis.ui.theme.HandDrawnJournalBox
import com.jlu.facialanalysis.ui.theme.InkSecondary
import com.jlu.facialanalysis.ui.theme.SageMist
import com.jlu.facialanalysis.ui.theme.SagePrimary
import com.jlu.facialanalysis.ui.theme.SketchStroke
import com.jlu.facialanalysis.vision.ImagePreprocessor
import java.util.concurrent.Executors

@Composable
fun CameraPreviewScreen(
    isSubmitting: Boolean,
    onBack: () -> Unit,
    onPickFromGallery: () -> Unit,
    onCapturedBytes: (ByteArray) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val permLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasCameraPermission) {
        BotanicalPaperBackground {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(28.dp),
                contentAlignment = Alignment.Center
            ) {
                HandDrawnJournalBox(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "开启相机权限即可随时拍摄",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "也可以直接从右下角打开相册挑选照片",
                            style = MaterialTheme.typography.bodySmall,
                            color = InkSecondary
                        )
                        Spacer(modifier = Modifier.height(20.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            OutlinedButton(
                                onClick = onPickFromGallery,
                                shape = RoundedCornerShape(50),
                                border = BorderStroke(1.dp, SketchStroke)
                            ) {
                                Text("从相册选图", color = SagePrimary)
                            }
                            Button(
                                onClick = { permLauncher.launch(Manifest.permission.CAMERA) },
                                colors = ButtonDefaults.buttonColors(containerColor = SagePrimary),
                                shape = RoundedCornerShape(50)
                            ) {
                                Text("允许使用相机")
                            }
                        }
                    }
                }
            }
        }
        return
    }

    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_FRONT) }
    var isCapturingFrame by remember { mutableStateOf(false) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val imageCapture = remember {
        val captureResSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    AndroidSize(1280, 1600),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(captureResSelector)
            .build()
    }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    var cameraProviderRef by remember { mutableStateOf<ProcessCameraProvider?>(null) }

    DisposableEffect(Unit) {
        onDispose {
            runCatching { cameraProviderRef?.unbindAll() }
            captureExecutor.shutdown()
        }
    }

    LaunchedEffect(lensFacing) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            runCatching {
                val cameraProvider = providerFuture.get()
                cameraProviderRef = cameraProvider
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val selector = CameraSelector.Builder()
                    .requireLensFacing(lensFacing)
                    .build()

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    selector,
                    preview,
                    imageCapture
                )
            }.onFailure { err ->
                cameraError = "相机暂时无法打开：${err.message}"
            }
        }, ContextCompat.getMainExecutor(context))
    }

    val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1C231E))
    ) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // 中央柔和手绘感浅绿取景四角线（纯粹干净，无任何数字百分比干扰）
        Canvas(modifier = Modifier.fillMaxSize()) {
            val boxW = size.width * 0.68f
            val boxH = boxW * 1.22f
            val left = (size.width - boxW) / 2f
            val top = (size.height - boxH) * 0.40f
            val right = left + boxW
            val bottom = top + boxH
            val arm = 26.dp.toPx()
            val strokeW = 2.2.dp.toPx()
            val tint = SageMist.copy(alpha = 0.72f)

            // Top-left
            drawLine(tint, Offset(left, top), Offset(left + arm, top), strokeW, StrokeCap.Round)
            drawLine(tint, Offset(left, top), Offset(left, top + arm), strokeW, StrokeCap.Round)
            // Top-right
            drawLine(tint, Offset(right, top), Offset(right - arm, top), strokeW, StrokeCap.Round)
            drawLine(tint, Offset(right, top), Offset(right, top + arm), strokeW, StrokeCap.Round)
            // Bottom-left
            drawLine(tint, Offset(left, bottom), Offset(left + arm, bottom), strokeW, StrokeCap.Round)
            drawLine(tint, Offset(left, bottom), Offset(left, bottom - arm), strokeW, StrokeCap.Round)
            // Bottom-right
            drawLine(tint, Offset(right, bottom), Offset(right - arm, bottom), strokeW, StrokeCap.Round)
            drawLine(tint, Offset(right, bottom), Offset(right, bottom - arm), strokeW, StrokeCap.Round)
        }

        // 左上角简洁返回按钮
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 18.dp, vertical = 14.dp)
        ) {
            Surface(
                onClick = onBack,
                shape = CircleShape,
                color = Color.Black.copy(alpha = 0.32f),
                border = BorderStroke(1.dp, CardCream.copy(alpha = 0.35f)),
                modifier = Modifier.size(44.dp)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Outlined.ArrowBack,
                        contentDescription = "返回",
                        tint = CardCream,
                        modifier = Modifier.size(22.dp)
                    )
                }
            }
        }

        // 底部柔和渐变控制栏：左切换镜头 · 中快门 · 右下角相册选图
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Color(0xFF1A221C).copy(alpha = 0.72f)
                        )
                    )
                )
                .navigationBarsPadding()
                .padding(horizontal = 36.dp, vertical = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (cameraError != null) {
                Text(
                    text = cameraError!!,
                    color = CardCream,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(bottom = 12.dp)
                )
            }

            val busy = isSubmitting || isCapturingFrame
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                // 左下：切换前后摄像头
                Surface(
                    onClick = {
                        if (!busy) {
                            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                                CameraSelector.LENS_FACING_BACK
                            } else {
                                CameraSelector.LENS_FACING_FRONT
                            }
                        }
                    },
                    shape = CircleShape,
                    color = Color.Black.copy(alpha = 0.34f),
                    border = BorderStroke(1.dp, CardCream.copy(alpha = 0.42f)),
                    modifier = Modifier.size(52.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Outlined.Cameraswitch,
                            contentDescription = "切换前后镜头",
                            tint = CardCream,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }

                // 中央：低饱和浅绿手绘感双环快门按钮
                Surface(
                    onClick = {
                        if (busy) return@Surface
                        isCapturingFrame = true
                        cameraError = null
                        imageCapture.takePicture(
                            captureExecutor,
                            object : ImageCapture.OnImageCapturedCallback() {
                                override fun onCaptureSuccess(image: ImageProxy) {
                                    try {
                                        val bytes = ImagePreprocessor.encodeImageProxyNormalized(
                                            image = image,
                                            mirrorHorizontally = isFront
                                        )
                                        ContextCompat.getMainExecutor(context).execute {
                                            isCapturingFrame = false
                                            onCapturedBytes(bytes)
                                        }
                                    } catch (t: Throwable) {
                                        ContextCompat.getMainExecutor(context).execute {
                                            isCapturingFrame = false
                                            cameraError = "处理照片时出错，请再试一次"
                                        }
                                    } finally {
                                        runCatching { image.close() }
                                    }
                                }

                                override fun onError(exception: ImageCaptureException) {
                                    ContextCompat.getMainExecutor(context).execute {
                                        isCapturingFrame = false
                                        cameraError = "拍摄未成功，请再试一次"
                                    }
                                }
                            }
                        )
                    },
                    shape = CircleShape,
                    color = CardCream.copy(alpha = 0.25f),
                    border = BorderStroke(2.5.dp, CardCream),
                    modifier = Modifier.size(80.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Surface(
                            shape = CircleShape,
                            color = if (busy) SageMist else SagePrimary,
                            modifier = Modifier.size(64.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                if (busy) {
                                    CircularProgressIndicator(
                                        color = SagePrimary,
                                        modifier = Modifier.size(26.dp),
                                        strokeWidth = 2.5.dp
                                    )
                                }
                            }
                        }
                    }
                }

                // 右下角：相册图标（直接从相册选图片分析）
                Surface(
                    onClick = {
                        if (!busy) {
                            onPickFromGallery()
                        }
                    },
                    shape = CircleShape,
                    color = Color.Black.copy(alpha = 0.34f),
                    border = BorderStroke(1.dp, CardCream.copy(alpha = 0.42f)),
                    modifier = Modifier.size(52.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Outlined.PhotoLibrary,
                            contentDescription = "从相册选择照片",
                            tint = CardCream,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }
            }
        }
    }
}
