package com.jlu.facialanalysis.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.jlu.facialanalysis.ui.UiState
import com.jlu.facialanalysis.ui.theme.BotanicalPaperBackground
import com.jlu.facialanalysis.ui.theme.CardCream
import com.jlu.facialanalysis.ui.theme.HandDrawnJournalBox
import com.jlu.facialanalysis.ui.theme.HandDrawnLeafDivider
import com.jlu.facialanalysis.ui.theme.InkPrimary
import com.jlu.facialanalysis.ui.theme.InkSecondary
import com.jlu.facialanalysis.ui.theme.SageDeep
import com.jlu.facialanalysis.ui.theme.SageMist
import com.jlu.facialanalysis.ui.theme.SagePrimary
import com.jlu.facialanalysis.ui.theme.SageSoftWash
import com.jlu.facialanalysis.ui.theme.SketchStroke
import com.jlu.facialanalysis.ui.theme.SoftClayText
import com.jlu.facialanalysis.ui.theme.WarmSandBg
import com.jlu.facialanalysis.ui.theme.WarmSandText

@Composable
fun AnalysisResultScreen(
    uiState: UiState,
    resolveUrl: (String?) -> String?,
    onBack: () -> Unit,
    onRetakeCamera: () -> Unit,
    onRepickGallery: () -> Unit,
    onOpenOfflineDemos: () -> Unit,
    onSelectCandidateFace: (String) -> Unit
) {
    val result = uiState.taskResult
    val primaryPhotoModel: Any? = uiState.localPreviewBytes
        ?: resolveUrl(result?.explanation?.originalImageUrl)
        ?: resolveUrl(result?.rejection?.previewImageUrl)
        ?: resolveUrl(result?.selection?.previewImageUrl)

    BotanicalPaperBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(horizontal = 22.dp, vertical = 14.dp)
        ) {
            // 顶部极简导航栏
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Surface(
                    onClick = onBack,
                    shape = CircleShape,
                    color = CardCream.copy(alpha = 0.88f),
                    border = BorderStroke(1.dp, SketchStroke),
                    modifier = Modifier.size(42.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Outlined.ArrowBack,
                            contentDescription = "返回首页",
                            tint = SageDeep,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "随心一拍",
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontFamily = FontFamily.Serif,
                            fontWeight = FontWeight.SemiBold,
                            letterSpacing = 1.6.sp
                        )
                    )
                    if (result?.isOfflineReplay == true) {
                        Surface(
                            color = WarmSandBg,
                            shape = RoundedCornerShape(50)
                        ) {
                            Text(
                                text = "示例",
                                color = WarmSandText,
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                            )
                        }
                    }
                }

                Surface(
                    onClick = onOpenOfflineDemos,
                    shape = CircleShape,
                    color = CardCream.copy(alpha = 0.88f),
                    border = BorderStroke(1.dp, SketchStroke),
                    modifier = Modifier.size(42.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Outlined.AutoAwesome,
                            contentDescription = "示例演示",
                            tint = SageDeep,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // 主体区域：所有内容完整收纳在同一个手绘信笺框内，不拆零碎小卡片
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                HandDrawnJournalBox(modifier = Modifier.fillMaxWidth()) {
                    when {
                        uiState.networkError != null && result == null -> {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 12.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                if (primaryPhotoModel != null) {
                                    PortraitPhotoFrame(photoModel = primaryPhotoModel)
                                    Spacer(modifier = Modifier.height(16.dp))
                                }
                                Text(
                                    text = "暂时没连上分析服务",
                                    style = MaterialTheme.typography.titleMedium.copy(color = SoftClayText)
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = "请检查右上角连接地址是否与电脑一致，或者点击右上角星星图标查看预设示例。",
                                    style = MaterialTheme.typography.bodyMedium.copy(color = InkSecondary),
                                    textAlign = TextAlign.Center
                                )
                            }
                        }

                        result == null || result.status in setOf("queued", "processing") -> {
                            Column(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                if (primaryPhotoModel != null) {
                                    PortraitPhotoFrame(photoModel = primaryPhotoModel)
                                    Spacer(modifier = Modifier.height(20.dp))
                                } else {
                                    Spacer(modifier = Modifier.height(24.dp))
                                }
                                CircularProgressIndicator(
                                    color = SagePrimary,
                                    modifier = Modifier.size(28.dp),
                                    strokeWidth = 2.4.dp
                                )
                                Spacer(modifier = Modifier.height(12.dp))
                                Text(
                                    text = "正在细细端详画面中的面容…",
                                    style = MaterialTheme.typography.bodyMedium.copy(
                                        fontFamily = FontFamily.Serif,
                                        color = SageDeep
                                    )
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                            }
                        }

                        result.status == "rejected" && result.rejection != null -> {
                            UnifiedRejectedContent(
                                rejection = result.rejection,
                                photoModel = uiState.localPreviewBytes ?: resolveUrl(result.rejection.previewImageUrl)
                            )
                        }

                        result.status == "need_selection" && result.selection != null -> {
                            UnifiedFaceSelectionContent(
                                selection = result.selection,
                                photoModel = resolveUrl(result.selection.previewImageUrl) ?: primaryPhotoModel,
                                isSubmitting = uiState.isSubmitting,
                                onSelectFace = onSelectCandidateFace
                            )
                        }

                        else -> {
                            // 单一整体框内展示：人脸照片 + 表观年龄 + 颜值评分 + 一段完整描述
                            val sc = result.scores
                            val imgRatio = result.targetFace?.let { tf ->
                                if (tf.imageWidth > 0 && tf.imageHeight > 0) {
                                    (tf.imageWidth.toFloat() / tf.imageHeight.toFloat()).coerceIn(0.76f, 1.25f)
                                } else {
                                    0.86f
                                }
                            } ?: 0.86f

                            Column(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                if (primaryPhotoModel != null) {
                                    PortraitPhotoFrame(
                                        photoModel = primaryPhotoModel,
                                        aspectRatio = imgRatio
                                    )
                                    Spacer(modifier = Modifier.height(18.dp))
                                }

                                if (sc != null) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clip(RoundedCornerShape(18.dp))
                                            .background(SageSoftWash.copy(alpha = 0.85f))
                                            .padding(vertical = 14.dp, horizontal = 12.dp)
                                            .height(IntrinsicSize.Min),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        MetricInlineItem(
                                            label = "表观年龄",
                                            value = "${sc.apparentAgeDisplay}",
                                            unit = "岁",
                                            modifier = Modifier.weight(1f)
                                        )

                                        Box(
                                            modifier = Modifier
                                                .width(1.dp)
                                                .fillMaxHeight()
                                                .padding(vertical = 4.dp)
                                                .background(SketchStroke)
                                        )

                                        MetricInlineItem(
                                            label = "颜值评分",
                                            value = "%.1f".format(sc.beautyDisplay),
                                            unit = "/ 5.0",
                                            modifier = Modifier.weight(1f)
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(16.dp))
                                HandDrawnLeafDivider()
                                Spacer(modifier = Modifier.height(14.dp))

                                val descriptionParagraph = result.buildUnifiedDescription()
                                Text(
                                    text = descriptionParagraph,
                                    style = MaterialTheme.typography.bodyLarge.copy(
                                        fontSize = 15.5.sp,
                                        lineHeight = 28.sp,
                                        color = InkPrimary
                                    ),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 4.dp)
                                )

                                if (!result.isTerminal) {
                                    Spacer(modifier = Modifier.height(12.dp))
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.Center
                                    ) {
                                        CircularProgressIndicator(
                                            color = SagePrimary,
                                            modifier = Modifier.size(14.dp),
                                            strokeWidth = 1.8.dp
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = "正在补充面容细节描述…",
                                            style = MaterialTheme.typography.bodySmall.copy(color = SageDeep)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(18.dp))

                // 底部简洁操作按钮
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally)
                ) {
                    OutlinedButton(
                        onClick = onRepickGallery,
                        shape = RoundedCornerShape(50),
                        border = BorderStroke(1.1.dp, SketchStroke),
                        colors = ButtonDefaults.outlinedButtonColors(
                            containerColor = CardCream.copy(alpha = 0.85f),
                            contentColor = SageDeep
                        ),
                        modifier = Modifier.weight(1f).height(48.dp)
                    ) {
                        Text("相册选图", style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium))
                    }

                    Button(
                        onClick = onRetakeCamera,
                        shape = RoundedCornerShape(50),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = SagePrimary,
                            contentColor = CardCream
                        ),
                        modifier = Modifier.weight(1.25f).height(48.dp)
                    ) {
                        Text("再拍一张", style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium))
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))
            }
        }
    }
}

@Composable
private fun PortraitPhotoFrame(
    photoModel: Any,
    aspectRatio: Float = 0.86f
) {
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = SageMist.copy(alpha = 0.45f),
        border = BorderStroke(1.dp, SketchStroke),
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp)
    ) {
        AsyncImage(
            model = photoModel,
            contentDescription = "人脸照片",
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(aspectRatio)
                .padding(5.dp)
                .clip(RoundedCornerShape(16.dp))
        )
    }
}

@Composable
private fun MetricInlineItem(
    label: String,
    value: String,
    unit: String,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall.copy(
                color = InkSecondary,
                letterSpacing = 0.8.sp
            )
        )
        Spacer(modifier = Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.Bottom) {
            Text(
                text = value,
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontFamily = FontFamily.Serif,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 28.sp,
                    color = SageDeep
                )
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = unit,
                style = MaterialTheme.typography.bodySmall.copy(color = InkSecondary),
                modifier = Modifier.padding(bottom = 4.dp)
            )
        }
    }
}
