package com.jlu.facialanalysis.ui.screens

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.jlu.facialanalysis.data.model.HealthStatus
import com.jlu.facialanalysis.data.network.BackendApiClient
import com.jlu.facialanalysis.data.offline.OfflineDemoRepository
import com.jlu.facialanalysis.data.offline.OfflineDemoSummary
import com.jlu.facialanalysis.ui.theme.BotanicalPaperBackground
import com.jlu.facialanalysis.ui.theme.CardCream
import com.jlu.facialanalysis.ui.theme.InkPrimary
import com.jlu.facialanalysis.ui.theme.InkSecondary
import com.jlu.facialanalysis.ui.theme.SageDeep
import com.jlu.facialanalysis.ui.theme.SageMist
import com.jlu.facialanalysis.ui.theme.SagePrimary
import com.jlu.facialanalysis.ui.theme.SageSoftWash
import com.jlu.facialanalysis.ui.theme.SketchStroke
import com.jlu.facialanalysis.ui.theme.SoftClayBg
import com.jlu.facialanalysis.ui.theme.SoftClayText
import com.jlu.facialanalysis.ui.theme.WarmSandBg
import com.jlu.facialanalysis.ui.theme.WarmSandText

@Composable
fun HomeScreen(
    onOpenCamera: () -> Unit,
    onOpenServerDialog: () -> Unit,
    onOpenOfflineDemos: () -> Unit
) {
    val infiniteTransition = rememberInfiniteTransition(label = "halo_breath")
    val haloScale by infiniteTransition.animateFloat(
        initialValue = 0.97f,
        targetValue = 1.04f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2800, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "halo_scale"
    )

    BotanicalPaperBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .padding(horizontal = 24.dp, vertical = 16.dp)
        ) {
            // 顶部极简标题与右上角两个轻量功能入口
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "随心一拍",
                    style = MaterialTheme.typography.headlineMedium.copy(
                        fontFamily = FontFamily.Serif,
                        fontWeight = FontWeight.SemiBold,
                        letterSpacing = 2.sp,
                        color = InkPrimary
                    )
                )

                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    TopSketchIconButton(
                        onClick = onOpenOfflineDemos
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.AutoAwesome,
                            contentDescription = "示例演示",
                            tint = SageDeep,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                    TopSketchIconButton(
                        onClick = onOpenServerDialog
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.Tune,
                            contentDescription = "连接设置",
                            tint = SageDeep,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }

            // 中央留白 + 大圆形手绘风拍照按钮
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .size(216.dp)
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null,
                                role = Role.Button,
                                onClick = onOpenCamera
                            )
                    ) {
                        // 外层手绘水彩浅绿晕染与双圈铅笔线条
                        Canvas(
                            modifier = Modifier
                                .fillMaxSize()
                                .scale(haloScale)
                        ) {
                            val center = Offset(size.width / 2f, size.height / 2f)
                            val outerR = size.minDimension / 2f - 8.dp.toPx()
                            drawCircle(
                                color = SageMist.copy(alpha = 0.62f),
                                radius = outerR,
                                center = center
                            )
                            drawCircle(
                                color = SketchStroke,
                                radius = outerR - 4.dp.toPx(),
                                center = Offset(center.x - 1.5.dp.toPx(), center.y + 1.5.dp.toPx()),
                                style = Stroke(width = 1.4.dp.toPx())
                            )
                            drawCircle(
                                color = SagePrimary.copy(alpha = 0.38f),
                                radius = outerR - 12.dp.toPx(),
                                center = Offset(center.x + 1.5.dp.toPx(), center.y - 1.dp.toPx()),
                                style = Stroke(
                                    width = 1.2.dp.toPx(),
                                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(16f, 10f), 0f)
                                )
                            )
                        }

                        // 核心低饱和鼠尾草绿大圆按钮
                        Surface(
                            onClick = onOpenCamera,
                            shape = CircleShape,
                            color = SagePrimary,
                            shadowElevation = 6.dp,
                            border = BorderStroke(2.5.dp, CardCream),
                            modifier = Modifier.size(148.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Canvas(modifier = Modifier.fillMaxSize()) {
                                    drawCircle(
                                        color = CardCream.copy(alpha = 0.35f),
                                        radius = size.minDimension / 2f - 9.dp.toPx(),
                                        style = Stroke(width = 1.dp.toPx())
                                    )
                                }
                                Icon(
                                    imageVector = Icons.Outlined.CameraAlt,
                                    contentDescription = "打开相机拍照",
                                    tint = CardCream,
                                    modifier = Modifier.size(50.dp)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(22.dp))

                    Text(
                        text = "一键分析年龄与颜值",
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontFamily = FontFamily.Serif,
                            fontWeight = FontWeight.Medium,
                            fontSize = 18.sp,
                            letterSpacing = 1.8.sp,
                            color = SageDeep
                        )
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
private fun TopSketchIconButton(
    onClick: () -> Unit,
    icon: @Composable () -> Unit
) {
    Surface(
        onClick = onClick,
        shape = CircleShape,
        color = CardCream.copy(alpha = 0.88f),
        border = BorderStroke(1.dp, SketchStroke),
        modifier = Modifier.size(42.dp)
    ) {
        Box(contentAlignment = Alignment.Center) {
            icon()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OfflineDemoBottomSheet(
    onDismiss: () -> Unit,
    onSelectOfflineDemo: (OfflineDemoSummary) -> Unit
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = CardCream
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "示例照片演示",
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold)
            )
            Text(
                text = "无需联网也可直接查看不同照片的分析效果：",
                style = MaterialTheme.typography.bodySmall
            )
            OfflineDemoRepository.DEMO_ITEMS.forEach { item ->
                Surface(
                    onClick = { onSelectOfflineDemo(item) },
                    color = SageSoftWash.copy(alpha = 0.7f),
                    shape = RoundedCornerShape(16.dp),
                    border = BorderStroke(1.dp, SketchStroke),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = item.title,
                                style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                            )
                            Text(
                                text = item.subtitle,
                                style = MaterialTheme.typography.bodySmall,
                                color = InkSecondary
                            )
                        }
                        Spacer(modifier = Modifier.width(10.dp))
                        Surface(
                            color = WarmSandBg,
                            shape = RoundedCornerShape(50)
                        ) {
                            Text(
                                text = item.badge,
                                color = WarmSandText,
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 12.sp),
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(20.dp))
        }
    }
}

@Composable
fun ServerConfigDialog(
    currentBaseUrl: String,
    isChecking: Boolean,
    healthStatus: HealthStatus?,
    healthError: String?,
    onDismiss: () -> Unit,
    onTestUrl: (String) -> Unit,
    onSaveUrl: (String) -> Boolean
) {
    var inputUrl by remember(currentBaseUrl) { mutableStateOf(currentBaseUrl) }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = CardCream,
        title = {
            Text(
                text = "连接设置",
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold)
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    text = "填入电脑端服务地址即可实时分析照片：",
                    style = MaterialTheme.typography.bodySmall
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    AssistChip(
                        onClick = { inputUrl = BackendApiClient.DEFAULT_USB_BASE_URL },
                        label = { Text("USB 连接") },
                        colors = AssistChipDefaults.assistChipColors(containerColor = SageSoftWash)
                    )
                    AssistChip(
                        onClick = { inputUrl = "http://172.17.0.173:8000" },
                        label = { Text("局域网地址") },
                        colors = AssistChipDefaults.assistChipColors(containerColor = SageSoftWash)
                    )
                }
                OutlinedTextField(
                    value = inputUrl,
                    onValueChange = { inputUrl = it },
                    label = { Text("服务地址") },
                    singleLine = true,
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier.fillMaxWidth()
                )
                FilledTonalButton(
                    onClick = { onTestUrl(inputUrl) },
                    enabled = !isChecking,
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    if (isChecking) {
                        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("正在检查连接…")
                    } else {
                        Text("测试连接")
                    }
                }

                if (healthStatus != null) {
                    Surface(
                        color = SageMist,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Outlined.CheckCircle,
                                contentDescription = null,
                                tint = SageDeep,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = if (healthStatus.warmedUp) "连接正常，随时可以拍照分析" else "已连上电脑，正在准备模型…",
                                color = SageDeep,
                                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium)
                            )
                        }
                    }
                }

                if (healthError != null) {
                    Surface(
                        color = SoftClayBg,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Outlined.ErrorOutline,
                                contentDescription = null,
                                tint = SoftClayText,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = healthError,
                                color = SoftClayText,
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (onSaveUrl(inputUrl)) {
                        onDismiss()
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = SagePrimary),
                shape = RoundedCornerShape(50)
            ) {
                Text("保存")
            }
        },
        dismissButton = {
            OutlinedButton(
                onClick = onDismiss,
                shape = RoundedCornerShape(50),
                border = BorderStroke(1.dp, SketchStroke)
            ) {
                Text("取消", color = InkSecondary)
            }
        }
    )
}
