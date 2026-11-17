package com.jlu.facialanalysis.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.jlu.facialanalysis.R
import com.jlu.facialanalysis.data.model.RejectionInfo
import com.jlu.facialanalysis.data.model.SelectionPayload
import com.jlu.facialanalysis.ui.theme.CardCream
import com.jlu.facialanalysis.ui.theme.HandDrawnLeafDivider
import com.jlu.facialanalysis.ui.theme.InkPrimary
import com.jlu.facialanalysis.ui.theme.InkSecondary
import com.jlu.facialanalysis.ui.theme.SageDeep
import com.jlu.facialanalysis.ui.theme.SageMist
import com.jlu.facialanalysis.ui.theme.SagePrimary
import com.jlu.facialanalysis.ui.theme.SketchStroke
import com.jlu.facialanalysis.ui.theme.SoftClayText

/**
 * 在单一手绘框内呈现「未识别到人脸 / 画面需重拍」温和提示
 */
@Composable
fun UnifiedRejectedContent(
    rejection: RejectionInfo,
    photoModel: Any?
) {
    val isNoHumanFace = rejection.code == "NO_HUMAN_FACE"
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (photoModel != null && !isNoHumanFace) {
            Surface(
                shape = RoundedCornerShape(20.dp),
                border = BorderStroke(1.dp, SketchStroke),
                modifier = Modifier
                    .fillMaxWidth(0.72f)
                    .aspectRatio(0.88f)
            ) {
                AsyncImage(
                    model = photoModel,
                    contentDescription = "拍摄画面",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            }
        } else {
            Image(
                painter = painterResource(id = R.drawable.illus_empty_gentle),
                contentDescription = "未识别到人脸插画",
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .size(196.dp)
                    .clip(RoundedCornerShape(22.dp))
            )
        }

        Spacer(modifier = Modifier.height(16.dp))
        HandDrawnLeafDivider()
        Spacer(modifier = Modifier.height(14.dp))

        Text(
            text = rejection.message,
            style = MaterialTheme.typography.titleLarge.copy(
                fontFamily = FontFamily.Serif,
                fontWeight = FontWeight.SemiBold,
                color = if (isNoHumanFace) SoftClayText else InkPrimary
            ),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(10.dp))

        val gentleNote = if (isNoHumanFace) {
            "画面里没有找到清晰的人类面孔，小动物、玩偶或风景是没法测年龄和颜值的哦，对准正脸再来一张吧。"
        } else {
            "这张照片光线或角度还不够理想，调整到光线柔和处正对镜头再拍一张会更准确。"
        }
        Text(
            text = gentleNote,
            style = MaterialTheme.typography.bodyMedium.copy(color = InkSecondary),
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
    }
}

/**
 * 多人合照时，在单一手绘框内轻点选择分析对象
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun UnifiedFaceSelectionContent(
    selection: SelectionPayload,
    photoModel: Any?,
    isSubmitting: Boolean,
    onSelectFace: (String) -> Unit
) {
    val ratio = (selection.imageWidth.toFloat() / selection.imageHeight.coerceAtLeast(1).toFloat())
        .coerceIn(0.65f, 1.5f)

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(ratio)
                .clip(RoundedCornerShape(18.dp))
        ) {
            AsyncImage(
                model = photoModel,
                contentDescription = "合照人物选择",
                contentScale = ContentScale.FillBounds,
                modifier = Modifier.fillMaxSize()
            )
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(selection.candidates, isSubmitting) {
                        if (isSubmitting) return@pointerInput
                        detectTapGestures { tapOffset ->
                            val normX = tapOffset.x / size.width.coerceAtLeast(1)
                            val normY = tapOffset.y / size.height.coerceAtLeast(1)
                            val hit = selection.candidates.firstOrNull { cand ->
                                cand.bboxNorm.size == 4 &&
                                    normX in cand.bboxNorm[0]..cand.bboxNorm[2] &&
                                    normY in cand.bboxNorm[1]..cand.bboxNorm[3]
                            }
                            if (hit != null) {
                                onSelectFace(hit.faceId)
                            }
                        }
                    }
            ) {
                selection.candidates.forEach { cand ->
                    if (cand.bboxNorm.size == 4) {
                        val left = cand.bboxNorm[0] * size.width
                        val top = cand.bboxNorm[1] * size.height
                        val w = (cand.bboxNorm[2] - cand.bboxNorm[0]) * size.width
                        val h = (cand.bboxNorm[3] - cand.bboxNorm[1]) * size.height
                        drawRoundRect(
                            color = CardCream,
                            topLeft = Offset(left, top),
                            size = Size(w, h),
                            cornerRadius = CornerRadius(18f, 18f),
                            style = Stroke(width = 6f)
                        )
                        drawRoundRect(
                            color = SagePrimary,
                            topLeft = Offset(left, top),
                            size = Size(w, h),
                            cornerRadius = CornerRadius(18f, 18f),
                            style = Stroke(width = 3.5f)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))
        HandDrawnLeafDivider()
        Spacer(modifier = Modifier.height(10.dp))

        Text(
            text = "画面中有多位人物，轻点照片中的人脸框即可分析：",
            style = MaterialTheme.typography.bodyMedium.copy(color = InkSecondary),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(10.dp))

        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterHorizontally),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            selection.candidates.forEachIndexed { idx, cand ->
                FilledTonalButton(
                    onClick = { onSelectFace(cand.faceId) },
                    enabled = !isSubmitting,
                    colors = ButtonDefaults.filledTonalButtonColors(
                        containerColor = SageMist,
                        contentColor = SageDeep
                    ),
                    shape = RoundedCornerShape(50)
                ) {
                    Text("分析第 ${idx + 1} 位人物")
                }
            }
        }
    }
}
