package com.jlu.facialanalysis.ui.theme

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.jlu.facialanalysis.R

// 「随心一拍」低饱和浅绿 × 暖米水彩手绘色板
val PaperBg = Color(0xFFF6F7F0)
val CardCream = Color(0xFFFCFDF9)
val SagePrimary = Color(0xFF6E9275)
val SageDeep = Color(0xFF496850)
val SageMist = Color(0xFFE4EFE4)
val SageSoftWash = Color(0xFFEDF4EC)
val SketchStroke = Color(0xFFB5C8B6)
val SketchInnerStroke = Color(0xFFD6E2D5)
val InkPrimary = Color(0xFF2C372F)
val InkSecondary = Color(0xFF637366)
val WarmSandBg = Color(0xFFF1ECE1)
val WarmSandText = Color(0xFF6B5B45)
val SoftClayBg = Color(0xFFF5EAE5)
val SoftClayText = Color(0xFF874E42)

// Backward-compatible semantic aliases
val PrimaryBlue = SagePrimary
val PrimaryBlueDark = SageDeep
val PrimaryBlueContainer = SageMist
val TealAccent = SageDeep
val AmberWarningBg = WarmSandBg
val AmberWarningText = WarmSandText
val DangerRedBg = SoftClayBg
val DangerRedText = SoftClayText
val SuccessGreenBg = SageMist
val SuccessGreenText = SageDeep
val NeutralSurface = PaperBg
val CardBorder = SketchStroke

private val AppColorScheme = lightColorScheme(
    primary = SagePrimary,
    onPrimary = CardCream,
    primaryContainer = SageMist,
    onPrimaryContainer = SageDeep,
    secondary = SageDeep,
    onSecondary = CardCream,
    background = PaperBg,
    onBackground = InkPrimary,
    surface = CardCream,
    onSurface = InkPrimary,
    surfaceVariant = SageSoftWash,
    onSurfaceVariant = InkSecondary,
    outline = SketchStroke,
    error = SoftClayText
)

private val AppTypography = Typography(
    headlineMedium = TextStyle(
        fontFamily = FontFamily.Serif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 26.sp,
        lineHeight = 34.sp,
        letterSpacing = 1.2.sp,
        color = InkPrimary
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.Serif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 22.sp,
        lineHeight = 30.sp,
        letterSpacing = 1.sp,
        color = InkPrimary
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.Serif,
        fontWeight = FontWeight.Medium,
        fontSize = 18.sp,
        lineHeight = 26.sp,
        letterSpacing = 0.6.sp,
        color = InkPrimary
    ),
    bodyLarge = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 28.sp,
        letterSpacing = 0.4.sp,
        color = InkPrimary
    ),
    bodyMedium = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 15.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.3.sp,
        color = InkPrimary
    ),
    bodySmall = TextStyle(
        fontWeight = FontWeight.Normal,
        fontSize = 13.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.2.sp,
        color = InkSecondary
    )
)

@Composable
fun FacialAnalysisTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = AppColorScheme,
        typography = AppTypography,
        content = content
    )
}

/**
 * 全屏水彩植物手绘宣纸背景
 */
@Composable
fun BotanicalPaperBackground(
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(PaperBg)
    ) {
        Image(
            painter = painterResource(id = R.drawable.bg_botanical_paper),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            alpha = 0.92f,
            modifier = Modifier.fillMaxSize()
        )
        content()
    }
}

/**
 * 手绘双线信笺外框：柔和错位手绘线 + 暖米纸底，用于承载单一完整报告框
 */
@Composable
fun HandDrawnJournalBox(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 26.dp,
    content: @Composable BoxScope.() -> Unit
) {
    Surface(
        color = CardCream.copy(alpha = 0.94f),
        shape = RoundedCornerShape(cornerRadius),
        shadowElevation = 2.dp,
        border = BorderStroke(1.2.dp, SketchStroke),
        modifier = modifier.drawBehind {
            val r = cornerRadius.toPx()
            // 手绘错位外衬浅绿水彩笔触感圆角线
            drawRoundRect(
                color = SageMist.copy(alpha = 0.65f),
                topLeft = Offset(4.dp.toPx(), 5.dp.toPx()),
                size = Size(size.width, size.height),
                cornerRadius = CornerRadius(r, r),
                style = Stroke(width = 2.dp.toPx())
            )
            drawRoundRect(
                color = SketchInnerStroke,
                topLeft = Offset(4.dp.toPx(), 4.dp.toPx()),
                size = Size(
                    (size.width - 8.dp.toPx()).coerceAtLeast(0f),
                    (size.height - 8.dp.toPx()).coerceAtLeast(0f)
                ),
                cornerRadius = CornerRadius((r - 4.dp.toPx()).coerceAtLeast(8f)),
                style = Stroke(
                    width = 1.dp.toPx(),
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 8f), 0f)
                )
            )
        }
    ) {
        Box(
            modifier = Modifier.padding(20.dp),
            content = content
        )
    }
}

/**
 * 手绘细枝嫩叶分割线
 */
@Composable
fun HandDrawnLeafDivider(modifier: Modifier = Modifier) {
    androidx.compose.foundation.Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(16.dp)
    ) {
        val midY = size.height / 2f
        val centerX = size.width / 2f
        val gap = 22.dp.toPx()

        drawLine(
            color = SketchStroke.copy(alpha = 0.75f),
            start = Offset(8.dp.toPx(), midY),
            end = Offset((centerX - gap).coerceAtLeast(8.dp.toPx()), midY),
            strokeWidth = 1.1.dp.toPx(),
            cap = StrokeCap.Round
        )
        drawLine(
            color = SketchStroke.copy(alpha = 0.75f),
            start = Offset((centerX + gap).coerceAtMost(size.width - 8.dp.toPx()), midY),
            end = Offset(size.width - 8.dp.toPx(), midY),
            strokeWidth = 1.1.dp.toPx(),
            cap = StrokeCap.Round
        )

        // 中央两片小手绘嫩叶
        val leftLeaf = Path().apply {
            moveTo(centerX, midY + 1.dp.toPx())
            quadraticBezierTo(centerX - 8.dp.toPx(), midY - 6.dp.toPx(), centerX - 13.dp.toPx(), midY)
            quadraticBezierTo(centerX - 7.dp.toPx(), midY + 5.dp.toPx(), centerX, midY + 1.dp.toPx())
        }
        val rightLeaf = Path().apply {
            moveTo(centerX, midY + 1.dp.toPx())
            quadraticBezierTo(centerX + 8.dp.toPx(), midY - 6.dp.toPx(), centerX + 13.dp.toPx(), midY)
            quadraticBezierTo(centerX + 7.dp.toPx(), midY + 5.dp.toPx(), centerX, midY + 1.dp.toPx())
        }
        drawPath(leftLeaf, color = SageMist)
        drawPath(leftLeaf, color = SagePrimary, style = Stroke(width = 1.1.dp.toPx()))
        drawPath(rightLeaf, color = SageMist)
        drawPath(rightLeaf, color = SagePrimary, style = Stroke(width = 1.1.dp.toPx()))
    }
}
