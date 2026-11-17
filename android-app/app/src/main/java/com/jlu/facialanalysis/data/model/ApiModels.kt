package com.jlu.facialanalysis.data.model

import org.json.JSONArray
import org.json.JSONObject

data class ModelHealthItem(
    val name: String,
    val loaded: Boolean,
    val version: String,
    val weightsStatus: String?
)

data class HealthStatus(
    val status: String,
    val warmedUp: Boolean,
    val device: String,
    val queueLength: Int,
    val queueCapacity: Int,
    val ttlSeconds: Int,
    val models: List<ModelHealthItem>
)

data class TaskSubmitResponse(
    val taskId: String,
    val imageId: String,
    val status: String,
    val queuePosition: Int,
    val createdAt: String
)

data class RejectionInfo(
    val code: String,
    val message: String,
    val detail: String?,
    val previewImageUrl: String? = null
)

data class CandidateFace(
    val faceId: String,
    val bboxPx: List<Int>,
    val bboxNorm: List<Float>,
    val score: Float
)

data class SelectionPayload(
    val imageWidth: Int,
    val imageHeight: Int,
    val previewImageUrl: String,
    val candidates: List<CandidateFace>
)

data class PoseDeg(
    val yaw: Float,
    val pitch: Float,
    val roll: Float
)

data class TargetFaceInfo(
    val faceId: String,
    val imageWidth: Int,
    val imageHeight: Int,
    val bboxPx: List<Int>,
    val bboxNorm: List<Float>,
    val poseDeg: PoseDeg?,
    val blurLaplacianVar: Float?,
    val brightnessMean: Float?
)

data class ScoresPayload(
    val beautyScore: Float,
    val beautyDisplay: Float,
    val scaleMin: Int,
    val scaleMax: Int,
    val distributionLabel: String,
    val ratingProbabilities: List<Float>,
    val distributionStd: Float?,
    val apparentAge: Float,
    val apparentAgeDisplay: Int,
    val weightsReady: Boolean,
    val weightsStatus: String = "untrained_init"
)

data class FeatureFieldObservation(
    val key: String,
    val label: String,
    val description: String,
    val visible: Boolean
)

data class TextAnalysisPayload(
    val status: String,
    val errorMessage: String?,
    val fields: List<FeatureFieldObservation>,
    val narrative: String? = null
)

data class TaskResult(
    val taskId: String,
    val imageId: String,
    val status: String,
    val isOfflineReplay: Boolean,
    val modelVersionsSummary: String?,
    val generatedAt: String?,
    val originalImageUrl: String?,
    val rejection: RejectionInfo?,
    val selection: SelectionPayload?,
    val targetFace: TargetFaceInfo?,
    val scores: ScoresPayload?,
    val textAnalysis: TextAnalysisPayload?,
    val ethicsNotice: String,
    val timingsMs: Map<String, Float>
) {
    val isTerminal: Boolean
        get() = status in setOf("rejected", "need_selection", "completed", "partial_error")

    fun buildUnifiedDescription(): String {
        val explicit = textAnalysis?.narrative?.trim()
        if (!explicit.isNullOrEmpty()) return explicit

        val sc = scores ?: return ""
        val age = sc.apparentAgeDisplay
        val beauty = sc.beautyDisplay
        val vibe = when {
            sc.beautyScore >= 4.0f -> "五官与骨相协调感十分出众，整体气质清亮灵动"
            sc.beautyScore >= 3.4f -> "五官比例舒展协调，整体气质清爽耐看"
            sc.beautyScore >= 2.8f -> "面容线条自然大方，整体观感亲切温和"
            else -> "面容神态真实松弛，整体风格质朴自然"
        }
        val lead = "画面中的人物表观年龄约 $age 岁，颜值评分为 ${"%.1f".format(beauty)} 分（满分 5.0 分），$vibe。"

        val polishMap = mapOf(
            "眼睛睁开，眼裂轮廓可见" to "双眸自然舒展、眼型清晰",
            "眼睛略微眯起" to "眉眼微弯、神态放松",
            "眼睛闭合" to "双目轻阖",
            "眉形较平直" to "眉形平直干净",
            "眉形呈弧形" to "眉形柔和呈自然弧度",
            "眉形有较明显转折" to "眉峰线条利落分明",
            "可见鼻梁线条较直" to "鼻梁线条平直挺括",
            "可见鼻部轮廓较圆润" to "鼻头线条柔和圆润",
            "可见鼻翼轮廓较宽" to "鼻部轮廓舒展大气",
            "双唇闭合" to "双唇轻合",
            "双唇分开" to "双唇微启",
            "嘴部张开，可见牙齿" to "唇齿微露",
            "可见面部轮廓偏椭圆" to "脸部轮廓柔和偏椭圆",
            "可见面部轮廓偏圆" to "脸部线条圆润饱满",
            "可见下颌轮廓有较明显转折" to "下颌轮廓清晰利落",
            "可见面部轮廓偏长" to "脸部线条修长舒展",
            "可见嘴角上扬，呈微笑表情" to "唇角轻扬带着浅浅笑意",
            "可见表情较平静" to "神态从容平和",
            "可见眉间收紧" to "眉宇间略带专注神色"
        )
        val order = listOf("contour", "brows", "eyes", "nose", "mouth", "expression")
        val byKey = textAnalysis?.fields?.associateBy { it.key }.orEmpty()
        val clauses = order.mapNotNull { k ->
            val item = byKey[k]
            if (item != null && item.visible && item.description.isNotBlank() && item.description != "无法判断") {
                val trimmed = item.description.trim().trimEnd('。', '！', '；', '，', ',', '.', ' ')
                polishMap[trimmed] ?: trimmed.replace(Regex("^可见(?=面部|脸部|下颌|鼻|眉|眼|双唇|嘴)"), "")
            } else {
                null
            }
        }
        return if (clauses.isEmpty()) {
            if (!isTerminal) "${lead}正在细细端详五官与神态细节…" else lead
        } else {
            "${lead}细看五官细节，${clauses.joinToString("，")}，整体面容舒展和谐。"
        }
    }
}

object ApiJsonParser {

    fun parseHealth(jsonStr: String): HealthStatus {
        val root = JSONObject(jsonStr)
        val modelsObj = root.optJSONObject("models") ?: JSONObject()
        val items = mutableListOf<ModelHealthItem>()
        val keys = modelsObj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            val item = modelsObj.optJSONObject(k) ?: continue
            items.add(
                ModelHealthItem(
                    name = k,
                    loaded = item.optBoolean("loaded", false),
                    version = item.optString("version", item.optString("source", "-")),
                    weightsStatus = item.optString("weights_status", "").ifBlank { null }
                )
            )
        }
        return HealthStatus(
            status = root.optString("status", "unknown"),
            warmedUp = root.optBoolean("warmed_up", false),
            device = root.optString("device", "unknown"),
            queueLength = root.optInt("queue_length", 0),
            queueCapacity = root.optInt("queue_capacity", 0),
            ttlSeconds = root.optInt("ttl_seconds", 120),
            models = items
        )
    }

    fun parseSubmitResponse(jsonStr: String): TaskSubmitResponse {
        val root = JSONObject(jsonStr)
        return TaskSubmitResponse(
            taskId = root.getString("task_id"),
            imageId = root.optString("image_id", ""),
            status = root.optString("status", "queued"),
            queuePosition = root.optInt("queue_position", 0),
            createdAt = root.optString("created_at", "")
        )
    }

    fun parseTaskResult(jsonStr: String): TaskResult {
        val root = JSONObject(jsonStr)

        val rejection = root.optJSONObject("rejection")?.let { rej ->
            val rawCode = rej.optString("code", "REJECTED")
            val defaultMsg = if (rawCode == "NO_HUMAN_FACE") "未识别到人脸" else "图像未通过分析门禁"
            RejectionInfo(
                code = rawCode,
                message = rej.optString("message", defaultMsg).ifBlank { defaultMsg },
                detail = rej.optString("detail", "").ifBlank { null },
                previewImageUrl = rej.optString("preview_image_url", "").ifBlank { null }
            )
        }

        val selection = root.optJSONObject("selection")?.let { sel ->
            val cands = mutableListOf<CandidateFace>()
            val arr = sel.optJSONArray("candidates") ?: JSONArray()
            for (i in 0 until arr.length()) {
                val c = arr.getJSONObject(i)
                cands.add(
                    CandidateFace(
                        faceId = c.getString("face_id"),
                        bboxPx = c.optJSONArray("bbox_px").toIntList(),
                        bboxNorm = c.optJSONArray("bbox_norm").toFloatList(),
                        score = c.optDouble("score", 0.0).toFloat()
                    )
                )
            }
            SelectionPayload(
                imageWidth = sel.optInt("image_width", 1),
                imageHeight = sel.optInt("image_height", 1),
                previewImageUrl = sel.optString("preview_image_url", ""),
                candidates = cands
            )
        }

        val targetFace = root.optJSONObject("target_face")?.let { tf ->
            val pose = tf.optJSONObject("pose_deg")?.let { p ->
                PoseDeg(
                    yaw = p.optDouble("yaw", 0.0).toFloat(),
                    pitch = p.optDouble("pitch", 0.0).toFloat(),
                    roll = p.optDouble("roll", 0.0).toFloat()
                )
            }
            TargetFaceInfo(
                faceId = tf.optString("face_id", "face_0"),
                imageWidth = tf.optInt("image_width", 0),
                imageHeight = tf.optInt("image_height", 0),
                bboxPx = tf.optJSONArray("bbox_px").toIntList(),
                bboxNorm = tf.optJSONArray("bbox_norm").toFloatList(),
                poseDeg = pose,
                blurLaplacianVar = tf.optNullableFloat("blur_laplacian_var"),
                brightnessMean = tf.optNullableFloat("brightness_mean")
            )
        }

        val scores = root.optJSONObject("scores")?.let { sc ->
            val rawBeauty = sc.optDouble("beauty_score", 0.0).toFloat()
            val rawAge = sc.optDouble("apparent_age", 0.0).toFloat()
            val ready = sc.optBoolean("weights_ready", false)
            val statusStr = sc.optString(
                "weights_status",
                if (ready) "trained_fold1" else "untrained_init"
            )
            ScoresPayload(
                beautyScore = rawBeauty,
                beautyDisplay = sc.optDouble("beauty_display", rawBeauty.toDouble()).toFloat(),
                scaleMin = sc.optInt("scale_min", 1),
                scaleMax = sc.optInt("scale_max", 5),
                distributionLabel = sc.optString("distribution_label", "模型预测评分分布（1—5 档）"),
                ratingProbabilities = sc.optJSONArray("rating_probabilities").toFloatList(),
                distributionStd = sc.optNullableFloat("distribution_std"),
                apparentAge = rawAge,
                apparentAgeDisplay = sc.optInt("apparent_age_display", kotlin.math.round(rawAge).toInt()),
                weightsReady = ready,
                weightsStatus = statusStr
            )
        }

        val textAnalysis = root.optJSONObject("text_analysis")?.let { ta ->
            val orderedKeys = listOf(
                "brows" to "眉部",
                "eyes" to "眼部",
                "nose" to "鼻部",
                "mouth" to "嘴部",
                "contour" to "脸部轮廓",
                "expression" to "可见表情"
            )
            val fieldsObj = ta.optJSONObject("fields") ?: JSONObject()
            val observations = orderedKeys.mapNotNull { (k, fallbackLabel) ->
                val f = fieldsObj.optJSONObject(k) ?: return@mapNotNull null
                FeatureFieldObservation(
                    key = k,
                    label = f.optString("label", fallbackLabel),
                    description = f.optString("description", "无法判断"),
                    visible = f.optBoolean("visible", true)
                )
            }
            TextAnalysisPayload(
                status = ta.optString("status", "ok"),
                errorMessage = ta.optString("error_message", "").ifBlank { null },
                fields = observations,
                narrative = ta.optString("narrative", "").takeIf { it.isNotBlank() && it != "null" }
            )
        }

        val versionsSummary = root.optJSONObject("model_versions")?.let { mv ->
            listOfNotNull(
                mv.optString("beauty", "").takeIf { it.isNotBlank() },
                mv.optString("age", "").takeIf { it.isNotBlank() },
                mv.optString("text", "").takeIf { it.isNotBlank() }
            ).joinToString(" · ")
        }

        val timings = mutableMapOf<String, Float>()
        root.optJSONObject("timings_ms")?.let { tm ->
            val keys = tm.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                timings[k] = tm.optDouble(k, 0.0).toFloat()
            }
        }

        return TaskResult(
            taskId = root.optString("task_id", ""),
            imageId = root.optString("image_id", ""),
            status = root.optString("status", "processing"),
            isOfflineReplay = root.optBoolean("is_offline_replay", false),
            modelVersionsSummary = versionsSummary,
            generatedAt = root.optString("generated_at", "").ifBlank { null },
            originalImageUrl = root.optString("original_image_url", "").takeIf { it.isNotBlank() && it != "null" },
            rejection = rejection,
            selection = selection,
            targetFace = targetFace,
            scores = scores,
            textAnalysis = textAnalysis,
            ethicsNotice = root.optString(
                "ethics_notice",
                "颜值评分与表观年龄仅反映特定公开数据集标注群体的主观统计趋势，不代表客观美丑标准或个人价值；图像仅在本地设备内存处理并定时销毁。"
            ),
            timingsMs = timings
        )
    }

    private fun JSONArray?.toIntList(): List<Int> {
        if (this == null) return emptyList()
        return (0 until length()).map { optInt(it, 0) }
    }

    private fun JSONArray?.toFloatList(): List<Float> {
        if (this == null) return emptyList()
        return (0 until length()).map { optDouble(it, 0.0).toFloat() }
    }

    private fun JSONObject.optNullableFloat(key: String): Float? {
        if (!has(key) || isNull(key)) return null
        val v = optDouble(key, Double.NaN)
        return if (v.isNaN()) null else v.toFloat()
    }
}
