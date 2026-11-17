package com.jlu.facialanalysis.data.offline

import android.content.Context
import com.jlu.facialanalysis.data.model.ApiJsonParser
import com.jlu.facialanalysis.data.model.TaskResult

data class OfflineDemoSummary(
    val id: String,
    val title: String,
    val subtitle: String,
    val badge: String,
    val assetJsonPath: String
)

object OfflineDemoRepository {

    val DEMO_ITEMS = listOf(
        OfflineDemoSummary(
            id = "demo_female_ok",
            title = "人物示例一（青年女性）",
            subtitle = "识别年龄、颜值评分与完整面容气质描述",
            badge = "人像示例",
            assetJsonPath = "offline_demos/demo_female_ok.json"
        ),
        OfflineDemoSummary(
            id = "demo_male_ok",
            title = "人物示例二（长者肖像）",
            subtitle = "不同年龄层人像的面容特征与气质描述",
            badge = "人像示例",
            assetJsonPath = "offline_demos/demo_male_ok.json"
        ),
        OfflineDemoSummary(
            id = "demo_dog_rejected",
            title = "非人脸示例（小狗照片）",
            subtitle = "自动识别非人类面孔并提示“未识别到人脸”",
            badge = "未识别到人脸",
            assetJsonPath = "offline_demos/demo_dog_rejected.json"
        ),
        OfflineDemoSummary(
            id = "demo_profile_rejected",
            title = "模糊画面示例（失焦重拍提示）",
            subtitle = "画面不够清晰时温和提示重拍",
            badge = "清晰度提示",
            assetJsonPath = "offline_demos/demo_profile_rejected.json"
        )
    )

    fun loadOfflineDemo(context: Context, item: OfflineDemoSummary): TaskResult {
        val jsonText = context.assets.open(item.assetJsonPath).bufferedReader(Charsets.UTF_8).use {
            it.readText()
        }
        return ApiJsonParser.parseTaskResult(jsonText)
    }
}
