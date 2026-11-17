package com.jlu.facialanalysis

import com.jlu.facialanalysis.data.model.ApiJsonParser
import com.jlu.facialanalysis.data.model.GenderViewKey
import com.jlu.facialanalysis.data.network.BackendApiClient
import com.jlu.facialanalysis.vision.PreviewCoordinateMapper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ApiContractAndMapperTest {

    @Test
    fun `dog face rejection parses NO_HUMAN_FACE with exact message and null scores`() {
        val json = """
            {
              "task_id": "tsk_dog_001",
              "image_id": "img_dog_001",
              "status": "rejected",
              "is_offline_replay": false,
              "rejection": {
                "code": "NO_HUMAN_FACE",
                "message": "未识别到人脸",
                "detail": "0 human candidates"
              },
              "scores": null,
              "geometry": null,
              "explanation": null,
              "text_analysis": null
            }
        """.trimIndent()

        val parsed = ApiJsonParser.parseTaskResult(json)
        assertEquals("rejected", parsed.status)
        assertTrue(parsed.isTerminal)
        assertNotNull(parsed.rejection)
        assertEquals("NO_HUMAN_FACE", parsed.rejection?.code)
        assertEquals("未识别到人脸", parsed.rejection?.message)
        assertNull(parsed.scores)
        assertNull(parsed.geometry)
        assertNull(parsed.textAnalysis)
    }

    @Test
    fun `multi face selection parses candidates and remains terminal for user action`() {
        val json = """
            {
              "task_id": "tsk_multi_01",
              "image_id": "img_multi_01",
              "status": "need_selection",
              "selection": {
                "image_width": 960,
                "image_height": 1280,
                "preview_image_url": "/api/v1/tasks/tsk_multi_01/artifacts/normalized.jpg",
                "candidates": [
                  {"face_id": "face_0", "bbox_px": [100, 200, 300, 450], "bbox_norm": [0.1, 0.2, 0.3, 0.45], "score": 0.94},
                  {"face_id": "face_1", "bbox_px": [500, 210, 720, 460], "bbox_norm": [0.5, 0.21, 0.72, 0.46], "score": 0.91}
                ]
              }
            }
        """.trimIndent()

        val parsed = ApiJsonParser.parseTaskResult(json)
        assertEquals("need_selection", parsed.status)
        assertTrue(parsed.isTerminal)
        assertEquals(2, parsed.selection?.candidates?.size)
        assertEquals("face_1", parsed.selection?.candidates?.get(1)?.faceId)
    }

    @Test
    fun `staged scores_ready is non-terminal while partial_error preserves scores`() {
        val stagedJson = """
            {
              "task_id": "tsk_stage_01",
              "image_id": "img_stage_01",
              "status": "scores_ready",
              "scores": {
                "beauty_score": 3.84,
                "beauty_display": 3.8,
                "scale_min": 1,
                "scale_max": 5,
                "rating_probabilities": [0.02, 0.08, 0.24, 0.51, 0.15],
                "apparent_age": 22.4,
                "apparent_age_display": 22,
                "weights_ready": true
              }
            }
        """.trimIndent()

        val staged = ApiJsonParser.parseTaskResult(stagedJson)
        assertFalse(staged.isTerminal)
        assertEquals(3.84f, staged.scores?.beautyScore ?: 0f, 1e-4f)
        assertEquals(22, staged.scores?.apparentAgeDisplay)

        val partialErrJson = """
            {
              "task_id": "tsk_stage_01",
              "image_id": "img_stage_01",
              "status": "partial_error",
              "scores": {
                "beauty_score": 3.84,
                "beauty_display": 3.8,
                "scale_min": 1,
                "scale_max": 5,
                "rating_probabilities": [0.02, 0.08, 0.24, 0.51, 0.15],
                "apparent_age": 22.4,
                "apparent_age_display": 22,
                "weights_ready": true
              },
              "text_analysis": {
                "status": "degraded",
                "error_message": "FaceLLM description timed out",
                "fields": {}
              }
            }
        """.trimIndent()
        val degraded = ApiJsonParser.parseTaskResult(partialErrJson)
        assertTrue(degraded.isTerminal)
        assertNotNull(degraded.scores)
        assertEquals("degraded", degraded.textAnalysis?.status)
    }

    @Test
    fun `artifact url resolver handles relative http and android_asset paths and blocks public hosts`() {
        assertEquals(
            "http://127.0.0.1:8000/api/v1/tasks/t1/artifacts/gradcam_overlay.jpg",
            BackendApiClient.resolveArtifactUrl("127.0.0.1:8000/", "/api/v1/tasks/t1/artifacts/gradcam_overlay.jpg")
        )
        assertEquals(
            "http://192.168.43.100:8000/api/v1/tasks/t1/artifacts/normalized.jpg",
            BackendApiClient.resolveArtifactUrl("http://192.168.43.100:8000", "/api/v1/tasks/t1/artifacts/normalized.jpg")
        )
        assertEquals(
            "file:///android_asset/offline_demos/female_orig.jpg",
            BackendApiClient.resolveArtifactUrl("http://127.0.0.1:8000", "file:///android_asset/offline_demos/female_orig.jpg")
        )
        // Public host must be blocked in both resolveArtifactUrl and normalizeBaseUrl
        assertNull(
            BackendApiClient.resolveArtifactUrl("http://127.0.0.1:8000", "https://example.org/photo.jpg")
        )
        val pubErr = runCatching { BackendApiClient.normalizeBaseUrl("https://example.org") }.exceptionOrNull()
        assertNotNull("Public baseUrl must throw ApiException", pubErr)
    }

    @Test
    fun `preview coordinate mapper mirrors horizontally only for front camera`() {
        val frontPt = PreviewCoordinateMapper.mapNormalizedPointToView(
            normX = 0.25f,
            normY = 0.5f,
            frameWidth = 100,
            frameHeight = 200,
            viewWidth = 100f,
            viewHeight = 200f,
            isFrontCamera = true
        )
        val backPt = PreviewCoordinateMapper.mapNormalizedPointToView(
            normX = 0.25f,
            normY = 0.5f,
            frameWidth = 100,
            frameHeight = 200,
            viewWidth = 100f,
            viewHeight = 200f,
            isFrontCamera = false
        )
        assertEquals(75f, frontPt.x, 1e-3f)
        assertEquals(25f, backPt.x, 1e-3f)
        assertEquals(100f, frontPt.y, 1e-3f)
    }

    @Test
    fun `all 4 bundled offline demo json assets exist and parse cleanly`() {
        val candidates = listOf(
            File("src/main/assets/offline_demos"),
            File("app/src/main/assets/offline_demos")
        )
        val dir = candidates.firstOrNull { it.exists() }
        assertNotNull("offline_demos assets folder must exist", dir)

        val expectedFiles = listOf(
            "demo_female_ok.json",
            "demo_male_ok.json",
            "demo_dog_rejected.json",
            "demo_profile_rejected.json"
        )
        expectedFiles.forEach { name ->
            val file = File(dir!!, name)
            assertTrue("Missing $name", file.exists())
            val parsed = ApiJsonParser.parseTaskResult(file.readText(Charsets.UTF_8))
            assertTrue("Offline replay flag must be true for $name", parsed.isOfflineReplay)
            if (name == "demo_dog_rejected.json") {
                assertEquals("NO_HUMAN_FACE", parsed.rejection?.code)
                assertEquals("未识别到人脸", parsed.rejection?.message)
                assertNull(parsed.scores)
                assertNull(parsed.explanation)
                assertNotNull(parsed.rejection?.previewImageUrl)
            }
            if (name == "demo_female_ok.json" || name == "demo_male_ok.json") {
                assertEquals(3, parsed.genderViews.size)
                assertTrue(parsed.genderViews.containsKey(GenderViewKey.FEMALE))
                assertTrue("Expected trained 2D-FAP weightsReady=true in $name", parsed.scores?.weightsReady == true)
                assertEquals("trained_fold1", parsed.scores?.weightsStatus)
                assertTrue(
                    "Expected real SCUT-FBP5500 confirmed correlation trend in $name",
                    parsed.genderViews[GenderViewKey.FEMALE]?.summary?.contains("训练集确认关联指标") == true
                )
            }
        }
    }

    @Test
    fun `real backend http_smoke json parses cleanly in Kotlin ApiJsonParser`() {
        val smokeCandidates = listOf(
            File("src/test/resources/http_smoke.json"),
            File("app/src/test/resources/http_smoke.json"),
            File("../backend/http_smoke.json"),
            File("../../backend/http_smoke.json")
        )
        val smokeFile = smokeCandidates.firstOrNull { it.exists() }
        assertNotNull("http_smoke.json fixture must exist for cross-stack contract test", smokeFile)
        val root = org.json.JSONObject(smokeFile!!.readText(Charsets.UTF_8))
        val health = ApiJsonParser.parseHealth(root.getJSONObject("health").toString())
        assertEquals("ok", health.status)
        assertTrue(health.warmedUp)
        assertEquals(8, health.queueCapacity)
        assertEquals(120, health.ttlSeconds)
        assertTrue(health.models.any { it.name == "apparent_age" && it.loaded })
        assertTrue(health.models.any { it.name == "beauty_2dfap" && it.loaded && it.weightsStatus == "trained_fold1" })
        assertTrue(health.models.any { it.name == "gender_stats" && it.loaded })

        val result = ApiJsonParser.parseTaskResult(root.getJSONObject("result").toString())
        assertEquals("completed", result.status)
        assertTrue(result.isTerminal)
        assertNotNull(result.scores)
        assertTrue(result.scores?.weightsReady == true)
        assertEquals("trained_fold1", result.scores?.weightsStatus)
        assertEquals(5, result.scores?.ratingProbabilities?.size)
        assertNotNull(result.geometry)
        assertNotNull(result.explanation)
        assertEquals(5, result.explanation?.regionSensitivity?.size)
        assertEquals(6, result.textAnalysis?.fields?.size)
        assertEquals(3, result.genderViews.size)
        val desc = result.buildUnifiedDescription()
        assertTrue("Unified narrative must contain age and beauty score", desc.contains("岁") && desc.contains("分"))
    }
}
