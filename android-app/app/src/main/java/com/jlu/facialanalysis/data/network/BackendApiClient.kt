package com.jlu.facialanalysis.data.network

import com.jlu.facialanalysis.data.model.ApiJsonParser
import com.jlu.facialanalysis.data.model.HealthStatus
import com.jlu.facialanalysis.data.model.TaskResult
import com.jlu.facialanalysis.data.model.TaskSubmitResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class ApiException(
    val httpCode: Int,
    val errorCode: String,
    override val message: String
) : IOException(message)

class BackendApiClient(
    private val httpClient: OkHttpClient = defaultOkHttpClient()
) {

    companion object {
        const val MAX_UPLOAD_BYTES = 8 * 1024 * 1024 // 8 MB 契约上限
        const val DEFAULT_USB_BASE_URL = "http://127.0.0.1:8000"

        fun defaultOkHttpClient(): OkHttpClient =
            OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(15, TimeUnit.SECONDS)
                .writeTimeout(15, TimeUnit.SECONDS)
                .followRedirects(false)
                .followSslRedirects(false)
                .build()

        fun isLoopbackOrPrivateHost(rawHost: String): Boolean {
            val host = rawHost.trim().lowercase().removePrefix("[").removeSuffix("]")
            if (host == "localhost" || host == "::1" || host == "0:0:0:0:0:0:0:1") {
                return true
            }
            val parts = host.split(".")
            if (parts.size != 4) return false
            val octets = parts.map { it.toIntOrNull() ?: return false }
            if (octets.any { it !in 0..255 }) return false
            val a = octets[0]
            val b = octets[1]
            return when {
                a == 127 -> true // 127.0.0.0/8 loopback
                a == 10 -> true // 10.0.0.0/8 RFC1918 & 10.0.2.2 emulator
                a == 192 && b == 168 -> true // 192.168.0.0/16 hotspot/LAN
                a == 172 && b in 16..31 -> true // 172.16.0.0/12 hotspot/LAN
                a == 169 && b == 254 -> true // 169.254.0.0/16 link-local
                else -> false
            }
        }

        fun normalizeBaseUrl(raw: String): String {
            val trimmed = raw.trim().trimEnd('/')
            val withScheme = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
                trimmed
            } else {
                "http://$trimmed"
            }
            val parsed = withScheme.toHttpUrlOrNull()
                ?: throw ApiException(400, "INVALID_BASE_URL", "后端服务地址格式无效：$raw")
            if (!isLoopbackOrPrivateHost(parsed.host)) {
                throw ApiException(
                    403,
                    "PUBLIC_HOST_FORBIDDEN",
                    "隐私闭环限制：仅允许连接本地回环（127.0.0.1/localhost）或课堂局域网私网 IP，禁止向公网地址传输图像（${parsed.host}）"
                )
            }
            return parsed.toString().trimEnd('/')
        }

        fun resolveArtifactUrl(baseUrl: String, pathOrUrl: String?): String? {
            if (pathOrUrl.isNullOrBlank()) return null
            if (pathOrUrl.startsWith("file:///android_asset/")) {
                return pathOrUrl
            }
            if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
                val parsed = pathOrUrl.toHttpUrlOrNull() ?: return null
                return if (isLoopbackOrPrivateHost(parsed.host)) parsed.toString() else null
            }
            if (pathOrUrl.startsWith("file:///")) {
                return null
            }
            val cleanBase = runCatching { normalizeBaseUrl(baseUrl) }.getOrNull() ?: return null
            val cleanPath = if (pathOrUrl.startsWith("/")) pathOrUrl else "/$pathOrUrl"
            return "$cleanBase$cleanPath"
        }
    }

    suspend fun checkHealth(baseUrl: String): HealthStatus = withContext(Dispatchers.IO) {
        val url = "${normalizeBaseUrl(baseUrl)}/api/v1/health"
        val req = Request.Builder().url(url).get().build()
        httpClient.newCall(req).execute().use { resp ->
            val body = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                throw parseApiError(resp.code, body)
            }
            ApiJsonParser.parseHealth(body)
        }
    }

    suspend fun submitImageTask(
        baseUrl: String,
        imageBytes: ByteArray,
        mimeType: String = "image/jpeg",
        clientRequestId: String? = null
    ): TaskSubmitResponse = withContext(Dispatchers.IO) {
        if (imageBytes.isEmpty()) {
            throw ApiException(400, "EMPTY_IMAGE", "图像数据为空，请重新拍摄或选择照片")
        }
        if (imageBytes.size > MAX_UPLOAD_BYTES) {
            throw ApiException(413, "IMAGE_TOO_LARGE", "图片大小超过 8 MB 限制，请重新选择")
        }
        val baseHttpUrl = "${normalizeBaseUrl(baseUrl)}/api/v1/tasks".toHttpUrlOrNull()
            ?: throw ApiException(400, "INVALID_BASE_URL", "后端服务地址格式无效：$baseUrl")

        val finalUrl = baseHttpUrl.newBuilder().apply {
            if (!clientRequestId.isNullOrBlank()) {
                addQueryParameter("client_request_id", clientRequestId)
            }
        }.build()

        val reqBody = imageBytes.toRequestBody(mimeType.toMediaType())
        val req = Request.Builder()
            .url(finalUrl)
            .post(reqBody)
            .header("Accept", "application/json")
            .build()

        httpClient.newCall(req).execute().use { resp ->
            val body = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                throw parseApiError(resp.code, body)
            }
            ApiJsonParser.parseSubmitResponse(body)
        }
    }

    suspend fun selectCandidateFace(
        baseUrl: String,
        taskId: String,
        imageId: String,
        faceId: String
    ): Unit = withContext(Dispatchers.IO) {
        val url = "${normalizeBaseUrl(baseUrl)}/api/v1/tasks/$taskId/select-face"
        val payload = JSONObject()
            .put("image_id", imageId)
            .put("face_id", faceId)
            .toString()
        val req = Request.Builder()
            .url(url)
            .post(payload.toRequestBody("application/json; charset=utf-8".toMediaType()))
            .build()

        httpClient.newCall(req).execute().use { resp ->
            val body = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                throw parseApiError(resp.code, body)
            }
        }
    }

    suspend fun getTaskStatus(baseUrl: String, taskId: String): TaskResult =
        withContext(Dispatchers.IO) {
            val url = "${normalizeBaseUrl(baseUrl)}/api/v1/tasks/$taskId"
            val req = Request.Builder().url(url).get().build()
            httpClient.newCall(req).execute().use { resp ->
                val body = resp.body?.string().orEmpty()
                if (!resp.isSuccessful) {
                    throw parseApiError(resp.code, body)
                }
                ApiJsonParser.parseTaskResult(body)
            }
        }

    suspend fun deleteTaskQuietly(baseUrl: String, taskId: String) = withContext(Dispatchers.IO) {
        if (taskId.isBlank() || taskId.startsWith("offline_")) return@withContext
        runCatching {
            val url = "${normalizeBaseUrl(baseUrl)}/api/v1/tasks/$taskId"
            val req = Request.Builder().url(url).delete().build()
            httpClient.newCall(req).execute().close()
        }
    }

    private fun parseApiError(httpCode: Int, rawBody: String): ApiException {
        return try {
            val obj = JSONObject(rawBody)
            val detailObj = obj.optJSONObject("detail")
            val code = detailObj?.optString("error_code")
                ?: obj.optString("error_code", "HTTP_$httpCode")
            val msg = detailObj?.optString("message")
                ?: obj.optString("message", obj.optString("detail", "请求失败（HTTP $httpCode）"))
            ApiException(httpCode, code, msg)
        } catch (_: Exception) {
            ApiException(httpCode, "HTTP_$httpCode", "连接后端服务失败（HTTP $httpCode）")
        }
    }
}
