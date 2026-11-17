package com.jlu.facialanalysis.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.jlu.facialanalysis.data.model.HealthStatus
import com.jlu.facialanalysis.data.model.TaskResult
import com.jlu.facialanalysis.data.network.BackendApiClient
import com.jlu.facialanalysis.data.offline.OfflineDemoRepository
import com.jlu.facialanalysis.data.offline.OfflineDemoSummary
import com.jlu.facialanalysis.vision.ImagePreprocessor
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.UUID

enum class AppScreen {
    HOME,
    CAMERA,
    RESULT
}

data class UiState(
    val screen: AppScreen = AppScreen.HOME,
    val baseUrl: String = BackendApiClient.DEFAULT_USB_BASE_URL,
    val showServerDialog: Boolean = false,
    val showOfflineSheet: Boolean = false,
    val isCheckingHealth: Boolean = false,
    val healthStatus: HealthStatus? = null,
    val healthError: String? = null,
    val isSubmitting: Boolean = false,
    val activeTaskId: String? = null,
    val queuePosition: Int = 0,
    val taskResult: TaskResult? = null,
    val networkError: String? = null,
    val clientTimingsMs: Map<String, Long> = emptyMap(),
    val localPreviewBytes: ByteArray? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val PREFS_NAME = "facial_analysis_prefs"
        private const val KEY_BASE_URL = "backend_base_url"
        private const val POLL_INTERVAL_MS = 350L
        private const val MAX_POLL_ROUNDS = 300 // 约 105s 超时上限，对齐后端 120s TTL 与 45s 文字回退上限
    }

    private val prefs = application.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val apiClient = BackendApiClient()

    private val _uiState = MutableStateFlow(
        UiState(
            baseUrl = prefs.getString(KEY_BASE_URL, BackendApiClient.DEFAULT_USB_BASE_URL)
                ?: BackendApiClient.DEFAULT_USB_BASE_URL
        )
    )
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null
    private var requestStartNano: Long = 0L

    private fun monotonicElapsedMs(startNano: Long): Long =
        ((System.nanoTime() - startNano).coerceAtLeast(0L)) / 1_000_000L

    fun openCamera() {
        _uiState.update { it.copy(screen = AppScreen.CAMERA, networkError = null) }
    }

    fun backToHome() {
        cleanupActiveRemoteTask()
        _uiState.update {
            it.copy(
                screen = AppScreen.HOME,
                isSubmitting = false,
                activeTaskId = null,
                taskResult = null,
                networkError = null,
                clientTimingsMs = emptyMap(),
                localPreviewBytes = null
            )
        }
    }

    fun toggleServerDialog(show: Boolean) {
        _uiState.update { it.copy(showServerDialog = show) }
        if (show) {
            runHealthCheck(_uiState.value.baseUrl)
        }
    }

    fun toggleOfflineSheet(show: Boolean) {
        _uiState.update { it.copy(showOfflineSheet = show) }
    }

    fun updateAndSaveBaseUrl(newUrl: String): Boolean {
        val normalized = runCatching {
            BackendApiClient.normalizeBaseUrl(newUrl)
        }.getOrElse { err ->
            _uiState.update {
                it.copy(
                    isCheckingHealth = false,
                    healthStatus = null,
                    healthError = err.message ?: "无效的本地服务地址"
                )
            }
            return false
        }
        prefs.edit().putString(KEY_BASE_URL, normalized).apply()
        _uiState.update { it.copy(baseUrl = normalized) }
        runHealthCheck(normalized)
        return true
    }

    fun runHealthCheck(targetUrl: String = _uiState.value.baseUrl) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(isCheckingHealth = true, healthError = null, healthStatus = null)
            }
            runCatching {
                apiClient.checkHealth(targetUrl)
            }.onSuccess { health ->
                _uiState.update {
                    it.copy(isCheckingHealth = false, healthStatus = health, healthError = null)
                }
            }.onFailure { err ->
                _uiState.update {
                    it.copy(
                        isCheckingHealth = false,
                        healthStatus = null,
                        healthError = err.message ?: "无法连接到后端推理服务，请检查 adb reverse 或热点 IP"
                    )
                }
            }
        }
    }

    fun submitFromGalleryUri(uri: Uri) {
        if (_uiState.value.isSubmitting) return // 防抖锁
        val clickNano = System.nanoTime()
        viewModelScope.launch {
            _uiState.update { it.copy(isSubmitting = true, networkError = null) }
            val imageBytes = runCatching {
                ImagePreprocessor.loadAndNormalizeFromUri(getApplication(), uri)
            }.getOrElse { err ->
                _uiState.update {
                    it.copy(
                        isSubmitting = false,
                        screen = AppScreen.RESULT,
                        networkError = err.message ?: "读取相册图片失败"
                    )
                }
                return@launch
            }
            submitNormalizedImageBytes(imageBytes, clickNano)
        }
    }

    fun submitNormalizedImageBytes(imageBytes: ByteArray, startNanoOverride: Long? = null) {
        cleanupActiveRemoteTask()
        val startNano = startNanoOverride ?: System.nanoTime()
        requestStartNano = startNano
        val prepMs = monotonicElapsedMs(startNano)
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    screen = AppScreen.RESULT,
                    isSubmitting = true,
                    taskResult = null,
                    networkError = null,
                    clientTimingsMs = linkedMapOf("image_prep_ms" to prepMs),
                    localPreviewBytes = imageBytes
                )
            }
            val requestId = "req_${UUID.randomUUID().toString().replace("-", "").take(12)}"
            val baseUrl = _uiState.value.baseUrl
            runCatching {
                apiClient.submitImageTask(
                    baseUrl = baseUrl,
                    imageBytes = imageBytes,
                    mimeType = "image/jpeg",
                    clientRequestId = requestId
                )
            }.onSuccess { submitResp ->
                val submitAckMs = monotonicElapsedMs(startNano)
                _uiState.update {
                    it.copy(
                        isSubmitting = false,
                        activeTaskId = submitResp.taskId,
                        queuePosition = submitResp.queuePosition,
                        clientTimingsMs = it.clientTimingsMs + ("upload_ack_ms" to submitAckMs)
                    )
                }
                startPollingTask(baseUrl, submitResp.taskId, startNano)
            }.onFailure { err ->
                _uiState.update {
                    it.copy(
                        isSubmitting = false,
                        networkError = err.message
                            ?: "提交分析任务失败，请确认演示电脑 FastAPI 服务已启动或切换离线缓存回放（Plan B）"
                    )
                }
            }
        }
    }

    fun selectCandidateFace(faceId: String) {
        val current = _uiState.value
        val taskId = current.activeTaskId ?: current.taskResult?.taskId ?: return
        val imageId = current.taskResult?.imageId ?: return
        if (current.isSubmitting) return

        pollingJob?.cancel()
        val selectStartNano = System.nanoTime()
        requestStartNano = selectStartNano
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isSubmitting = true,
                    networkError = null,
                    clientTimingsMs = linkedMapOf("select_ack_ms" to 0L)
                )
            }
            runCatching {
                apiClient.selectCandidateFace(current.baseUrl, taskId, imageId, faceId)
            }.onSuccess {
                val ackMs = monotonicElapsedMs(selectStartNano)
                _uiState.update {
                    it.copy(
                        isSubmitting = false,
                        clientTimingsMs = it.clientTimingsMs + ("select_ack_ms" to ackMs)
                    )
                }
                startPollingTask(current.baseUrl, taskId, selectStartNano)
            }.onFailure { err ->
                _uiState.update {
                    it.copy(isSubmitting = false, networkError = err.message ?: "选择目标人脸失败")
                }
            }
        }
    }

    fun loadOfflineDemo(item: OfflineDemoSummary) {
        cleanupActiveRemoteTask()
        runCatching {
            OfflineDemoRepository.loadOfflineDemo(getApplication(), item)
        }.onSuccess { demoResult ->
            _uiState.update {
                it.copy(
                    showOfflineSheet = false,
                    screen = AppScreen.RESULT,
                    isSubmitting = false,
                    activeTaskId = demoResult.taskId,
                    taskResult = demoResult,
                    networkError = null,
                    clientTimingsMs = emptyMap(),
                    localPreviewBytes = null
                )
            }
        }.onFailure { err ->
            _uiState.update {
                it.copy(
                    showOfflineSheet = false,
                    networkError = "加载离线缓存样例失败：${err.message}"
                )
            }
        }
    }

    fun resolveImageUrl(pathOrUrl: String?): String? =
        BackendApiClient.resolveArtifactUrl(_uiState.value.baseUrl, pathOrUrl)

    private fun startPollingTask(baseUrl: String, taskId: String, startNano: Long) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            var rounds = 0
            while (isActive && rounds < MAX_POLL_ROUNDS) {
                rounds++
                val res = runCatching { apiClient.getTaskStatus(baseUrl, taskId) }
                res.onSuccess { taskResult ->
                    val nowMs = monotonicElapsedMs(startNano)
                    _uiState.update { state ->
                        val nextTimings = LinkedHashMap(state.clientTimingsMs)
                        if (taskResult.scores != null && !nextTimings.containsKey("scores_e2e_ms")) {
                            nextTimings["scores_e2e_ms"] = nowMs
                        }
                        if (taskResult.isTerminal) {
                            nextTimings["total_e2e_ms"] = nowMs
                        }
                        state.copy(
                            taskResult = taskResult,
                            networkError = null,
                            clientTimingsMs = nextTimings
                        )
                    }
                    if (taskResult.isTerminal) {
                        return@launch
                    }
                }.onFailure { err ->
                    _uiState.update {
                        it.copy(networkError = err.message ?: "轮询任务状态中断")
                    }
                    return@launch
                }
                delay(POLL_INTERVAL_MS)
            }
            if (rounds >= MAX_POLL_ROUNDS) {
                _uiState.update {
                    it.copy(networkError = "等待后端分析超时，请检查演示电脑推理队列状态")
                }
            }
        }
    }

    private fun cleanupActiveRemoteTask() {
        pollingJob?.cancel()
        pollingJob = null
        val prevTaskId = _uiState.value.activeTaskId
        val prevIsOffline = _uiState.value.taskResult?.isOfflineReplay == true
        val baseUrl = _uiState.value.baseUrl
        if (!prevTaskId.isNullOrBlank() && !prevIsOffline) {
            viewModelScope.launch {
                apiClient.deleteTaskQuietly(baseUrl, prevTaskId)
            }
        }
    }

    override fun onCleared() {
        cleanupActiveRemoteTask()
        super.onCleared()
    }
}
