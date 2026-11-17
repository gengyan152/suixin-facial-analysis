package com.jlu.facialanalysis

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.jlu.facialanalysis.ui.AppScreen
import com.jlu.facialanalysis.ui.MainViewModel
import com.jlu.facialanalysis.ui.screens.AnalysisResultScreen
import com.jlu.facialanalysis.ui.screens.CameraPreviewScreen
import com.jlu.facialanalysis.ui.screens.HomeScreen
import com.jlu.facialanalysis.ui.screens.OfflineDemoBottomSheet
import com.jlu.facialanalysis.ui.screens.ServerConfigDialog
import com.jlu.facialanalysis.ui.theme.FacialAnalysisTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            FacialAnalysisTheme {
                val vm: MainViewModel = viewModel()
                val uiState by vm.uiState.collectAsStateWithLifecycle()

                val photoPickerLauncher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.PickVisualMedia()
                ) { uri ->
                    if (uri != null) {
                        vm.submitFromGalleryUri(uri)
                    }
                }

                val launchGallery = {
                    photoPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }

                BackHandler(enabled = uiState.screen != AppScreen.HOME) {
                    vm.backToHome()
                }

                when (uiState.screen) {
                    AppScreen.HOME -> {
                        HomeScreen(
                            onOpenCamera = vm::openCamera,
                            onOpenServerDialog = { vm.toggleServerDialog(true) },
                            onOpenOfflineDemos = { vm.toggleOfflineSheet(true) }
                        )
                    }

                    AppScreen.CAMERA -> {
                        CameraPreviewScreen(
                            isSubmitting = uiState.isSubmitting,
                            onBack = vm::backToHome,
                            onPickFromGallery = launchGallery,
                            onCapturedBytes = { bytes -> vm.submitNormalizedImageBytes(bytes) }
                        )
                    }

                    AppScreen.RESULT -> {
                        AnalysisResultScreen(
                            uiState = uiState,
                            resolveUrl = vm::resolveImageUrl,
                            onBack = vm::backToHome,
                            onRetakeCamera = vm::openCamera,
                            onRepickGallery = launchGallery,
                            onOpenOfflineDemos = { vm.toggleOfflineSheet(true) },
                            onSelectCandidateFace = vm::selectCandidateFace
                        )
                    }
                }

                if (uiState.showServerDialog) {
                    ServerConfigDialog(
                        currentBaseUrl = uiState.baseUrl,
                        isChecking = uiState.isCheckingHealth,
                        healthStatus = uiState.healthStatus,
                        healthError = uiState.healthError,
                        onDismiss = { vm.toggleServerDialog(false) },
                        onTestUrl = vm::runHealthCheck,
                        onSaveUrl = vm::updateAndSaveBaseUrl
                    )
                }

                if (uiState.showOfflineSheet) {
                    OfflineDemoBottomSheet(
                        onDismiss = { vm.toggleOfflineSheet(false) },
                        onSelectOfflineDemo = vm::loadOfflineDemo
                    )
                }
            }
        }
    }
}
