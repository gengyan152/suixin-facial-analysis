# Android 客户端 ——「随心一拍」(`android-app`)

基于 **Kotlin + Jetpack Compose + Material 3 + CameraX** 构建的移动端应用，采用**低饱和浅绿 × 暖米水彩手绘风格**，追求简洁直觉的交互体验。

## 核心界面与交互

1. **极简手绘风首页（`HomeScreen.kt`）**：
   - 暖米水彩植物信笺背景（`bg_botanical_paper`）搭配衬线标题 **「随心一拍」**；
   - 右上角仅保留两个轻量圆形图标入口：**示例照片演示** 与 **连接设置**；
   - 画面中央为**低饱和浅绿大圆形拍照按钮**，下方配文 **“一键分析年龄与颜值”**。
2. **纯净取景与相册直达（`CameraPreviewScreen.kt`）**：
   - 全屏纯净取景器配柔和浅绿四角取景框与端侧实时 `MediaPipe` 人脸框检测；
   - 底部三键直觉布局：**切换前后镜头（左下）** · **双环快门按钮（中央）** · **相册选图入口（右下角）**；
   - 内置 `1280×1600` 分辨率约束与 `inSampleSize` 单次矩阵变换，兼容高像素摄像头并控制上传体积在 `8 MB` 契约上限内。
3. **单一手绘信笺框完整报告（`AnalysisResultScreen.kt`）**：
   - 所有分析结果完整收纳在**同一个手绘双线信笺框**内；
   - 呈现 **人脸照片**、**表观年龄**、**颜值评分** 以及 **连贯自然的中文面容与颜值气质描述**（由后端 `compose_narrative` 与端侧 `buildUnifiedDescription` 协同生成）；
   - 非人脸（如小狗、猫、风景）自动搭配手绘插画（`illus_empty_gentle`）温和提示 **“未识别到人脸”**；多人合照支持直接在框内轻点目标人物继续分析。

## 命令行测试与构建

### 方式一：使用封装脚本（自动处理 Windows 非 ASCII 路径问题）

在仓库根目录或 `android-app/` 下执行：

```powershell
# 仅运行单元测试
.\android-app\build_apk.ps1 -Tasks @(":app:testDebugUnitTest")

# 运行单元测试并构建 Debug APK
.\android-app\build_apk.ps1
```

### 方式二：标准 Gradle / Android Studio 构建

若仓库位于纯 ASCII 路径下，也可直接使用 Android Studio 打开 `android-app/` 目录，或运行标准 Gradle Wrapper：

```powershell
cd android-app
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
```

构建产物输出至 `android-app/app/build/outputs/apk/debug/app-debug.apk`。
