param(
    [string[]]$Tasks = @(":app:testDebugUnitTest", ":app:assembleDebug")
)
$ErrorActionPreference = "Stop"

# Project root contains non-ASCII characters ("大三\安卓开发"), which causes Windows aapt2.exe
# and Java @argfile launcher to mangle paths under GBK codepages.
# Create/reuse a zero-copy NTFS junction under $HOME\.gradle so all outputs still land directly
# in android-app\app\build\ while toolchains see pure ASCII paths.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GradleHome = Join-Path $HOME ".gradle"
if (-not (Test-Path $GradleHome)) {
    New-Item -ItemType Directory -Path $GradleHome | Out-Null
}
$AsciiRoot = Join-Path $GradleHome "jlu_facial_recognition_ws"
if (Test-Path $AsciiRoot) {
    $item = Get-Item $AsciiRoot -Force
    if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        Remove-Item -Recurse -Force $AsciiRoot
    } elseif ($item.Target -and ($item.Target -notcontains $ProjectRoot)) {
        cmd /c rmdir "$AsciiRoot" | Out-Null
    }
}
if (-not (Test-Path $AsciiRoot)) {
    cmd /c mklink /J "$AsciiRoot" "$ProjectRoot" | Out-Null
}

$AsciiAppDir = Join-Path $AsciiRoot "android-app"
$ToolchainRoot = Join-Path $AsciiRoot ".toolchain"
if (Test-Path $ToolchainRoot) {
    $JdkDir = Get-ChildItem (Join-Path $ToolchainRoot "jdk17") -Directory | Select-Object -First 1
    $SdkDir = Join-Path $ToolchainRoot "sdk"
    $GradleBin = Join-Path $ToolchainRoot "gradle\gradle-8.7\bin\gradle.bat"
    if ($JdkDir) { $env:JAVA_HOME = $JdkDir.FullName }
    if (Test-Path $SdkDir) {
        $env:ANDROID_HOME = $SdkDir
        $env:ANDROID_SDK_ROOT = $SdkDir
        $EscapedSdk = $SdkDir.Replace("\", "\\")
        Set-Content -Path (Join-Path $AsciiAppDir "local.properties") -Value "sdk.dir=$EscapedSdk" -Encoding UTF8
    }
} else {
    $GradleBin = Join-Path $AsciiAppDir "gradlew.bat"
    $SdkDir = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } elseif ($env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT } else { Join-Path $env:LOCALAPPDATA "Android\Sdk" }
    if ((Test-Path $SdkDir) -and -not (Test-Path (Join-Path $AsciiAppDir "local.properties"))) {
        $EscapedSdk = $SdkDir.Replace("\", "\\")
        Set-Content -Path (Join-Path $AsciiAppDir "local.properties") -Value "sdk.dir=$EscapedSdk" -Encoding UTF8
    }
}

$env:GRADLE_USER_HOME = $GradleHome
if ($env:HTTPS_PROXY -match "127\.0\.0\.1:(\d+)") {
    $port = $Matches[1]
    $env:GRADLE_OPTS = "-Dhttp.proxyHost=127.0.0.1 -Dhttp.proxyPort=$port -Dhttps.proxyHost=127.0.0.1 -Dhttps.proxyPort=$port"
}

Push-Location $AsciiAppDir
try {
    & $GradleBin @Tasks "--no-daemon" "--stacktrace"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
