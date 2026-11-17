param(
    [Parameter(Mandatory = $true)][string]$BindHost,
    [Parameter(Mandatory = $true)][string]$ModeLabel,
    [string]$Python = "",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"

$BackendDir = Split-Path -Parent $PSScriptRoot
$RootDir = Split-Path -Parent $BackendDir
if (-not (Test-Path (Join-Path $BackendDir "app\main.py"))) {
    throw "Backend app not found under: $BackendDir"
}

$LocalEnvFile = Join-Path $RootDir ".local_env.ps1"
if (Test-Path $LocalEnvFile) {
    . $LocalEnvFile
}

function Test-BackendDeps([string]$PyExe) {
    try {
        & $PyExe -c "import uvicorn, fastapi, torch, cv2, mediapipe, transformers" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Resolve-BackendPython([string]$Requested, [string]$ProjectRoot, [string]$BackendPath) {
    if ($Requested) { return $Requested }
    if ($env:PYTHON) { return $env:PYTHON }

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:LOCAL_TORCH_PYTHON) {
        $candidates.Add($env:LOCAL_TORCH_PYTHON)
    }
    $candidates.Add((Join-Path $BackendPath ".venv\Scripts\python.exe"))
    $candidates.Add((Join-Path $ProjectRoot ".venv\Scripts\python.exe"))
    if ($env:VIRTUAL_ENV) {
        $candidates.Add((Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))
    }
    if ($env:CONDA_PREFIX -and $env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne "base") {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }
    if ($env:CONDA_PREFIX) {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "envs\torch_env\python.exe"))
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $candidates.Add($cmd.Source)
    }

    $seen = @{}
    $existing = @()
    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        if ($seen.ContainsKey($candidate)) { continue }
        $seen[$candidate] = $true
        if (Test-Path $candidate) {
            $existing += $candidate
            if (Test-BackendDeps $candidate) {
                return $candidate
            }
        }
    }

    if ($existing.Count -gt 0) {
        throw "Found Python ($($existing[0])), but backend dependencies are missing. Activate your environment or run: pip install -r backend/requirements.txt (or pass -Python <path>)."
    }
    throw "Python interpreter not found. Pass -Python <path> or activate your virtual environment."
}

$PythonExe = Resolve-BackendPython -Requested $Python -ProjectRoot $RootDir -BackendPath $BackendDir
$WeightsDir = Join-Path $BackendDir "weights"
if (-not (Test-Path (Join-Path $WeightsDir "FaceLLM-1B\model"))) {
    Write-Warning "Model weights were not found under $WeightsDir. Run 'python backend/scripts/download_weights.py' and 'python backend/scripts/prepare_age.py' first."
}

Set-Location -LiteralPath $BackendDir
$env:HF_HOME = Join-Path $WeightsDir "hf_cache"
$env:TORCH_HOME = Join-Path $WeightsDir "torch_cache"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONUTF8 = "1"

Write-Host "Starting backend [$ModeLabel] on http://${BindHost}:${Port} using: $PythonExe" -ForegroundColor Cyan
if ($BindHost -eq "127.0.0.1") {
    Write-Host "Tip: run 'adb reverse tcp:${Port} tcp:${Port}' on another terminal for USB debugging." -ForegroundColor DarkGray
} else {
    Write-Host "Tip: enter http://<your-laptop-lan-ip>:${Port} in the Android app server settings." -ForegroundColor DarkGray
}

& $PythonExe -m uvicorn app.main:app --host $BindHost --port $Port --workers 1 --no-access-log
exit $LASTEXITCODE
