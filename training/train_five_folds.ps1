param(
    [Parameter(Mandatory=$true)][string]$Images,
    [Parameter(Mandatory=$true)][string]$Splits,
    [string]$Python = '',
    [int]$BatchSize = 32,
    [ValidateSet('fp32','fp16','bf16')][string]$Precision = 'fp32',
    [int]$Workers = 4,
    [double]$DevFraction = 0,
    [string]$Output = "$PSScriptRoot\artifacts\2dfap"
)
$ErrorActionPreference = 'Stop'
$LocalEnvFile = Join-Path (Split-Path -Parent $PSScriptRoot) '.local_env.ps1'
if (Test-Path $LocalEnvFile) {
    . $LocalEnvFile
}
if (-not $Python) {
    if ($env:PYTHON) {
        $Python = $env:PYTHON
    } elseif ($env:LOCAL_TORCH_PYTHON -and (Test-Path $env:LOCAL_TORCH_PYTHON)) {
        $Python = $env:LOCAL_TORCH_PYTHON
    } elseif (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
        $Python = "$PSScriptRoot\.venv\Scripts\python.exe"
    } else {
        $Python = 'python'
    }
}
foreach ($fold in 1..5) {
    $foldRoot = Join-Path $Splits "cross_validation_$fold"
    & $Python "$PSScriptRoot\train_2dfap.py" --images $Images `
        --train-list "$foldRoot\train_${fold}_aligned_ext.txt" `
        --test-list "$foldRoot\test_${fold}_aligned_ext.txt" `
        --fold $fold --batch-size $BatchSize --precision $Precision --workers $Workers `
        --dev-fraction $DevFraction --output "$Output\fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "Fold $fold failed" }
}
& $Python "$PSScriptRoot\summarize_folds.py" "$Output\fold1" "$Output\fold2" "$Output\fold3" "$Output\fold4" "$Output\fold5" --output "$Output\five_fold_summary.json"
if ($LASTEXITCODE -ne 0) { throw 'Five-fold aggregation failed' }
