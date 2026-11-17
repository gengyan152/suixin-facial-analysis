param([string]$Python = "", [int]$Port = 8000)
& (Join-Path $PSScriptRoot "backend\scripts\launch_backend.ps1") -BindHost "127.0.0.1" -ModeLabel "USB Loopback" -Python $Python -Port $Port
exit $LASTEXITCODE
