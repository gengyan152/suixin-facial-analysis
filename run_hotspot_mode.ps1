param([string]$Python = "", [int]$Port = 8000)
& (Join-Path $PSScriptRoot "backend\scripts\launch_backend.ps1") -BindHost "0.0.0.0" -ModeLabel "Hotspot/LAN" -Python $Python -Port $Port
exit $LASTEXITCODE
