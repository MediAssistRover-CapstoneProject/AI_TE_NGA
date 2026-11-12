$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Chua co .venv. Hay tao virtual environment va cai requirements truoc."
}

Write-Host "=========================================================="
Write-Host " MediAssistRover - Laptop Media Simulator"
Write-Host " Giu cua so nay mo de BE co the gui cuoc goi SOS toi staff"
Write-Host "=========================================================="
& $python (Join-Path $PSScriptRoot "media_simulator.py")
