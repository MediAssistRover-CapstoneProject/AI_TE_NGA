# =====================================================================
# MediAssistRover - Khởi động AI Vision (Fall, SOS, Seizure)
# =====================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Cấu hình nguồn video (0: Webcam mặc định, hoặc đường dẫn file .mp4)
if (-not $env:VIDEO_SOURCE) {
    $env:VIDEO_SOURCE = "0"
}

# Tối ưu tần suất xử lý các mô hình phụ để giảm tải CPU
if (-not $env:HAND_EVERY) { $env:HAND_EVERY = "2" }
if (-not $env:FACE_EVERY) { $env:FACE_EVERY = "3" }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MEDIASSISTROVER VISION AI SERVICE (v2)" -ForegroundColor Green
Write-Host "  - Fall Detection: Multi-phase v2 (Confidence Score)" -ForegroundColor Yellow
Write-Host "  - Hand SOS: Signal for Help (3-step gesture)" -ForegroundColor Yellow
Write-Host "  - Seizure Motion: Bilateral rhythmic analysis" -ForegroundColor Yellow
Write-Host "  - Video Source: $env:VIDEO_SOURCE" -ForegroundColor Gray
Write-Host "  - Web Dashboard: http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" -m uvicorn full_ai:app --host 127.0.0.1 --port 8001
