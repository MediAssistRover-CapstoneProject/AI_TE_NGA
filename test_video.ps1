param(
    [string]$VideoPath = "D:\VideoTHOOO\20261018_C3356.MP4"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$IsStreamOrWebcam = ($VideoPath -match "^(http://|https://|rtsp://|rtmp://|/dev/video|\d+$)")
if (-not $IsStreamOrWebcam -and -not (Test-Path $VideoPath)) {
    Write-Host "Khong tim thay file video tai: $VideoPath" -ForegroundColor Red
    exit 1
}

$env:VIDEO_SOURCE = $VideoPath
$env:POSE_BACKEND = "yolo"
$env:YOLO_CONF = "0.14"
$env:YOLO_IMGSZ = "800"
$env:MAX_FRAME_DIM = "800"
$env:TARGET_STREAM_FPS = "28"
$env:MAX_POSES = "6"
$env:HAND_EVERY = "4"
$env:ENABLE_FACE = "false"
$env:MQTT_ENABLED = "true"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MEDIASSISTROVER - HIGH-SPEED REAL-TIME AI VISION SERVER" -ForegroundColor Green
Write-Host "  - Nguon Video/Raspi Stream: $VideoPath" -ForegroundColor Yellow
Write-Host "  - Engine Pose: YOLOv8-pose ONNX Runtime (800px, Multi-person)" -ForegroundColor Green
Write-Host "  - Toc do Video Stream: Real-time 1.0x Wall-Clock Sync" -ForegroundColor Green
Write-Host "  - Toc do AI Inference: ~13-15 FPS (ONNX Accelerated)" -ForegroundColor Green
Write-Host "  - Bo phat hien te nga: Multi-Person FallDetector v2" -ForegroundColor Cyan
Write-Host "  - Web Dashboard: http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" -m uvicorn full_ai:app --host 0.0.0.0 --port 8001
