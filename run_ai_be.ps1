param(
    [string]$CameraSource = "http://192.168.1.100:8090/video"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$env:VIDEO_SOURCE = $CameraSource
$env:POSE_BACKEND = "yolo"
$env:YOLO_CONF = "0.14"
$env:YOLO_IMGSZ = "800"
$env:MAX_FRAME_DIM = "800"
$env:TARGET_STREAM_FPS = "24"
$env:AI_EVERY = "2"
$env:MAX_POSES = "6"
$env:HAND_EVERY = "6"
$env:ENABLE_FACE = "false"
$env:MQTT_ENABLED = "true"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MEDIASSISTROVER - AI SERVER KET NOI CAMERA RASPI & BE_MAR" -ForegroundColor Green
Write-Host "  - Nguon Camera/Video: $CameraSource" -ForegroundColor Yellow
Write-Host "  - Engine AI:          YOLOv8-pose ONNX Runtime (Rect 800px)" -ForegroundColor Green
Write-Host "  - Ket noi BE_MAR:     MQTT TLS -> mar/robots/{robotId}/vision/detections" -ForegroundColor Cyan
Write-Host "  - Web AI Dashboard:   http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" -m uvicorn full_ai:app --host 0.0.0.0 --port 8001
