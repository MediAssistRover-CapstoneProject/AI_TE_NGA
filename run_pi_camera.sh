#!/usr/bin/env bash
# ==============================================================================
# MEDIASSISTROVER - SCRIPT CHAY CAMERA STREAMER TREN RASPBERRY PI
# Cach chay tren Raspberry Pi:
#   chmod +x run_pi_camera.sh
#   ./run_pi_camera.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PI_CAMERA_SOURCE="${PI_CAMERA_SOURCE:-0}"     # "0" cho USB Webcam (/dev/video0), hoac "csi" cho Pi Camera
export PI_STREAM_PORT="${PI_STREAM_PORT:-8090}"
export PI_FRAME_WIDTH="${PI_FRAME_WIDTH:-640}"
export PI_FRAME_HEIGHT="${PI_FRAME_HEIGHT:-480}"
export PI_TARGET_FPS="${PI_TARGET_FPS:-24}"
export PI_JPEG_QUALITY="${PI_JPEG_QUALITY:-75}"

if ! python3 -c "import cv2" >/dev/null 2>&1; then
    echo "[PI-SETUP] Dang cai dat opencv-python-headless..."
    pip3 install --user opencv-python-headless || sudo apt-get update && sudo apt-get install -y python3-opencv
fi

PI_IP=$(hostname -I | awk '{print $1}')
echo "=========================================================="
echo "  MEDIASSISTROVER - RASPBERRY PI CAMERA STREAMER"
echo "  - Nguon Camera: $PI_CAMERA_SOURCE (${PI_FRAME_WIDTH}x${PI_FRAME_HEIGHT} @ ${PI_TARGET_FPS}fps)"
echo "  - URL Stream cho BE AI: http://${PI_IP}:${PI_STREAM_PORT}/video"
echo "  - Kiem tra Health:      http://${PI_IP}:${PI_STREAM_PORT}/health"
echo "=========================================================="

exec python3 pi_camera_streamer.py
