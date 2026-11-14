# MediAssistRover - Module AI Nhận Diện Té Ngã & Cảnh Báo Khẩn Cấp (AI_TE_NGA)

> Dự án thuộc hệ sinh thái **MediAssistRover (MAR)** - Robot hỗ trợ y tế bệnh viện thông minh.  
> Module này chịu trách nhiệm: **Phát hiện té ngã đa người (Multi-Person Fall Detection)**, **Nhận diện cử chỉ cầu cứu (SOS Signal for Help)**, và **Cảnh báo co giật bất thường (Seizure Detection)** từ camera robot, đồng thời truyền dữ liệu khẩn cấp về Backend qua MQTT TLS.

---

## Phát hiện té ngã khi robot tuần tra

`fall_service.py` chạy YOLOv8n-Pose bằng ONNX Runtime trên CPU. Khi workflow bật
quan sát té ngã, Pi dùng nguồn ảnh ROS hiện có và gửi ảnh JPEG trong lúc robot
đứng yên tại điểm quan sát. Service xử lý ảnh trong RAM và trả kết quả cho Pi.
Pi báo sự kiện cho BE. Nút SOS hoạt động theo luồng SOS của BE.

- `GET /health` kiểm tra service và model đã khởi động.
- `POST /v1/fall/frame` nhận Bearer token, header `X-Fall-Session` là UUID của lượt
  quan sát và body `image/jpeg` tối đa 256 KiB. Ảnh mỗi chiều tối đa 640 px.
- Kết quả gồm `suspected_fall`, `confidence` và `algorithm`. Lượt quan sát đặt
  lại theo UUID mới hoặc khi ảnh bị gián đoạn quá một giây. Kết quả nghi té ngã
  cần được staff xác minh. Người đã nằm trước khi bắt đầu quan sát có thể bị bỏ sót.

Trên EC2 ARM64, chuẩn bị Python 3.12 với `venv` và file
`/etc/mediassistrover/fall-ai.env` thuộc root, mode `0600`. Đặt `FALL_API_TOKEN`
từ nguồn bí mật triển khai. Service mặc định lắng nghe loopback ở cổng `7895`.
Chạy `sudo bash deploy/install-fall.sh <release-sha>` để kiểm tra sẵn sàng và
khôi phục release trước nếu khởi động thất bại. Nếu TLS kết thúc ở proxy trên
EC2 khác, cấu hình `FALL_BIND_HOST` bằng IP riêng và chỉ cấp quyền từ proxy đó.

GitHub Actions chạy kiểm tra trên ARM64 khi push `main` hoặc mở pull request.
Việc áp dụng release lên EC2 hiện dùng bộ cài thủ công.

---

## 1. Kiến Trúc & Luồng Hoạt Động (Architecture)

```text
┌────────────────────────────────────────┐
│  1. RASPBERRY PI (TRÊN ROBOT)          │
│  - Chạy: pi_camera_streamer.py         │
│  - Đọc Camera UVC/CSI (640x480 @ 24fps)│
│  - Siêu nhẹ (< 25MB RAM, < 5% CPU)     │
└──────────────────┬─────────────────────┘
                   │ Luồng HTTP MJPEG Stream (LAN)
                   │ http://<IP_RASPI>:8090/video
                   ▼
┌────────────────────────────────────────┐
│  2. AI VISION SERVICE (SERVER / BE)    │
│  - Chạy: full_ai.py (Cổng 8001)        │
│  - Model: YOLOv8-pose ONNX (Rect 800)  │
│  - Xử lý: FallDetector v2 (Multi-phase)│
│  - Tốc độ: ~22-28 FPS thời gian thực   │
└──────────────────┬─────────────────────┘
                   │ Gửi JSON sự kiện qua MQTT TLS (Port 8883)
                   │ Topic: mar/robots/{robotId}/vision/detections
                   ▼
┌────────────────────────────────────────┐
│  3. BACKEND SPRING BOOT (BE_MAR)       │
│  - Nhận qua: VisionEventHandler.java   │
│  - Bảng DetectionEvent: Lưu JSON gốc   │
│  - Bảng EmergencyIncident: Lưu TEXT    │
│  - Bảng Alert: Phát chuông Y tá / Web  │
└────────────────────────────────────────┘
```

---

## 2. Cấu Trúc Thư Mục (Project Structure)

```text
├── MAR_Fall_AI/
│   ├── full_ai.py                 # Core AI Service (FastAPI + YOLOv8 ONNX + MediaPipe)
│   ├── fall_detector.py           # Thuật toán phân tích động học té ngã (MultiPersonFallTracker)
│   ├── hand_sos.py                # Thuật toán nhận diện cử chỉ tay cầu cứu Quốc tế (SOS)
│   ├── seizure_motion.py          # Thuật toán nhận diện chuyển động co giật bất thường
│   ├── mar_mqtt.py                # MQTT Client kết nối EMQX Cloud (TLS 8883)
│   ├── pi_camera_streamer.py      # Streamer siêu nhẹ chạy trên Raspberry Pi của Robot
│   ├── run_pi_camera.sh           # Script 1-click khởi động camera trên Raspberry Pi
│   ├── run_ai_be.ps1              # Script khởi động AI kết nối vào Camera Robot trên BE
│   ├── test_video.ps1             # Script kiểm thử AI với file video có sẵn
│   ├── yolov8n-pose.onnx          # Model YOLOv8 pose tối ưu hóa ONNX Runtime (CPU Fast)
│   ├── pose_landmarker.task       # MediaPipe Pose Task
│   ├── hand_landmarker.task       # MediaPipe Hand Task
│   ├── full_ai.html               # Giao diện Web Dashboard giám sát AI trực tiếp
│   └── requirements.txt           # Danh sách thư viện Python cần thiết
├── BE_MAR/                        # Mã nguồn Backend Spring Boot của dự án
└── README.md                      # Tài liệu hướng dẫn này
```

---

## 3. Hướng Dẫn Cài Đặt & Vận Hành (Setup & Run)

### PHẦN A: Chạy AI Tại Máy Chủ Backend (Windows / Linux)

#### 1. Yêu cầu môi trường
* Python 3.10 hoặc 3.11
* Cài đặt môi trường ảo (venv):
  ```bash
  cd MAR_Fall_AI
  python -m venv .venv
  
  # Kích hoạt venv (Windows):
  .\.venv\Scripts\activate
  
  # Kích hoạt venv (Linux):
  source .venv/bin/activate
  
  # Cài đặt thư viện:
  pip install -r requirements.txt
  ```

#### 2. Cấu hình MQTT (`MAR_Fall_AI/.env`)
Tạo file `.env` trong thư mục `MAR_Fall_AI`:
```dotenv
MQTT_ENABLED=true
MQTT_HOST=sd1db820.ala.us-east-1.emqxsl.com
MQTT_PORT=8883
MQTT_CLIENT_ID=mar-ai-vision-lenovo
MQTT_USERNAME=mar-vision
MQTT_PASSWORD=1234567
ROBOT_ID=0e3f1083-2546-4f45-8004-5eda5cb75a6f
MQTT_TOPIC=mar/robots/0e3f1083-2546-4f45-8004-5eda5cb75a6f/vision/detections
MQTT_STATUS_TOPIC=mar/robots/0e3f1083-2546-4f45-8004-5eda5cb75a6f/vision/status
```

#### 3. Khởi chạy AI
* **Cách 1: Kiểm thử với File Video có sẵn trên máy**:
  ```powershell
  cd MAR_Fall_AI
  .\test_video.ps1 -VideoPath "D:\VideoTHOOO\20261018_C3356.MP4"
  ```
* **Cách 2: Chạy với Webcam cắm trên máy chủ**:
  ```powershell
  cd MAR_Fall_AI
  .\run_ai_be.ps1 -CameraSource "0"
  ```
* **Cách 3: Chạy kết nối Camera trực tiếp từ Robot Raspberry Pi**:
  ```powershell
  cd MAR_Fall_AI
  .\run_ai_be.ps1 -CameraSource "http://<IP_RASPBERRY_PI>:8090/video"
  ```
* **Mở Web Dashboard xem trực tiếp**: [http://127.0.0.1:8001](http://127.0.0.1:8001)

---

### PHẦN B: Cài Đặt Camera Streamer Trên Raspberry Pi (Robot)

Trên Raspberry Pi của Robot, bạn **không cần cài các thư viện AI nặng**, chỉ cần copy 2 file:
* `MAR_Fall_AI/pi_camera_streamer.py`
* `MAR_Fall_AI/run_pi_camera.sh`

#### 1. Copy sang Raspberry Pi:
```bash
scp MAR_Fall_AI/pi_camera_streamer.py MAR_Fall_AI/run_pi_camera.sh pi@<IP_RASPBERRY_PI>:/home/pi/
```

#### 2. Khởi chạy trên Raspberry Pi:
```bash
ssh pi@<IP_RASPBERRY_PI>
cd /home/pi
chmod +x run_pi_camera.sh

# Nếu dùng Webcam USB:
./run_pi_camera.sh

# Nếu dùng Camera cáp bẹ CSI (Pi Cam v2/v3):
PI_CAMERA_SOURCE=csi ./run_pi_camera.sh
```
> Luồng video sẽ phát tại: `http://<IP_RASPBERRY_PI>:8090/video`.

---

## 4. Cơ Chế Lưu Dữ Liệu Trong Database Của Backend (`BE_MAR`)

Khi AI phát hiện sự kiện khẩn cấp, gói tin MQTT được gửi về Backend và lưu vào cơ sở dữ liệu MySQL (`backend_robot_v1`):

1. **Bảng `DetectionEvent` (Lưu JSON gốc)**:
   * Cột `event_payload_json` (`JSON`): Lưu toàn bộ metrics thô do AI gửi về (`confidence`, `angle_deg`, `hip_speed`, `person_id`,...).
2. **Bảng `EmergencyIncident` (Chuyển sang TEXT sự cố)**:
   * Cột `description` (`TEXT`): Tự động biên dịch thông số JSON thành chuỗi tiếng Việt chi tiết:
     > *"CẢNH BÁO TÉ NGÃ: Camera AI của robot MediAssist Rover 01 tại khu vực 'Tầng 1' phát hiện người có dấu hiệu té ngã [Đối tượng: Người #1, Độ tin cậy: 88.0%, Góc nghiêng thân: 68.5°, Tốc độ rơi: 0.34/s] – cần nhân viên y tế xác minh khẩn cấp."*
3. **Bảng `Alert` (Phát thông báo Y tá)**:
   * Cột `message` (`TEXT`): Chuỗi cảnh báo đỏ gửi về App Y tá / Web Admin kèm còi báo động và yêu cầu nhân viên bấm Xác nhận (ACK).
