# HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH: AI FALL DETECTION TẠI BACKEND (`BE_MAR`) & KẾT NỐI CAMERA RASPBERRY PI

---

## 1. Sơ Đồ Tổng Quan Luồng Hoạt Động

```text
┌────────────────────────────────────────┐
│  1. RASPBERRY PI (TRÊN ROBOT)          │
│  - Chạy: pi_camera_streamer.py         │
│  - Chỉ đọc Camera (640x480 @ 24fps)    │
│  - Chiếm < 25MB RAM, < 5% CPU Pi       │
└──────────────────┬─────────────────────┘
                   │ Luồng Video HTTP MJPEG (LAN)
                   │ http://<IP_RASPI>:8090/video
                   ▼
┌────────────────────────────────────────┐
│  2. AI VISION SERVICE (TẠI MÁY BE)     │
│  - Thư mục: D:\MAR_Fall_AI\MAR_Fall_AI │
│  - Chạy: full_ai.py (Cổng 8001)        │
│  - Engine: YOLOv8-pose ONNX (Rect 800) │
│  - Phát hiện: Té ngã, SOS, Co giật     │
└──────────────────┬─────────────────────┘
                   │ Gửi JSON sự kiện qua MQTT TLS (Port 8883)
                   │ Topic: mar/robots/{robotId}/vision/detections
                   ▼
┌────────────────────────────────────────┐
│  3. BACKEND SPRING BOOT (BE_MAR)       │
│  - Thư mục: D:\BE_MAR\BE_MAR           │
│  - Nhận qua: VisionEventHandler.java   │
│  - Lưu JSON -> Bảng DetectionEvent     │
│  - Chuyển sang TEXT -> Bảng            │
│    EmergencyIncident & Bảng Alert      │
└────────────────────────────────────────┘
```

---

## 2. PHẦN A: Cài Đặt & Kết Nối AI Với Backend (`BE_MAR`)

### Bước A1: Cấu hình kết nối MQTT trong `BE_MAR`
Mở file cấu hình môi trường `D:\BE_MAR\BE_MAR\.env` (nếu chưa có thì copy từ `.env.example` sang `.env`) và đảm bảo các biến MQTT khớp với cấu hình của AI:

```dotenv
# Bật MQTT trên Backend Spring Boot
MQTT_ENABLED=true
MQTT_BROKER_URI=ssl://sd1db820.ala.us-east-1.emqxsl.com:8883
MQTT_CLIENT_ID=mar-be-mqtt
MQTT_USERNAME=mar-vision
MQTT_PASSWORD=1234567

# Danh sách UUID Robot được phép gửi sự kiện (để trống = cho phép mọi robot đã đăng ký trong DB)
MQTT_ROBOT_IDS=0e3f1083-2546-4f45-8004-5eda5cb75a6f
MQTT_ALLOW_SIMULATION=true
```

> **Lưu ý về Database (`Robot` & `Map`)**:  
> Trong bảng `Robot` của MySQL (`backend_robot_v1`), cần có bản ghi Robot với `robot_id` tương ứng (`0e3f1083-2546-4f45-8004-5eda5cb75a6f`). Khi sự kiện té ngã gửi về:
> - **Bảng `DetectionEvent`** (cột `event_payload_json` kiểu `JSON`): Lưu nguyên bản gói JSON từ AI (kèm trường `event_text_vi`).
> - **Bảng `EmergencyIncident`** (cột `description` kiểu `TEXT`): Tự động lưu đoạn văn bản tiếng Việt đã chuyển đổi chi tiết (Tên robot, khu vực bản đồ, người số mấy, độ tin cậy %, góc nghiêng thân, tốc độ rơi).
> - **Bảng `Alert`** (cột `message` kiểu `TEXT` và `data_json` kiểu `JSON`): Lưu cảnh báo khẩn cấp để hiển thị lên Web/App Y tá.

### Bước A2: Khởi động Backend (`BE_MAR`)
Mở PowerShell tại `D:\BE_MAR\BE_MAR`:
```powershell
cd D:\BE_MAR\BE_MAR
.\run.ps1
```
*(Hoặc chạy trực tiếp bằng Maven nếu MySQL đã bật sẵn trên máy: `mvn spring-boot:run`)*.
* Kiểm tra Backend đã sống tại: `http://localhost:8080/actuator/health`
* Xem tài liệu API tại: `http://localhost:8080/swagger-ui/index.html`

---

### Bước A3: Cấu hình & Chạy AI (`MAR_Fall_AI`) tại máy chủ BE
1. Kiểm tra file `D:\MAR_Fall_AI\MAR_Fall_AI\.env`:
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
2. **Cách chạy AI để test với File Video trên máy BE**:
   ```powershell
   cd D:\MAR_Fall_AI\MAR_Fall_AI
   .\test_video.ps1 -VideoPath "D:\VideoTHOOO\20261018_C3356.MP4"
   ```
3. **Cách chạy AI để test với Webcam cắm trực tiếp vào máy BE**:
   ```powershell
   cd D:\MAR_Fall_AI\MAR_Fall_AI
   .\run_ai_be.ps1 -CameraSource "0"
   ```
4. Mở trình duyệt truy cập Dashboard AI tại: **`http://127.0.0.1:8001`**

---

## 3. PHẦN B: Hướng Dẫn Cài Đặt & Kết Nối Camera Thật Trên Raspberry Pi (Robot)

Trên Raspberry Pi của Robot, bạn **KHÔNG cần copy cả thư mục AI nặng**, chỉ cần copy đúng **2 file siêu nhẹ** sau từ `D:\MAR_Fall_AI\MAR_Fall_AI` sang Raspberry Pi:
1. `pi_camera_streamer.py`
2. `run_pi_camera.sh`

---

### Bước B1: Copy 2 file sang Raspberry Pi
Đảm bảo máy tính BE và Raspberry Pi nối chung một mạng Wi-Fi / LAN. Mở PowerShell trên máy tính Windows và chạy lệnh `scp` (thay `192.168.1.100` bằng địa chỉ IP thật của Raspberry Pi):

```powershell
cd D:\MAR_Fall_AI\MAR_Fall_AI
scp pi_camera_streamer.py run_pi_camera.sh pi@192.168.1.100:/home/pi/
```

---

### Bước B2: Cài đặt và khởi chạy Camera Streamer trên Raspberry Pi
SSH vào Raspberry Pi:
```bash
ssh pi@192.168.1.100
```

Trên terminal của Raspberry Pi, chạy các lệnh sau:
```bash
cd /home/pi
chmod +x run_pi_camera.sh

# Trường hợp 1: Nếu Robot dùng Camera cắm cổng USB (Webcam UVC /dev/video0)
./run_pi_camera.sh

# Trường hợp 2: Nếu Robot dùng Camera cáp bẹ CSI (Raspberry Pi Camera Module v2/v3)
PI_CAMERA_SOURCE=csi ./run_pi_camera.sh
```

Khi chạy thành công, màn hình Raspberry Pi sẽ in ra:
```text
==========================================================
  MEDIASSISTROVER - RASPBERRY PI CAMERA STREAMER
  - Nguon Camera: 0 (640x480 @ 24fps)
  - URL Stream cho BE AI: http://192.168.1.100:8090/video
  - Kiem tra Health:      http://192.168.1.100:8090/health
==========================================================
```

---

### Bước B3: (Tùy chọn) Cài đặt tự động chạy nền mỗi khi bật nguồn Robot
Nếu muốn mỗi khi bật công tắc nguồn Robot, Raspberry Pi tự động phát camera mà không cần gõ lệnh tay, chạy lệnh sau trên Raspberry Pi:

```bash
sudo tee /etc/systemd/system/mar-pi-camera.service > /dev/null << 'EOF'
[Unit]
Description=MediAssistRover Raspberry Pi Camera Streamer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi
Environment="PI_CAMERA_SOURCE=0"
Environment="PI_STREAM_PORT=8090"
Environment="PI_FRAME_WIDTH=640"
Environment="PI_FRAME_HEIGHT=480"
Environment="PI_TARGET_FPS=24"
ExecStart=/usr/bin/python3 /home/pi/pi_camera_streamer.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mar-pi-camera.service
sudo systemctl start mar-pi-camera.service
```

---

### Bước B4: Kết nối AI trên Máy BE vào Camera Thật của Raspberry Pi
Sau khi Raspberry Pi đã phát luồng tại `http://<IP_RASPI>:8090/video` (ví dụ `http://192.168.1.100:8090/video`):

1. Trên máy chủ BE (Windows), mở PowerShell tại `D:\MAR_Fall_AI\MAR_Fall_AI`:
   ```powershell
   cd D:\MAR_Fall_AI\MAR_Fall_AI
   .\run_ai_be.ps1 -CameraSource "http://192.168.1.100:8090/video"
   ```
2. Mở trình duyệt vào **`http://127.0.0.1:8001`**:
   * Bạn sẽ thấy hình ảnh trực tiếp từ Camera trên Robot truyền về với tốc độ **22–28 FPS thời gian thực**, kèm khung xương nhận diện đa người của YOLOv8-pose ONNX.
3. Khi có người ngã trước Camera Robot:
   * AI chuyển trạng thái sang **`SUSPECTED_FALL`** (khung xương đổi màu Đỏ).
   * AI lập tức bắn gói tin MQTT QoS 1 lên topic `mar/robots/{robotId}/vision/detections`.
   * Backend `BE_MAR` nhận gói tin qua `VisionEventHandler`, lưu JSON vào bảng `DetectionEvent` và lưu bản dịch TEXT tiếng Việt vào bảng `EmergencyIncident` + `Alert`.
