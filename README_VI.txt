MEDIASSISTROVER - CANH BAO CHUYEN DONG NGHI CO GIAT (POC)
==========================================================
Khong phai phan mem chan doan co giat/dong kinh, khong duoc su dung
lam he thong an toan benh nhan doc lap. Can canh bao nhan vien xac minh.

A) CAI DAT
1. Giai nen 3 file seizure_motion.py, patch_full_ai_seizure.py,
   test_seizure_motion.py vao D:\DoAn\MAR_Fall_AI
2. Mo PowerShell:
   cd D:\DoAn\MAR_Fall_AI
   .\.venv\Scripts\python.exe -m pip install numpy
   .\.venv\Scripts\python.exe .\test_seizure_motion.py
   .\.venv\Scripts\python.exe .\patch_full_ai_seizure.py
   .\.venv\Scripts\python.exe -m py_compile .\full_ai.py .\seizure_motion.py

B) CHAY AI
   $env:VIDEO_SOURCE="0"
   $env:HAND_EVERY="2"
   $env:FACE_EVERY="3"
   .\.venv\Scripts\python.exe -m uvicorn full_ai:app --host 127.0.0.1 --port 8001
   http://127.0.0.1:8001

C) CO CHE
- Phan tich 33 pose landmarks tren nhieu frame cua MediaPipe.
- Can thay hai vai va hai co tay; fps >= 8.
- Tim dao dong co nhip tren CA HAI canh tay trong khoang 6s.
- Xac minh nhieu cua so lien tiep 1.5s, cooldown 60s.
- Ghi su kien SUSPECTED_SEIZURE; confidence=null, simulation=true.
- Khong gui anh, khuon mat hay toa do qua MQTT.
- Khi co mar_mqtt.py va mqtt_publisher da cai theo ban truoc,
  su kien moi duoc gui qua MQTT_TOPIC hien co neu MQTT_ENABLED=true.
- Day la detector mo ta CHUYEN DONG CO Nhip, khong phat hien
  duoc tat ca cac loai con co giat. Vay tay/nha y/dung may rung
  co the tao canh bao nham; co the bo sot neu camera che khuat.
- Neu robot di chuyen, can dung yen/bu chuyen dong camera trong
  giai doan phan tich; thuat toan chua duoc validate tren robot di chuyen.

D) KIEM TRA
   Invoke-RestMethod http://127.0.0.1:8001/api/state |
       Select-Object -ExpandProperty seizure | ConvertTo-Json -Depth 6
   (Invoke-RestMethod http://127.0.0.1:8001/api/state).last_seizure_event |
       ConvertTo-Json -Depth 6

   MQTT enabled + connected: bat su kien qua EMQX QoS1,
   can su dung subscriber account rieng de xem topic vision/detections.
   BE_MAR CHUA mac dinh subscribe/translate event nay, khong tu tao workflow.

E) PHUC HOI
   Copy-Item .\full_ai.py.before_seizure.bak .\full_ai.py -Force
   Copy-Item .\full_ai.html.before_seizure.bak .\full_ai.html -Force

An toan: Neu nghi nguoi that dang co giat, uu tien nhan vien y te,
khong cho vat vao mieng va khong giu chat tay chan ho.
