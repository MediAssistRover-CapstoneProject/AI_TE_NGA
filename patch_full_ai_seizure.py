"""Idempotent, fail-fast patch for MediAssistRover full_ai.py/full_ai.html."""
from pathlib import Path
import ast
import shutil

root = Path.cwd()
backend = root / 'full_ai.py'
frontend = root / 'full_ai.html'
module = root / 'seizure_motion.py'
if not backend.is_file() or not frontend.is_file() or not module.is_file():
    raise SystemExit('ERROR: Place seizure_motion.py and patch script beside full_ai.py/full_ai.html')

src = backend.read_text(encoding='utf-8-sig')
html = frontend.read_text(encoding='utf-8-sig')


def replace_exact(data, old, new, label):
    if new in data:
        return data
    if data.count(old) != 1:
        raise RuntimeError(f'PATCH FAIL {label}: Expected one marker, found {data.count(old)}')
    return data.replace(old, new, 1)

src = replace_exact(
    src,
    'from app import FallDetector, draw_pose',
    'from app import FallDetector, draw_pose\nfrom seizure_motion import SeizureMotionDetector',
    'import'
)
src = replace_exact(
    src,
    '    detector = FallDetector()',
    '    detector = FallDetector()\n    seizure_detector = SeizureMotionDetector()',
    'initialization'
)
src = replace_exact(
    src,
    '                # 21 điểm mỗi bàn tay',
    '''                # Synchronized with fresh Pose landmarks (one observation per AI frame).
                seizure_status, seizure_event = seizure_detector.update(
                    pose_result.pose_landmarks[0]
                    if pose_result.pose_landmarks else None,
                    now
                )
                if seizure_event is not None:
                    print('[VISION] SUSPECTED CONVULSIVE MOTION (DEMO):',
                          seizure_event['event_id'])
                    # Reuses the existing MQTT adapter if installed.
                    # NO automatic Backend workflow activation here.
                    if 'mqtt_publisher' in locals():
                        mqtt_publisher.emit_event(seizure_event)

                # 21 điểm mỗi bàn tay''',
    'processing'
)
src = replace_exact(
    src,
    '                        "last_event": last_event',
    '                        "seizure": seizure_status,\n'
    '                        "last_seizure_event": seizure_detector.last_event,\n'
    '                        "last_event": last_event',
    'state fields'
)

card = '''
    <div class="card" id="seizureCard">
      <h2>Phát hiện chuyển động nghi co giật — DEMO</h2>
      <p>Trạng thái: <strong id="seizurePhase">Đang khởi động...</strong></p>
      <p id="seizureExplanation">Phân tích dao động hai cánh tay trong cửa sổ thời gian 6 giây.</p>
      <p>FPS: <strong id="seizureFPS">—</strong></p>
      <p id="seizureLastEvent">Chưa ghi nhận sự kiện.</p>
      <details><summary>Chỉ số chuyển động (không phải độ tin cậy y tế)</summary>
        <pre id="seizureMetrics"></pre>
      </details>
      <small>Chưa được kiểm định lâm sàng; chỉ đánh dấu nghi ngờ để nhân viên xác minh. Không dùng làm hệ thống cấp cứu độc lập.</small>
    </div>
'''
if 'id="seizureCard"' not in html:
    html = replace_exact(html, '</aside>', card + '\n</aside>', 'HTML alert card')

js = '''
<script>
// UI extension; works with existing onData() whether the original AI
// dashboard is connected by HTTP polling or WebSocket.
const previousOnDataForSeizure = onData;
onData = function(data) {
  previousOnDataForSeizure(data);
  const s = data.seizure || {};
  const phase = s.phase || 'NO_DATA';
  const phases = {
    NO_PERSON: 'Không thấy người',
    LOW_VISIBILITY: 'Không thấy đủ vai hoặc cổ tay',
    LOW_FPS: 'FPS quá thấp để phân tích chuyển động',
    OBSERVING: 'Đang phân tích chuyển động',
    RHYTHMIC_MOTION_CHECKING: 'Phát hiện chuyển động nhịp điệu, đang xác minh',
    SUSPECTED_SEIZURE: 'NGHI CÓ CHUYỂN ĐỘNG CO GIẬT — CẦN KIỂM TRA',
    COOLDOWN: 'Đã ghi nhận cảnh báo, đang chờ chống lặp'
  };
  document.getElementById('seizurePhase').textContent = phases[phase] || phase;
  document.getElementById('seizurePhase').style.color =
    phase === 'SUSPECTED_SEIZURE' ? '#fb7185' :
    phase === 'RHYTHMIC_MOTION_CHECKING' ? '#fde047' : '#4ade80';
  document.getElementById('seizureFPS').textContent = String(data.fps ?? 0);
  if (document.getElementById('seizureMetrics').parentElement.open) {
    document.getElementById('seizureMetrics').textContent =
      JSON.stringify(s.metrics || {}, null, 2);
  }
  const event = data.last_seizure_event;
  if (event) {
    document.getElementById('seizureLastEvent').textContent =
      'Sự kiện thử nghiệm gần nhất: ' + event.event_type + ' · ' + event.detected_at;
  }
};
</script>
'''
if 'previousOnDataForSeizure' not in html:
    html = replace_exact(html, '</body>', js + '\n</body>', 'HTML visualization')

# Add light-weight seizure state to the existing MQTT status topic if the
# previously installed MQTT publisher is present. Never send landmarks/frames.
mqtt_module = root / 'mar_mqtt.py'
mqtt_old = mqtt_module.read_text(encoding='utf-8-sig') if mqtt_module.is_file() else None
mqtt_new = mqtt_old
if mqtt_old is not None and '"seizure_phase":' not in mqtt_old:
    mqtt_new = replace_exact(
        mqtt_old,
        '            "fall_state": snapshot.get("fall_state"),',
        '            "fall_state": snapshot.get("fall_state"),\n'
        '            "seizure_phase": (snapshot.get("seizure") or {}).get("phase"),',
        'MQTT status')

ast.parse(src)
ast.parse(module.read_text(encoding='utf-8'))
if mqtt_new is not None:
    ast.parse(mqtt_new)
for path, old, new in ((backend, backend.read_text(encoding='utf-8-sig'), src),
                       (frontend, frontend.read_text(encoding='utf-8-sig'), html)):
    if old == new:
        print('ALREADY PATCHED:', path.name)
        continue
    backup = path.with_suffix(path.suffix + '.before_seizure.bak')
    shutil.copy2(path, backup)
    path.write_text(new, encoding='utf-8')
    print('PATCHED:', path.name, '(backup:', backup.name + ')')
if mqtt_old is not None and mqtt_old != mqtt_new:
    backup = mqtt_module.with_suffix(mqtt_module.suffix + '.before_seizure.bak')
    shutil.copy2(mqtt_module, backup)
    mqtt_module.write_text(mqtt_new, encoding='utf-8')
    print('PATCHED:', mqtt_module.name, '(backup:', backup.name + ')')
print('SEIZURE MOTION PATCH SUCCESS')
