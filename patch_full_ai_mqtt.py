"""One-time patch. Backup, validate markers, then replace WS with MQTT + HTTP poll."""
from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
project = Path.cwd()
pyfile = project / "full_ai.py"
htmlfile = project / "full_ai.html"

if not pyfile.is_file() or not htmlfile.is_file():
    raise SystemExit("ERROR: Run in D:\\DoAn\\MAR_Fall_AI (full_ai.py & full_ai.html required)")
if not (root / "mar_mqtt.py").is_file():
    raise SystemExit("ERROR: Place mar_mqtt.py beside this patch script before running")

src = pyfile.read_text(encoding="utf-8-sig")
html = htmlfile.read_text(encoding="utf-8-sig")

if "from mar_mqtt import VisionMqttPublisher" not in src:
    marker = "from app import FallDetector, draw_pose"
    if marker not in src:
        raise SystemExit("ERROR: Expected FallDetector import not found. No files changed.")
    src = src.replace(marker, marker + "\nfrom mar_mqtt import VisionMqttPublisher", 1)

if "mqtt_publisher = VisionMqttPublisher()" not in src:
    marker = "    detector = FallDetector()"
    if marker not in src:
        raise SystemExit("ERROR: Expected FallDetector() initialization not found. No files changed.")
    src = src.replace(marker, marker + "\n    mqtt_publisher = VisionMqttPublisher()", 1)

if "mqtt_publisher.emit_event(last_event)" not in src:
    pattern = re.compile(r"^([ \t]+)with lock:\n[ \t]+if encoded:", re.M)
    match = pattern.search(src)
    if not match:
        raise SystemExit("ERROR: Expected JPEG/state update marker not found. No files changed.")
    indent = match.group(1)
    insert = (
        f"{indent}# Send only deduplicated, minimal vision events through MQTT.\n"
        f"{indent}mqtt_publisher.emit_event(last_event)\n"
        f"{indent}if 'sos_detector' in locals():\n"
        f"{indent}    mqtt_publisher.emit_event(sos_detector.last_event)\n\n"
    )
    src = src[:match.start()] + insert + src[match.start():]

if '"mqtt": mqtt_publisher.status()' not in src:
    pattern = re.compile(r'^(\s*)"last_event": last_event(,?)\s*$', re.M)
    # Last event must exist in state = {...} block.
    matches = list(pattern.finditer(src))
    if len(matches) != 1:
        raise SystemExit(f"ERROR: Expected one last_event field, got {len(matches)}. No files changed.")
    match = matches[0]
    indent = match.group(1)
    old = match.group(0)
    new = f'{indent}"last_event": last_event,\n{indent}"mqtt": mqtt_publisher.status()'
    src = src.replace(old, new, 1)

if "mqtt_publisher.emit_status(state)" not in src:
    # Put status publishing outside state lock but before next iteration.
    needle = '\n    except Exception as exc:\n        print("FULL AI ERROR:"'
    replacement = '\n                mqtt_publisher.emit_status(state)\n' + needle
    if needle not in src:
        raise SystemExit("ERROR: Expected FULL AI ERROR handler not found. No files changed.")
    src = src.replace(needle, replacement, 1)

if "mqtt_publisher.close()" not in src:
    marker = "    finally:\n        if cap is not None:\n            cap.release()"
    if marker not in src:
        raise SystemExit("ERROR: Expected camera release block not found. No files changed.")
    src = src.replace(marker, marker + "\n        mqtt_publisher.close()", 1)

# Remove only the original WebSocket endpoint and its import.
ws_start = re.search(r'^@app\.websocket\(["\']\/ws["\']\)\s*$', src, re.M)
if ws_start:
    tail = src[ws_start.start():]
    next_route = re.search(r'^@app\.(?:get|post|put|delete|websocket)\(', tail[3:], re.M)
    if next_route:
        # Explicitly refuse to blindly delete later endpoints.
        raise SystemExit("ERROR: Other endpoint after /ws. Manual review required. No files changed.")
    src = src[:ws_start.start()].rstrip() + "\n"

src = src.replace("from fastapi import FastAPI, WebSocket", "from fastapi import FastAPI")

# Browser telemetry via HTTP; no WebSocket usage or MQTT credentials in browser.
# Handles the original connect() at the end of the main JS block.
html_pattern = re.compile(r"function connect\(\)\s*\{.*?\n\}\s*\n\s*connect\(\);", re.S)
replacement = '''// HTTP polling replaces the previous browser WebSocket telemetry.
let pollBusy = false;
async function pollState() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    onData(await response.json());
  } catch (error) {
    document.getElementById("status").textContent =
      "Mất kết nối API: " + error.message;
  } finally {
    pollBusy = false;
  }
}
setInterval(pollState, 500);
pollState();'''
if "async function pollState()" not in html:
    html, n = html_pattern.subn(replacement, html, count=1)
    if n != 1:
        raise SystemExit("ERROR: Browser connect() WebSocket block not found. No files changed.")

# Syntax check before any writes.
ast.parse(src)

if "from mar_mqtt import VisionMqttPublisher" not in src:
    raise SystemExit("ERROR: Publisher missing after patch")

for target, contents in ((pyfile, src), (htmlfile, html)):
    previous = target.read_text(encoding="utf-8-sig")
    if previous != contents:
        backup = target.with_suffix(target.suffix + ".before_mqtt.bak")
        shutil.copy2(target, backup)
        target.write_text(contents, encoding="utf-8")
        print("PATCHED", target.name, "BACKUP", backup.name)
    else:
        print("ALREADY PATCHED", target.name)

if (root / "mar_mqtt.py").resolve() != (project / "mar_mqtt.py").resolve():
    shutil.copy2(root / "mar_mqtt.py", project / "mar_mqtt.py")
print("MQTT PATCH SUCCESS. Add .env and pip install paho-mqtt python-dotenv.")
