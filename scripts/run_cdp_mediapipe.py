import asyncio
import http.server
import json
import socketserver
import subprocess
import threading
import time
from pathlib import Path
import urllib.request
import websockets
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "tests"), **kwargs)

    def log_message(self, format, *args):
        pass

def start_server(port=8899):
    httpd = socketserver.TCPServer(("", port), CustomHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd

async def run_cdp_test():
    port = 8899
    httpd = start_server(port)

    brave_bin = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    proc = subprocess.Popen([
        brave_bin,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--no-sandbox",
        "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    await asyncio.sleep(1.0)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json/list")
        tabs = json.loads(req.read().decode("utf-8"))
        ws_url = tabs[0]["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await ws.send(json.dumps({"id": 2, "method": "Runtime.enable"}))

            target_url = f"http://127.0.0.1:{port}/fixtures/run_js_mediapipe.html"
            await ws.send(json.dumps({
                "id": 3,
                "method": "Page.navigate",
                "params": {"url": target_url}
            }))

            result = None
            for step in range(30):
                await asyncio.sleep(0.5)

                eval_msg = {
                    "id": 100 + step,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": "JSON.stringify({res: window.__MEDIAPIPE_RESULT__ || null, logs: window.__LOGS__ || [], err: window.__ERROR__ || null})",
                        "returnByValue": True
                    }
                }
                await ws.send(json.dumps(eval_msg))

                while True:
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=0.3)
                        data = json.loads(resp)
                        if data.get("id") == 100 + step:
                            val = data.get("result", {}).get("result", {}).get("value")
                            if val:
                                parsed = json.loads(val)
                                logs = parsed.get("logs", [])
                                if step % 4 == 0 and logs:
                                    print(f"   [Browser Log] {logs[-1]}")
                                if parsed.get("err"):
                                    print(f"   [Browser Error] {parsed['err']}")
                                if parsed.get("res"):
                                    result = parsed["res"]
                                    break
                    except asyncio.TimeoutError:
                        break
                if result:
                    break

            return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
        httpd.shutdown()

def main():
    print("=" * 70)
    print("🔬 PHASE 6: PYTHON MEDIAPIPE VS JAVASCRIPT MEDIAPIPE FORENSIC AUDIT")
    print("=" * 70)

    js_data = asyncio.run(run_cdp_test())
    with open(FIXTURES_DIR / "js_mediapipe_frame_20.json", "w") as f:
        json.dump(js_data, f, indent=2)

    py_json_path = FIXTURES_DIR / "python_mediapipe_frame_20.json"
    with open(py_json_path, "r") as f:
        py_data = json.load(f)

    print("\n1. DETECTION PARITY:")
    print(f"   Python MediaPipe Hands Detected:     {py_data['detected_hands_count']}")
    print(f"   JavaScript MediaPipe Hands Detected: {js_data.get('detected_hands_count', 0) if js_data else 0}")

    if js_data and js_data.get("hands"):
        py_h = py_data["hands"][0]
        js_h = js_data["hands"][0]

        print("\n2. HANDEDNESS & WRIST COORDINATES (Frame 20):")
        print(f"   Python Handedness:          {py_h['handedness']} (Confidence: {py_h['handedness_score']:.3f})")
        print(f"   JavaScript Handedness:      {js_h['handedness']} (Confidence: {js_h['handedness_score']:.3f})")
        print(f"   Python Wrist [X, Y, Z]:     [{py_h['wrist_x']:.5f}, {py_h['wrist_y']:.5f}, {py_h['wrist_z']:.5f}]")
        print(f"   JavaScript Wrist [X, Y, Z]: [{js_h['wrist_x']:.5f}, {js_h['wrist_y']:.5f}, {js_h['wrist_z']:.5f}]")

        py_lms = np.array(py_h["landmarks"], dtype=np.float32)
        js_lms = np.array(js_h["landmarks"], dtype=np.float32)

        diff_xyz = np.abs(py_lms - js_lms)
        max_diff_x = float(np.max(diff_xyz[:, 0]))
        max_diff_y = float(np.max(diff_xyz[:, 1]))
        max_diff_z = float(np.max(diff_xyz[:, 2]))

        mean_diff_x = float(np.mean(diff_xyz[:, 0]))
        mean_diff_y = float(np.mean(diff_xyz[:, 1]))
        mean_diff_z = float(np.mean(diff_xyz[:, 2]))

        print("\n3. RAW LANDMARK COORDINATE DELTAS (21 Points):")
        print(f"   X-axis: Max Diff = {max_diff_x:.6f} | Mean Diff = {mean_diff_x:.6f}")
        print(f"   Y-axis: Max Diff = {max_diff_y:.6f} | Mean Diff = {mean_diff_y:.6f}")
        print(f"   Z-axis: Max Diff = {max_diff_z:.6f} | Mean Diff = {mean_diff_z:.6f}")

        # Check Mirror Hypothesis
        mirror_diff_x = float(np.max(np.abs(py_lms[:, 0] - (1.0 - js_lms[:, 0]))))
        print(f"   Mirror Check: Max |py_x - (1 - js_x)| = {mirror_diff_x:.6f}")

        from signbridge.preprocessing.normalize import normalize_hand
        py_norm = normalize_hand(py_lms)
        js_norm = normalize_hand(js_lms)
        norm_diff = float(np.max(np.abs(py_norm - js_norm)))
        norm_mean = float(np.mean(np.abs(py_norm - js_norm)))

        print("\n4. NORMALIZED TENSOR COMPARISON:")
        print(f"   Max Normalized Difference:  {norm_diff:.6f}")
        print(f"   Mean Normalized Difference: {norm_mean:.6f}")

        if norm_diff < 0.05:
            print("\n   ✅ RESULT: Python and JS MediaPipe produce identical normalized representations (< 0.05 max diff).")
        else:
            print("\n   ⚠️ RESULT: Measurable difference between Python and JS normalization.")
    else:
        print("\n❌ Failed to extract JS landmarks.")

    print("=" * 70)

if __name__ == "__main__":
    main()
