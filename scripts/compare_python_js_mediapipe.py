import http.server
import json
import socketserver
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

received_data = {}
server_done_event = threading.Event()

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "tests"), **kwargs)

    def do_POST(self):
        global received_data
        if self.path == "/api/save_landmarks":
            content_length = int(self.headers["Content-Length"])
            post_body = self.rfile.read(content_length)
            received_data = json.loads(post_body.decode("utf-8"))
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            server_done_event.set()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def run_server(port):
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        httpd.timeout = 1.0
        while not server_done_event.is_set():
            httpd.handle_request()

def main():
    print("=" * 70)
    print("🔬 PHASE 6: PYTHON MEDIAPIPE VS JAVASCRIPT MEDIAPIPE COMPARISON")
    print("=" * 70)

    port = 8899
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    time.sleep(0.5)

    py_json_path = FIXTURES_DIR / "python_mediapipe_frame_20.json"
    with open(py_json_path, "r") as f:
        py_data = json.load(f)

    brave_bin = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    url = f"http://127.0.0.1:{port}/fixtures/run_js_mediapipe.html"

    cmd = [
        brave_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--remote-allow-origins=*",
        url,
    ]

    print(f"Launching Brave Browser to process test frame with MediaPipe JS...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for response up to 15 seconds
    server_done_event.wait(timeout=15.0)
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()

    global received_data
    js_data = received_data

    with open(FIXTURES_DIR / "js_mediapipe_frame_20.json", "w") as f:
        json.dump(js_data, f, indent=2)

    print("\n1. DETECTION COUNT:")
    print(f"   Python MediaPipe Hands:     {py_data['detected_hands_count']}")
    print(f"   JavaScript MediaPipe Hands: {js_data.get('detected_hands_count', 0)}")

    if js_data and js_data.get("hands"):
        py_h = py_data["hands"][0]
        js_h = js_data["hands"][0]

        print("\n2. HANDEDNESS & WRIST COORDINATES:")
        print(f"   Python Handedness:          {py_h['handedness']} (Confidence: {py_h['handedness_score']:.3f})")
        print(f"   JavaScript Handedness:      {js_h['handedness']} (Confidence: {js_h['handedness_score']:.3f})")
        print(f"   Python Wrist [X, Y, Z]:     [{py_h['wrist_x']:.5f}, {py_h['wrist_y']:.5f}, {py_h['wrist_z']:.5f}]")
        print(f"   JavaScript Wrist [X, Y, Z]: [{js_h['wrist_x']:.5f}, {js_h['wrist_y']:.5f}, {js_h['wrist_z']:.5f}]")

        py_lms = np.array(py_h["landmarks"], dtype=np.float32)
        js_lms = np.array(js_h["landmarks"], dtype=np.float32)

        # Coordinate differences
        diff_xyz = np.abs(py_lms - js_lms)
        max_diff_x = np.max(diff_xyz[:, 0])
        max_diff_y = np.max(diff_xyz[:, 1])
        max_diff_z = np.max(diff_xyz[:, 2])

        mean_diff_x = np.mean(diff_xyz[:, 0])
        mean_diff_y = np.mean(diff_xyz[:, 1])
        mean_diff_z = np.mean(diff_xyz[:, 2])

        print("\n3. RAW LANDMARK DIFFERENCE METRICS (21 Points):")
        print(f"   X-axis: Max Diff = {max_diff_x:.6f} | Mean Diff = {mean_diff_x:.6f}")
        print(f"   Y-axis: Max Diff = {max_diff_y:.6f} | Mean Diff = {mean_diff_y:.6f}")
        print(f"   Z-axis: Max Diff = {max_diff_z:.6f} | Mean Diff = {mean_diff_z:.6f}")

        # Check Mirror Hypothesis
        mirror_diff_x = np.abs(py_lms[:, 0] - (1.0 - js_lms[:, 0]))
        print(f"   Mirror Check: Max |py_x - (1 - js_x)| = {np.max(mirror_diff_x):.6f}")

        from signbridge.preprocessing.normalize import normalize_hand
        py_norm = normalize_hand(py_lms)
        js_norm = normalize_hand(js_lms)
        norm_diff = np.max(np.abs(py_norm - js_norm))
        norm_mean = np.mean(np.abs(py_norm - js_norm))

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
