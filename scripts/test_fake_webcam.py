import asyncio
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

async def run_cdp_fake_camera_test(y4m_path, wait_before_capture=2.5, capture_wait=3.5):
    brave_bin = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    port = 9555
    cmd = [
        brave_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--remote-debugging-port={port}",
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-video-capture={y4m_path}",
        "--autoplay-policy=no-user-gesture-required",
        "http://127.0.0.1:8000/",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await asyncio.sleep(1.5)

    try:
        req = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")
        tabs = json.loads(req.read().decode("utf-8"))
        ws_url = tabs[0]["webSocketDebuggerUrl"]

        pending_futures = {}
        console_logs = []

        async with websockets.connect(ws_url) as ws:
            async def reader():
                while True:
                    try:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        msg_id = msg.get("id")
                        if msg_id and msg_id in pending_futures:
                            pending_futures[msg_id].set_result(msg)
                        elif msg.get("method") == "Runtime.consoleAPICalled":
                            args = [a.get("value", a.get("description", "")) for a in msg["params"]["args"]]
                            console_logs.append(" ".join(str(x) for x in args))
                        elif msg.get("method") == "Runtime.exceptionThrown":
                            console_logs.append(f"EXCEPTION: {msg['params']['exceptionDetails']}")
                    except Exception as e:
                        print(f"Websocket reader exception: {e}")
                        print(f"Browser process status: {proc.poll()}")
                        break

            reader_task = asyncio.create_task(reader())

            async def send_command(method, params=None):
                nonlocal req_id
                req_id += 1
                curr_id = req_id
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                pending_futures[curr_id] = fut
                await ws.send(json.dumps({"id": curr_id, "method": method, "params": params or {}}))
                try:
                    resp = await asyncio.wait_for(fut, timeout=10.0)
                    del pending_futures[curr_id]
                    return resp
                except Exception as e:
                    print(f"Error executing {method}: {e}")
                    print(f"Browser logs: {console_logs}")
                    raise

            req_id = 0
            await send_command("Page.enable")
            await send_command("Runtime.enable")

            # 1. Wait for page load
            await asyncio.sleep(2.0)

            # 2. Click "Enable Camera"
            print("Clicking enable camera...")
            await send_command("Runtime.evaluate", {
                "expression": "document.getElementById('btn-start-camera').click()"
            })
            print("Clicked enable camera.")

            # 3. Allow camera stream & MediaPipe to start
            await asyncio.sleep(wait_before_capture)

            # 4. Click "Sign Now"
            print("Clicking sign now...")
            await send_command("Runtime.evaluate", {
                "expression": "document.getElementById('btn-capture-sign').click()"
            })
            print("Clicked sign now.")

            # 5. Wait for 2.0s capture + network roundtrip
            await asyncio.sleep(capture_wait)

            # 6. Read prediction state from DOM
            eval_expr = """
            (() => {
              const primary = document.getElementById('primary-gloss')?.textContent || '—';
              const conf = document.getElementById('primary-confidence')?.textContent || '0.0%';
              const badge = document.getElementById('confidence-badge')?.textContent || '';
              const assessment = document.getElementById('confidence-assessment')?.textContent || '';
              const top5Els = document.querySelectorAll('#top5-list .ranking-item:not(.empty)');
              const top5 = Array.from(top5Els).map(el => ({
                gloss: el.querySelector('.rank-gloss')?.textContent || '',
                confidence: el.querySelector('.rank-prob')?.textContent || ''
              }));
              return JSON.stringify({
                primary_gloss: primary,
                confidence: conf,
                badge: badge,
                assessment: assessment,
                top5: top5
              });
            })()
            """
            eval_resp = await send_command("Runtime.evaluate", {
                "expression": eval_expr,
                "returnByValue": True
            })

            reader_task.cancel()

            val = eval_resp.get("result", {}).get("result", {}).get("value")
            dom_data = json.loads(val) if val else {}
            return dom_data, console_logs

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

def main():
    print("=" * 75)
    print("🤖 PHASE 5: REPRODUCIBLE END-TO-END FAKE WEBCAM TEST")
    print("=" * 75)

    test_cases = [
        ("APPLE", "fake_apple.y4m"),
        ("LABEL", "fake_label.y4m"),
        ("BLUE",  "fake_blue.y4m"),
        ("LOVE",  "fake_love.y4m"),
        ("WATER", "fake_water.y4m"),
    ]

    all_results = []
    for target_gloss, y4m_filename in test_cases:
        y4m_path = (FIXTURES_DIR / y4m_filename).resolve()
        print(f"\n▶ Testing Sign: {target_gloss} ({y4m_filename})...")

        dom_data, logs = asyncio.run(run_cdp_fake_camera_test(y4m_path))
        p_gloss = dom_data.get("primary_gloss", "—")
        conf = dom_data.get("confidence", "0.0%")
        top5 = dom_data.get("top5", [])

        is_top1 = p_gloss.upper() == target_gloss.upper()
        is_top5 = any(t.get("gloss", "").upper() == target_gloss.upper() for t in top5)

        status_str = "✅ TOP-1 MATCH" if is_top1 else ("🟡 TOP-5 MATCH" if is_top5 else "❌ MISMATCH")

        print(f"   Top-1 Recognized: {p_gloss} ({conf}) [{status_str}]")
        print(f"   Top-5 Candidates: {[(t['gloss'], t['confidence']) for t in top5]}")
        for l in logs[-4:]:
            print(f"   [Browser Log] {l}")

        all_results.append({
            "target": target_gloss,
            "top1": p_gloss,
            "confidence": conf,
            "top5": top5,
            "status": status_str
        })

    print("\n" + "=" * 75)
    print("🏁 FINAL FAKE WEBCAM SUMMARY:")
    print("=" * 75)
    for r in all_results:
        print(f"   {r['target']:<10} | Top-1: {r['top1']:<15} | Conf: {r['confidence']:<8} | {r['status']}")
    print("=" * 75)

if __name__ == "__main__":
    main()
