import asyncio
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
y4m_path = (PROJECT_ROOT / "tests" / "fixtures" / "fake_apple.y4m").resolve()

brave_bin = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
cmd = [
    brave_bin,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--remote-debugging-port=9444",
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    f"--use-file-for-fake-video-capture={y4m_path}",
    "--autoplay-policy=no-user-gesture-required",
    "http://127.0.0.1:8000/",
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2.0)

async def run_debug():
    req = urllib.request.urlopen("http://127.0.0.1:9444/json/list")
    tabs = json.loads(req.read().decode("utf-8"))
    ws_url = tabs[0]["webSocketDebuggerUrl"]

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
        await ws.send(json.dumps({"id": 2, "method": "Runtime.enable"}))

        async def listen_logs():
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("method") == "Runtime.consoleAPICalled":
                        args = [a.get("value", a.get("description", "")) for a in data["params"]["args"]]
                        print(f"   [CONSOLE] {' '.join(str(x) for x in args)}", flush=True)
                    elif data.get("method") == "Runtime.exceptionThrown":
                        print(f"   [EXCEPTION] {data['params']['exceptionDetails']}", flush=True)
                except Exception:
                    break

        log_task = asyncio.create_task(listen_logs())

        await asyncio.sleep(2.0)

        print("1. Clicking 'btn-start-camera'...", flush=True)
        await ws.send(json.dumps({
            "id": 10,
            "method": "Runtime.evaluate",
            "params": {"expression": "document.getElementById('btn-start-camera').click()"}
        }))

        # Let camera run for 3 seconds so MediaPipe processes frames
        await asyncio.sleep(3.0)

        print("2. Clicking 'btn-capture-sign'...", flush=True)
        await ws.send(json.dumps({
            "id": 20,
            "method": "Runtime.evaluate",
            "params": {"expression": "document.getElementById('btn-capture-sign').click()"}
        }))

        # Wait 3 seconds for 2.0s capture + inference
        await asyncio.sleep(3.0)

        print("3. Querying Prediction State...", flush=True)
        eval_pred = """
        (() => {
          return JSON.stringify({
            primary_gloss: document.getElementById('primary-gloss')?.textContent,
            confidence: document.getElementById('primary-confidence')?.textContent,
            assessment: document.getElementById('confidence-assessment')?.textContent,
            top5: Array.from(document.querySelectorAll('#top5-list .ranking-item')).map(el => ({
              text: el.innerText
            }))
          });
        })()
        """
        await ws.send(json.dumps({
            "id": 30,
            "method": "Runtime.evaluate",
            "params": {"expression": eval_pred, "returnByValue": True}
        }))

        while True:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(resp)
                if data.get("id") == 30:
                    val = data.get("result", {}).get("result", {}).get("value")
                    print(f"\n🎯 Prediction Result: {val}", flush=True)
                    break
            except asyncio.TimeoutError:
                break

        await asyncio.sleep(1.0)
        log_task.cancel()

try:
    asyncio.run(run_debug())
finally:
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
