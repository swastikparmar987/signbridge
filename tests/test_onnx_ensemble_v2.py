import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestONNXEnsembleV2(unittest.TestCase):
    def test_pytorch_vs_onnx_ensemble_parity(self):
        from signbridge.inference.onnx_predictor import KinematicEnsembleONNXPredictor
        from signbridge.inference.predictor import EnsemblePredictor, SignBridgePredictor

        cfg_path = PROJECT_ROOT / "trained_models" / "full_2731" / "ensemble_v2" / "ensemble_config.json"
        exp16_onnx = cfg_path.parent / "exp16.onnx"
        exp17_onnx = cfg_path.parent / "exp17.onnx"
        self.assertTrue(cfg_path.exists(), f"Missing {cfg_path}")
        self.assertTrue(exp16_onnx.exists(), f"Missing {exp16_onnx}")
        self.assertTrue(exp17_onnx.exists(), f"Missing {exp17_onnx}")

        pt_ens = EnsemblePredictor(
            [
                (
                    SignBridgePredictor(
                        checkpoint_path=PROJECT_ROOT / "trained_models" / "full_2731" / "exp16_kinematic_conformer" / "best_model.pth",
                        device="cpu",
                        ood_threshold=0.4842,
                    ),
                    0.55,
                ),
                (
                    SignBridgePredictor(
                        checkpoint_path=PROJECT_ROOT / "trained_models" / "full_2731" / "exp17_pyramid_bigru" / "best_model.pth",
                        device="cpu",
                        ood_threshold=0.4842,
                    ),
                    0.45,
                ),
            ],
            temperatures=[1.15, 1.15],
            ood_threshold=0.4842,
        )
        onnx_ens = KinematicEnsembleONNXPredictor(ensemble_config_path=cfg_path)

        rng = np.random.default_rng(42)
        seq = (rng.standard_normal((32, 42, 3), dtype=np.float32) * 0.18 + 0.5).astype(np.float32)

        pt_res = pt_ens.predict_sequence(seq, is_normalized=False, top_k=5)
        ox_res = onnx_ens.predict_sequence(seq, is_normalized=False, top_k=5)

        self.assertEqual(pt_res["raw_predicted_gloss"], ox_res["raw_predicted_gloss"])
        self.assertEqual(pt_res["is_no_sign"], ox_res["is_no_sign"])
        self.assertAlmostEqual(pt_res["raw_confidence"], ox_res["raw_confidence"], delta=0.05)
        self.assertAlmostEqual(pt_res["ood_prob"], ox_res["ood_prob"], delta=0.002)

        emb_pt = np.asarray(pt_res["embedding"], dtype=np.float32)
        emb_ox = np.asarray(ox_res["embedding"], dtype=np.float32)
        self.assertEqual(emb_pt.shape, (512,))
        self.assertEqual(emb_ox.shape, (512,))
        cos = float(np.dot(emb_pt, emb_ox) / (np.linalg.norm(emb_pt) * np.linalg.norm(emb_ox)))
        self.assertGreater(cos, 0.9999)

        # Resting zero sequence should trigger NO_SIGN identically
        empty_seq = np.zeros((32, 42, 3), dtype=np.float32)
        self.assertEqual(pt_ens.predict_sequence(empty_seq)["predicted_gloss"], "NO_SIGN")
        self.assertEqual(onnx_ens.predict_sequence(empty_seq)["predicted_gloss"], "NO_SIGN")

    def test_zero_torch_app_boot_and_endpoints(self):
        script = """
import sys, json, os, shutil, tempfile
from pathlib import Path
for mod in ("torch", "cv2", "mediapipe", "transformers", "kokoro_onnx", "pandas", "scipy"):
    sys.modules[mod] = None

from fastapi.testclient import TestClient
import signbridge.web.app
web_mod = sys.modules["signbridge.web.app"]

tmp_pers = tempfile.mkdtemp(prefix="sb_onnx_test_")
web_mod._personalization_manager.data_dir = Path(tmp_pers)
try:
  with TestClient(web_mod.app) as client:
    h = client.get("/api/health")
    assert h.status_code == 200, h.text
    hd = h.json()
    assert hd["status"] == "healthy"
    assert hd["engine"] == "onnxruntime", hd
    assert hd["num_classes"] == 2731, hd
    assert hd["model_variants"]["production_exp13"] is True, hd

    d = client.get("/api/dictionary/sign/HELLO")
    assert d.status_code == 200, d.text
    dd = d.json()
    assert dd["gloss"] == "HELLO"
    assert dd["total_frames"] == 32

    import numpy as np
    rng = np.random.default_rng(7)
    seq = (rng.standard_normal((32, 42, 3), dtype=np.float32) * 0.15 + 0.5).tolist()
    p = client.post("/api/predict/sequence", json={"sequence": seq, "top_k": 5})
    assert p.status_code == 200, p.text
    pd = p.json()
    assert len(pd["top_k"]) == 5
    assert len(pd["embedding"]) == 512

    ps = client.get("/api/personalization/profile?user_id=sb_unittest_user")
    assert ps.status_code == 200, ps.text
    assert ps.json()["signs"] == []
    assert ps.json()["enabled"] is False

    tts = client.post("/api/tts/speak", json={"text": "Hello"})
    assert tts.status_code == 200, tts.text
    assert tts.json()["success"] is False

    import psutil
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    assert rss_mb < 390.0, f"RSS too high for 512MB Render Free tier: {rss_mb:.1f} MB"
    print("ZERO_TORCH_E2E_OK", json.dumps({"engine": hd["engine"], "classes": hd["num_classes"], "rss_mb": round(rss_mb, 1), "top1": pd["top_k"][0]}))
finally:
  shutil.rmtree(tmp_pers, ignore_errors=True)
"""
        env = os.environ.copy()
        env["SIGNBRIDGE_USE_ONNX"] = "1"
        env["SIGNBRIDGE_ONNX_INT8"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertIn("ZERO_TORCH_E2E_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
