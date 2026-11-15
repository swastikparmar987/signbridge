import json
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from signbridge.preprocessing.landmarks import compute_sequence_activity
from signbridge.web.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestApiContinuousPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        packed_dir = PROJECT_ROOT / "landmark_cache_packed"
        cls.has_packed = (packed_dir / "test_x.npy").exists()
        if cls.has_packed:
            cls.test_x = np.load(packed_dir / "test_x.npy", mmap_mode="r")
            cls.test_y = np.load(packed_dir / "test_y.npy", mmap_mode="r")
            cls.class_names = json.loads((packed_dir / "class_names.json").read_text(encoding="utf-8"))

    def _pick_two_active_clips(self):
        chosen = []
        seen_labels = set()
        for i in range(min(300, len(self.test_x))):
            act = compute_sequence_activity(self.test_x[i])
            lbl = int(self.test_y[i])
            if act["active_frames"] >= 16 and act["motion_score"] >= 0.12 and lbl not in seen_labels:
                chosen.append((self.class_names[lbl], np.asarray(self.test_x[i], dtype=np.float32)))
                seen_labels.add(lbl)
            if len(chosen) == 2:
                break
        return chosen

    def test_vocabulary_full_2731(self) -> None:
        resp = self.client.get("/api/vocabulary?limit=2731")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_matches"], 2731)
        self.assertEqual(data["returned"], 2731)

    def test_continuous_stream_finalize_and_reset(self) -> None:
        if not self.has_packed:
            self.skipTest("landmark_cache_packed not present")
        clips = self._pick_two_active_clips()
        self.assertEqual(len(clips), 2)

        sid = "pytest_cont_real_clips"
        self.assertEqual(self.client.post("/api/continuous/reset", json={"session_id": sid}).status_code, 200)

        stream = np.concatenate(
            [
                np.repeat(clips[0][1][:1], 3, axis=0),
                clips[0][1],
                np.repeat(clips[0][1][-1:], 6, axis=0),
                clips[1][1],
                np.repeat(clips[1][1][-1:], 20, axis=0),
                np.zeros((24, 42, 3), dtype=np.float32),
            ],
            axis=0,
        )

        finalized = None
        dt_ms = 1000.0 / 30.0
        t_ms = 0.0
        for pos in range(0, len(stream), 6):
            chunk = stream[pos : pos + 6]
            ts = [t_ms + k * dt_ms for k in range(len(chunk))]
            t_ms += len(chunk) * dt_ms
            r = self.client.post(
                "/api/continuous/stream",
                json={
                    "session_id": sid,
                    "frames": chunk.tolist(),
                    "timestamps_ms": ts,
                    "model_variant": "production_exp13",
                    "auto_finalize_on_pause": True,
                    "auto_translate_on_pause": True,
                    "top_k": 5,
                },
            )
            self.assertEqual(r.status_code, 200, r.text)
            payload = r.json()
            for key in ("newly_committed", "active_glosses", "active_hypotheses", "in_stroke", "pause_finalized"):
                self.assertIn(key, payload)
            if payload.get("pause_finalized") and payload.get("finalized_utterance"):
                finalized = payload["finalized_utterance"]
                break

        self.assertIsNotNone(finalized, "Expected trailing 1.4s pause to auto-finalize continuous utterance")
        self.assertGreaterEqual(len(finalized["glosses"]), 1)
        self.assertEqual(len(finalized["hypotheses"]), len(finalized["glosses"]))
        self.assertTrue(str(finalized.get("translation", {}).get("english", "")).strip())

        # Manual finalize on empty buffer should return empty cleanly after reset
        r_reset = self.client.post("/api/continuous/reset", json={"session_id": sid})
        self.assertEqual(r_reset.status_code, 200)
        r_fin = self.client.post(
            "/api/continuous/finalize",
            json={"session_id": sid, "model_variant": "production_exp13", "translate": True, "clear_after": True},
        )
        self.assertEqual(r_fin.status_code, 200)
        self.assertEqual(r_fin.json()["glosses"], [])

    def test_continuous_ood_zero_and_static_suppression(self) -> None:
        sid = "pytest_cont_ood_rest"
        self.client.post("/api/continuous/reset", json={"session_id": sid})

        # Create a rigid non-moving hand silhouette for 48 frames (zero velocity -> no active stroke)
        hand = np.zeros((42, 3), dtype=np.float32)
        hand[0] = [0.5, 0.7, 0.0]
        hand[5] = [0.48, 0.55, 0.0]
        hand[9] = [0.5, 0.52, 0.0]
        hand[13] = [0.52, 0.54, 0.0]
        hand[17] = [0.54, 0.57, 0.0]
        hand[8] = [0.47, 0.42, 0.0]
        hand[12] = [0.5, 0.4, 0.0]

        static_seq = np.repeat(hand[None, :, :], 48, axis=0)
        dt_ms = 1000.0 / 30.0
        for pos in range(0, len(static_seq), 8):
            chunk = static_seq[pos : pos + 8]
            ts = [(pos + k) * dt_ms for k in range(len(chunk))]
            r = self.client.post(
                "/api/continuous/stream",
                json={"session_id": sid, "frames": chunk.tolist(), "timestamps_ms": ts, "model_variant": "production_exp13"},
            )
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["newly_committed"], [])
            self.assertEqual(data["active_glosses"], [])

    def test_nlp_smooth_topk_and_non_manual(self) -> None:
        r = self.client.post(
            "/api/nlp/smooth",
            json={
                "glosses": ["TOMORROW", "YOU", "GO", "WHERE"],
                "hypotheses": [
                    [{"gloss": "TOMORROW", "confidence": 0.88}],
                    [{"gloss": "YOU", "confidence": 0.91}],
                    [{"gloss": "GO", "confidence": 0.76}, {"gloss": "COME", "confidence": 0.14}],
                    [{"gloss": "WHERE", "confidence": 0.79}, {"gloss": "WHAT", "confidence": 0.12}],
                ],
                "non_manual": {"wh_question": True, "yn_question": False, "negative": False},
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        english = str(data.get("english", "")).strip()
        self.assertTrue(english)
        self.assertNotIn("<think>", english)
        self.assertTrue(english.endswith("?"), f"Expected WH-question mark, got: {english!r}")

    def test_fingerspelling_predict_endpoint(self) -> None:
        # Static flat hand resting frame (ptp < 0.035 -> should suppress cleanly)
        flat = np.full((24, 42, 3), 0.5, dtype=np.float32).tolist()
        r = self.client.post(
            "/api/fingerspelling/predict",
            json={"sequence": flat, "min_char_prob": 0.25, "target_len": 24},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("char", data)
        self.assertIn("latency_ms", data)
        self.assertIn("is_active", data)
        self.assertFalse(data["is_active"])
        self.assertIsNone(data["char"])

    def test_custom_sign_learner_arbitrary_name_and_prediction(self) -> None:
        if not self.has_packed:
            self.skipTest("landmark_cache_packed not present")
        clips = self._pick_two_active_clips()
        self.assertGreaterEqual(len(clips), 1)
        base_clip = clips[0][1]
        # Resample to 32 frames (shape expected by /api/personalization/sample)
        idx32 = np.linspace(0, len(base_clip) - 1, 32).round().astype(int)
        seq32 = np.asarray(base_clip[idx32], dtype=np.float32)

        uid = "pytest_custom_user"
        custom_name = "COFFEE_BREAK"
        sid = "pytest_custom_cont"
        try:
            self.client.delete(f"/api/personalization?user_id={uid}")

            # 1. Enroll 3 realistic variations (slight speed/trajectory differences so normalization stays natural & non-duplicate)
            idx_fast = np.linspace(0, max(1, int((len(base_clip) - 1) * 0.84)), 32).round().astype(int)
            idx_late = np.linspace(min(4, len(base_clip) - 1), len(base_clip) - 1, 32).round().astype(int)
            phase = np.sin(np.linspace(0.0, np.pi, 32))[:, None, None].astype(np.float32)
            v1 = seq32.copy()
            v2 = np.asarray(base_clip[idx_fast], dtype=np.float32) + 0.018 * phase
            v3 = np.asarray(base_clip[idx_late], dtype=np.float32) - 0.016 * phase
            variants = [v1, v2, v3]
            for idx, v in enumerate(variants):
                r = self.client.post(
                    "/api/personalization/sample",
                    json={
                        "gloss": custom_name,
                        "sequence": v.tolist(),
                        "user_id": uid,
                        "model_variant": "production_exp13",
                    },
                )
                self.assertEqual(r.status_code, 200, r.text)
                payload = r.json()
                self.assertTrue(payload["accepted"])
                self.assertEqual(payload["gloss"], custom_name)
                self.assertFalse(payload["in_base_vocab"])
                self.assertEqual(payload["learned"], idx >= 2)

            # 2. Explicitly train custom model & test disable/enable endpoints
            r_train = self.client.post("/api/personalization/train", json={"user_id": uid})
            self.assertEqual(r_train.status_code, 200, r_train.text)
            train_data = r_train.json()
            self.assertTrue(train_data["enabled"])
            self.assertEqual(train_data["total_signs"], 1)
            self.assertEqual(train_data["trained_signs"][0]["gloss"], custom_name)

            r_dis = self.client.post("/api/personalization/disable", json={"user_id": uid})
            self.assertEqual(r_dis.status_code, 200)
            self.assertFalse(r_dis.json()["enabled"])
            r_en = self.client.post("/api/personalization/enable", json={"user_id": uid})
            self.assertEqual(r_en.status_code, 200)
            self.assertTrue(r_en.json()["enabled"])

            # 3. Sequence prediction must match COFFEE_BREAK on BOTH production_exp13 & demo_50
            for variant in ("production_exp13", "demo_50"):
                r_seq = self.client.post(
                    "/api/predict/sequence",
                    json={
                        "sequence": seq32.tolist(),
                        "model_variant": variant,
                        "user_id": uid,
                        "personalization_enabled": True,
                        "top_k": 5,
                    },
                )
                self.assertEqual(r_seq.status_code, 200, r_seq.text)
                pred = r_seq.json()
                self.assertEqual(pred["predicted_gloss"], custom_name, f"{variant} returned {pred}")
                self.assertTrue(pred["is_custom_sign"], f"{variant} is_custom_sign missing")
                self.assertTrue(pred["personalization"]["matched"], f"{variant} personalization not matched")

            # 4. Continuous stream + auto-finalize pause must emit custom sign COFFEE_BREAK
            self.client.post("/api/continuous/reset", json={"session_id": sid})
            stream = np.concatenate(
                [
                    np.repeat(base_clip[:1], 3, axis=0),
                    base_clip,
                    np.repeat(base_clip[-1:], 45, axis=0),
                ],
                axis=0,
            )
            dt_ms = 1000.0 / 30.0
            t_ms = 0.0
            seen_glosses = []
            finalized = None
            for pos in range(0, len(stream), 6):
                chunk = stream[pos : pos + 6]
                ts = [t_ms + k * dt_ms for k in range(len(chunk))]
                t_ms += len(chunk) * dt_ms
                rc = self.client.post(
                    "/api/continuous/stream",
                    json={
                        "session_id": sid,
                        "frames": chunk.tolist(),
                        "timestamps_ms": ts,
                        "model_variant": "production_exp13",
                        "user_id": uid,
                        "personalization_enabled": True,
                        "auto_finalize_on_pause": True,
                        "auto_translate_on_pause": False,
                        "top_k": 5,
                    },
                )
                self.assertEqual(rc.status_code, 200, rc.text)
                pdata = rc.json()
                seen_glosses.extend(pdata.get("active_glosses", []))
                if pdata.get("pause_finalized") and pdata.get("finalized_utterance"):
                    finalized = pdata["finalized_utterance"]
                    break

            self.assertIsNotNone(finalized)
            self.assertIn(custom_name, finalized.get("glosses", []) or seen_glosses)
        finally:
            self.client.post("/api/continuous/reset", json={"session_id": sid})
            self.client.delete(f"/api/personalization?user_id={uid}")

    def test_dictionary_endpoints_serve_32_frame_exemplars(self):
        r_vocab = self.client.get("/api/vocabulary?limit=10&category=two_hand")
        self.assertEqual(r_vocab.status_code, 200, r_vocab.text)
        vdata = r_vocab.json()
        self.assertGreater(vdata["total_matches"], 1000)
        self.assertTrue(all(item.get("two_hand") is True for item in vdata["items"]))

        for gloss, expected_two in [("HELLO", False), ("FRIEND", True)]:
            r_sign = self.client.get(f"/api/dictionary/sign/{gloss}")
            self.assertEqual(r_sign.status_code, 200, r_sign.text)
            sdata = r_sign.json()
            self.assertEqual(sdata["gloss"], gloss)
            self.assertEqual(sdata["two_hand"], expected_two)
            self.assertEqual(sdata["total_frames"], 32)
            self.assertEqual(len(sdata["frames"]), 32)
            self.assertEqual(len(sdata["frames"][0]), 42)
            self.assertEqual(len(sdata["steps"]), 3)

    def test_personalization_isolated_per_new_user(self):
        clips = self._pick_two_active_clips()
        self.assertGreaterEqual(len(clips), 1)
        base_clip = clips[0][1]
        idx32 = np.linspace(0, len(base_clip) - 1, 32).round().astype(int)
        seq32 = np.asarray(base_clip[idx32], dtype=np.float32)

        user_a = "pytest_user_a"
        user_b = "pytest_user_b"
        try:
            self.client.delete(f"/api/personalization?user_id={user_a}")
            self.client.delete(f"/api/personalization?user_id={user_b}")

            for idx in range(3):
                phase = (0.015 * (idx + 1)) * np.sin(np.linspace(0.0, np.pi, 32))[:, None, None].astype(np.float32)
                sample = seq32 if idx == 0 else seq32 + phase
                r = self.client.post(
                    "/api/personalization/sample",
                    json={
                        "gloss": "MY_CUSTOM_WAVE",
                        "sequence": sample.tolist(),
                        "user_id": user_a,
                        "model_variant": "production_exp13",
                    },
                )
                self.assertEqual(r.status_code, 200, r.text)

            r_train = self.client.post("/api/personalization/train", json={"user_id": user_a})
            self.assertEqual(r_train.status_code, 200, r_train.text)
            self.assertEqual(r_train.json()["total_signs"], 1)

            # Brand-new user B must see 0 custom signs, disabled routing, and intact base model prediction.
            r_prof_b = self.client.get(f"/api/personalization/profile?user_id={user_b}")
            self.assertEqual(r_prof_b.status_code, 200, r_prof_b.text)
            prof_b = r_prof_b.json()
            self.assertEqual(prof_b["signs"], [])
            self.assertFalse(prof_b["enabled"])

            r_base = self.client.post(
                "/api/predict/sequence",
                json={
                    "sequence": seq32.tolist(),
                    "model_variant": "production_exp13",
                    "user_id": user_b,
                    "personalization_enabled": True,
                    "top_k": 5,
                },
            )
            self.assertEqual(r_base.status_code, 200, r_base.text)
            pred_b = r_base.json()
            self.assertNotEqual(pred_b["predicted_gloss"], "MY_CUSTOM_WAVE")
            self.assertFalse(pred_b.get("is_custom_sign", False))
            self.assertFalse(pred_b.get("personalization", {}).get("matched", False))
        finally:
            self.client.delete(f"/api/personalization?user_id={user_a}")
            self.client.delete(f"/api/personalization?user_id={user_b}")


if __name__ == "__main__":
    unittest.main()
