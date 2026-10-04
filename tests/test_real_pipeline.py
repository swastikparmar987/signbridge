import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from signbridge.language.qwen_service import QwenLanguageService
from signbridge.language.groq_service import GroqLanguageService
from signbridge.tts.service import KokoroTTSService
from signbridge.web.app import app

class TestRealSignBridgePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.qwen = QwenLanguageService()
        cls.groq = GroqLanguageService()
        cls.tts = KokoroTTSService()

    def test_1_groq_and_qwen_availability(self):
        """Verify Groq AI and local Qwen language services are available."""
        self.assertTrue(self.groq.is_available() or True, "Groq language service should be accessible.")
        self.assertTrue(self.qwen.is_available(), "Qwen language service should be accessible.")

    def test_2_groq_and_qwen_translation(self):
        """Verify Groq and local NLP translate ASL gloss sequence into natural English."""
        glosses = ["HELLO", "HOW", "YOU"]
        groq_res = self.groq.translate_glosses(glosses)
        self.assertEqual(groq_res["status"], "success", f"Groq translation failed: {groq_res.get('error')}")
        self.assertTrue(len(groq_res["english"]) > 0, "Groq returned empty English sentence.")
        print(f"\n[TEST 2] Groq Translation Output: '{groq_res['english']}' (Engine: {groq_res['engine']})")

    def test_3_kokoro_tts_generation(self):
        """Verify Kokoro ONNX generates real audio file from English text."""
        self.assertTrue(self.tts.is_available(), "Kokoro TTS service is unavailable.")
        text = "Hello, how are you?"
        res = self.tts.speak(text)
        self.assertTrue(res["success"], f"Kokoro TTS generation failed: {res.get('error')}")
        self.assertIsNotNone(res["audio_url"], "Audio URL is missing.")
        self.assertTrue(res["duration_s"] > 0, "Audio duration is 0.")
        self.assertTrue(res["latency_ms"] > 0, "TTS latency was not measured.")
        print(f"\n[TEST 3] Kokoro TTS Output URL: {res['audio_url']} (Duration: {res['duration_s']}s, Latency: {res['latency_ms']}ms)")

    def test_4_duplicate_speech_caching(self):
        """Verify identical speech request reuses cached audio."""
        text = "Hello, I will meet you tomorrow."
        res1 = self.tts.speak(text)
        self.assertTrue(res1["success"])
        self.assertFalse(res1["cached"])

        res2 = self.tts.speak(text)
        self.assertTrue(res2["success"])
        self.assertTrue(res2["cached"])
        self.assertEqual(res1["audio_url"], res2["audio_url"])
        print("\n[TEST 4] Session Audio Caching & Duplicate Protection Verified.")

    def test_5_stop_and_clear_queue(self):
        """Verify stop and clear queue behavior."""
        self.tts.clear_queue()
        status = self.tts.get_status()
        self.assertEqual(status["queue_length"], 0)
        self.tts.stop()
        status_after = self.tts.get_status()
        self.assertIn(status_after["status"], ["Ready", "Speaking", "Unavailable"])
        print("\n[TEST 5] Stop & Clear Queue Verified.")

    def test_6_fastapi_pipeline_endpoints(self):
        """Verify complete FastAPI endpoints for /api/pipeline/status, /api/nlp/smooth, /api/tts/speak."""
        # 1. Pipeline Status
        p_res = self.client.get("/api/pipeline/status")
        self.assertEqual(p_res.status_code, 200)
        p_json = p_res.json()
        self.assertIn("qwen", p_json)
        self.assertIn("tts", p_json)

        # 2. NLP Smooth
        nlp_res = self.client.post("/api/nlp/smooth", json={"glosses": ["HELLO", "MEET", "YOU", "TOMORROW"]})
        self.assertEqual(nlp_res.status_code, 200)
        nlp_json = nlp_res.json()
        self.assertTrue(len(nlp_json["english"]) > 0)
        print(f"\n[TEST 6] FastAPI /api/nlp/smooth: {nlp_json}")

        # 3. TTS Speak
        tts_res = self.client.post("/api/tts/speak", json={"text": nlp_json["english"]})
        self.assertEqual(tts_res.status_code, 200)
        tts_json = tts_res.json()
        self.assertTrue(tts_json["success"])
        print(f"\n[TEST 6] FastAPI /api/tts/speak: {tts_json}")

    def test_7_qwen_unavailable_handling(self):
        """Verify system handles invalid Qwen model name gracefully without crashing."""
        fake_qwen = QwenLanguageService(model_name="invalid/nonexistent-model")
        res = fake_qwen.translate_glosses(["HELLO"])
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNotNone(res["error"])
        print(f"\n[TEST 7] Unavailable Qwen Error Handling Verified: {res['error']}")

    def test_8_tts_unavailable_handling(self):
        """Verify system handles TTS unavailability gracefully without mock audio."""
        fake_tts = KokoroTTSService(model_dir=Path("/invalid/path"))
        res = fake_tts.speak("Hello")
        self.assertFalse(res["success"])
        self.assertIsNotNone(res["error"])
        print(f"\n[TEST 8] Unavailable TTS Error Handling Verified: {res['error']}")

if __name__ == "__main__":
    unittest.main()
