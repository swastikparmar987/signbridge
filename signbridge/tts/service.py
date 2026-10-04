import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import soundfile as sf
    from kokoro_onnx import Kokoro
    KOKORO_MODULE_AVAILABLE = True
except ImportError:
    KOKORO_MODULE_AVAILABLE = False

logger = logging.getLogger(__name__)

class KokoroTTSService:
    """
    Primary local Text-To-Speech service for SignBridge using Kokoro ONNX.
    """
    def __init__(self, model_dir: Optional[Path] = None, output_dir: Optional[Path] = None, default_voice: str = "af_sarah"):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.model_dir = model_dir or (project_root / "trained_models" / "tts")
        self.output_dir = output_dir or (project_root / "signbridge" / "web" / "static" / "audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_voice = default_voice
        self.kokoro: Optional[Any] = None
        self._audio_cache: Dict[str, Dict[str, Any]] = {}
        self._speech_queue: List[str] = []
        self._current_speaking: Optional[str] = None
        self._is_initialized = False
        self._init_error: Optional[str] = None
        self._load_model()

    def _load_model(self):
        if not KOKORO_MODULE_AVAILABLE:
            self._init_error = "kokoro_onnx or soundfile module not installed"
            logger.warning(self._init_error)
            return

        model_path = self.model_dir / "kokoro-v0_19.onnx"
        voices_path = self.model_dir / "voices.bin"

        if not model_path.exists() or not voices_path.exists():
            self._init_error = f"Kokoro model files missing at {self.model_dir}"
            logger.warning(self._init_error)
            return

        try:
            self.kokoro = Kokoro(str(model_path), str(voices_path))
            self._is_initialized = True
            self._init_error = None
            logger.info("Kokoro ONNX TTS initialized successfully.")
        except Exception as e:
            self._init_error = f"Failed to initialize Kokoro: {str(e)}"
            logger.error(self._init_error)

    def is_available(self) -> bool:
        return self._is_initialized and self.kokoro is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "engine": "Kokoro ONNX",
            "voice": self.default_voice,
            "status": "Speaking" if self._current_speaking else ("Ready" if self.is_available() else "Unavailable"),
            "queue_length": len(self._speech_queue),
            "cached_items": len(self._audio_cache),
            "error": self._init_error
        }

    def clear_queue(self):
        """Clear all pending items in the speech queue."""
        self._speech_queue.clear()

    def stop(self):
        """Stop current playback and clear pending queue."""
        self.clear_queue()
        self._current_speaking = None

    def speak(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate audio for the given text.
        Includes session-level audio caching and duplicate speech protection.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "success": False,
                "error": "Empty text provided for TTS",
                "audio_url": None,
                "latency_ms": 0.0,
                "duration_s": 0.0,
                "cached": False
            }

        if not self.is_available():
            return {
                "success": False,
                "error": self._init_error or "Kokoro TTS is unavailable",
                "audio_url": None,
                "latency_ms": 0.0,
                "duration_s": 0.0,
                "cached": False
            }

        target_voice = voice or self.default_voice
        text_hash = hashlib.md5(f"{clean_text}_{target_voice}".encode("utf-8")).hexdigest()

        # Session Audio Caching
        if text_hash in self._audio_cache:
            cached_data = self._audio_cache[text_hash]
            wav_path = Path(cached_data["file_path"])
            if wav_path.exists():
                return {
                    "success": True,
                    "audio_url": cached_data["audio_url"],
                    "latency_ms": 1.0,
                    "duration_s": cached_data["duration_s"],
                    "cached": True,
                    "error": None
                }

        start_time = time.time()
        try:
            self._current_speaking = clean_text
            samples, sample_rate = self.kokoro.create(
                clean_text,
                voice=target_voice,
                speed=1.0,
                lang="en-us"
            )

            filename = f"speech_{text_hash[:12]}.wav"
            file_path = self.output_dir / filename
            sf.write(str(file_path), samples, sample_rate)

            latency_ms = round((time.time() - start_time) * 1000.0, 2)
            duration_s = round(len(samples) / sample_rate, 2)
            audio_url = f"/static/audio/{filename}"

            res_data = {
                "file_path": str(file_path),
                "audio_url": audio_url,
                "duration_s": duration_s,
                "latency_ms": latency_ms
            }
            self._audio_cache[text_hash] = res_data
            self._current_speaking = None

            return {
                "success": True,
                "audio_url": audio_url,
                "latency_ms": latency_ms,
                "duration_s": duration_s,
                "cached": False,
                "error": None
            }

        except Exception as e:
            self._current_speaking = None
            latency_ms = round((time.time() - start_time) * 1000.0, 2)
            logger.error(f"Kokoro audio generation failed: {e}")
            return {
                "success": False,
                "error": f"Kokoro TTS generation error: {str(e)}",
                "audio_url": None,
                "latency_ms": latency_ms,
                "duration_s": 0.0,
                "cached": False
            }
