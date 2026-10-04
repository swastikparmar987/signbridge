import time
import re
import logging
from typing import Dict, Any, Optional, List

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class QwenLanguageService:
    """
    Local ASL Gloss to Natural English Language Service for SignBridge.
    Translates ASL Glosses into natural English without requiring external servers like Ollama
    or automatic large model downloads.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device or ("mps" if TRANSFORMERS_AVAILABLE and torch.backends.mps.is_available() else "cpu") if TRANSFORMERS_AVAILABLE else "cpu"
        self._is_initialized = False
        self._init_error = None
        self.system_prompt = (
            "You are an ASL gloss to natural English translator.\n"
            "Convert the input ASL glosses into a single fluent, natural English sentence."
        )

        # Pre-compiled high-frequency ASL gloss to English mappings
        self.gloss_phrases = {
            "HELLO HOW YOU": "Hello, how are you?",
            "HOW YOU": "How are you?",
            "WHAT YOUR NAME": "What is your name?",
            "YOU NAME WHAT": "What is your name?",
            "ME NAME": "My name is",
            "NICE MEET YOU": "Nice to meet you.",
            "HAPPY MEET YOU": "Happy to meet you.",
            "THANK YOU": "Thank you.",
            "GOOD MORNING": "Good morning.",
            "GOOD NIGHT": "Good night.",
            "SEE YOU LATER": "See you later.",
            "ME WANT WATER": "I want water.",
            "ME NEED HELP": "I need help.",
            "WHERE BATHROOM": "Where is the bathroom?",
            "PLEASE HELP ME": "Please help me.",
            "ME LOVE YOU": "I love you.",
            "YES PLEASE": "Yes, please.",
            "NO THANK YOU": "No, thank you."
        }

    def load_model(self) -> bool:
        """Loads Qwen tokenizer and model into PyTorch memory ONLY if cached locally."""
        if self._is_initialized:
            return True

        if not TRANSFORMERS_AVAILABLE:
            self._init_error = "transformers or torch package not installed"
            return False

        try:
            logger.info(f"Checking for local cached {self.model_name} model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device,
                local_files_only=True
            )
            self._is_initialized = True
            self._init_error = None
            logger.info(f"Loaded local cached model {self.model_name}.")
            return True
        except Exception as e:
            self._init_error = f"Local cached model not loaded (using fast local rule engine): {str(e)}"
            return False

    def is_available(self) -> bool:
        return True

    def _rule_based_translation(self, glosses: List[str]) -> str:
        """Instant high-quality local rule-based ASL gloss to English translator."""
        clean_glosses = [g.strip().upper() for g in glosses if g.strip()]
        if not clean_glosses:
            return ""

        full_key = " ".join(clean_glosses)
        if full_key in self.gloss_phrases:
            return self.gloss_phrases[full_key]

        # Pronoun & Grammar mapping rules
        pronoun_map = {
            "ME": "I",
            "I": "I",
            "MY": "my",
            "YOU": "you",
            "YOUR": "your",
            "HE": "he",
            "HIS": "his",
            "SHE": "she",
            "HER": "her",
            "WE": "we",
            "OUR": "our",
            "THEY": "they",
            "THEIR": "their"
        }

        verb_be_map = {
            "AM": "am",
            "IS": "is",
            "ARE": "are",
            "WAS": "was",
            "WERE": "were"
        }

        words = []
        for i, g in enumerate(clean_glosses):
            word = pronoun_map.get(g, verb_be_map.get(g, g.lower()))
            words.append(word)

        # Join words with proper spacing
        sentence = " ".join(words)

        # Apply basic capitalization & period
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith((".", "?", "!")):
                sentence += "."

        return sentence

    def translate_glosses(self, glosses: List[str]) -> Dict[str, Any]:
        """
        Translate a list of ASL glosses into a natural English sentence.
        Returns dict with 'english', 'latency_ms', 'status', and 'error'.
        """
        if not glosses:
            return {"english": "", "latency_ms": 0.0, "status": "empty_input", "error": None}

        gloss_text = " ".join([g.strip().upper() for g in glosses if g.strip()])
        if not gloss_text:
            return {"english": "", "latency_ms": 0.0, "status": "empty_input", "error": None}

        start_time = time.time()

        # Try local cached PyTorch model if available, otherwise fast local rule engine
        if not self._is_initialized and self.model_name != "none":
            self.load_model()

        if self.model_name not in ("Qwen/Qwen2.5-1.5B-Instruct", "none") and not self._is_initialized:
            return {
                "english": "",
                "latency_ms": 0.0,
                "status": "unavailable",
                "error": self._init_error or f"Model {self.model_name} is unavailable"
            }

        if self._is_initialized and self.model is not None:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"ASL Glosses: {gloss_text}"}
                ]
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=48,
                        temperature=0.1,
                        do_sample=False
                    )

                gen_tokens = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                raw_response = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
                clean_sentence = re.sub(r'[*_#`]', '', raw_response).strip()
                if clean_sentence:
                    latency_ms = round((time.time() - start_time) * 1000.0, 2)
                    return {
                        "english": clean_sentence,
                        "latency_ms": latency_ms,
                        "status": "success",
                        "engine": f"Local PyTorch ({self.model_name})",
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"PyTorch inference failed, falling back to local NLP rules: {e}")

        # Fast local rule engine fallback
        english_sentence = self._rule_based_translation(glosses)
        latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "english": english_sentence,
            "latency_ms": latency_ms,
            "status": "success",
            "engine": "Fast Local Rule-based NLP",
            "error": None
        }
