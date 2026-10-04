import os
import time
import re
import logging
from typing import Dict, Any, Optional, List

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

def _load_env_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip("\"' ")
    return ""

class GroqLanguageService:
    """
    Groq AI Language Service for SignBridge.
    Translates ASL Glosses into natural, fluent English sentences using ultra-fast Groq LLMs.
    Falls back gracefully to fast local rule-based NLP if GROQ_API_KEY is not configured or offline.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen/qwen3.8-27b"):
        self.api_key = api_key or _load_env_key()
        self.model_name = model_name
        self.client = None
        self._init_error = None

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                self._init_error = f"Failed to initialize Groq client: {str(e)}"
                logger.warning(self._init_error)
        elif not GROQ_AVAILABLE:
            self._init_error = "groq Python SDK not installed"

        self.system_prompt = (
            "You are an ASL gloss to natural English translator.\n"
            "Convert the input ASL glosses into a single fluent, natural English sentence.\n"
            "Rules:\n"
            "- Preserve the exact meaning of the ASL glosses.\n"
            "- Correct grammar, verb tenses, and word order.\n"
            "- Do NOT add facts, names, or emotions not present in the glosses.\n"
            "- Return ONLY the final plain English sentence without any explanation or markdown formatting."
        )

        # High-frequency fallback dictionary for local instant rules
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

    def is_available(self) -> bool:
        return GROQ_AVAILABLE and bool(self.api_key) and self.client is not None

    def _rule_based_translation(self, glosses: List[str]) -> str:
        """Fast local rule-based ASL gloss to English translator fallback."""
        clean_glosses = [g.strip().upper() for g in glosses if g.strip()]
        if not clean_glosses:
            return ""

        full_key = " ".join(clean_glosses)
        if full_key in self.gloss_phrases:
            return self.gloss_phrases[full_key]

        pronoun_map = {
            "ME": "I", "I": "I", "MY": "my", "YOU": "you", "YOUR": "your",
            "HE": "he", "HIS": "his", "SHE": "she", "HER": "her",
            "WE": "we", "OUR": "our", "THEY": "they", "THEIR": "their"
        }

        verb_be_map = {
            "AM": "am", "IS": "is", "ARE": "are", "WAS": "was", "WERE": "were"
        }

        words = []
        for g in clean_glosses:
            word = pronoun_map.get(g, verb_be_map.get(g, g.lower()))
            words.append(word)

        sentence = " ".join(words)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith((".", "?", "!")):
                sentence += "."
        return sentence

    def translate_glosses(self, glosses: List[str]) -> Dict[str, Any]:
        """
        Translate a list of ASL glosses into a natural English sentence via Groq API,
        or fall back to local rule-based NLP if key is missing/invalid.
        """
        if not glosses:
            return {"english": "", "latency_ms": 0.0, "status": "empty_input", "error": None}

        gloss_text = " ".join([g.strip().upper() for g in glosses if g.strip()])
        if not gloss_text:
            return {"english": "", "latency_ms": 0.0, "status": "empty_input", "error": None}

        start_time = time.time()

        # Call Groq API if available and key present
        if self.is_available():
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"ASL Glosses: {gloss_text}"}
                    ],
                    model=self.model_name,
                    temperature=0.1,
                    max_tokens=64
                )
                raw_response = chat_completion.choices[0].message.content.strip()
                clean_sentence = re.sub(r'[*_#`]', '', raw_response).strip()
                if (clean_sentence.startswith('"') and clean_sentence.endswith('"')) or (clean_sentence.startswith("'") and clean_sentence.endswith("'")):
                    clean_sentence = clean_sentence[1:-1].strip()

                if clean_sentence:
                    latency_ms = round((time.time() - start_time) * 1000.0, 2)
                    return {
                        "english": clean_sentence,
                        "latency_ms": latency_ms,
                        "status": "success",
                        "engine": f"Groq AI ({self.model_name})",
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"Groq API call failed: {e}. Falling back to local NLP rules.")

        # Fallback to local rule engine if Groq API key is missing or call failed
        english_sentence = self._rule_based_translation(glosses)
        latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "english": english_sentence,
            "latency_ms": latency_ms,
            "status": "success",
            "engine": "Fast Local Rule-based NLP (Groq key unconfigured or fallback)",
            "error": None
        }
