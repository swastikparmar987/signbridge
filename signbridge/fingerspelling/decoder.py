import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

class FingerspellingDecoder:
    """
    Decodes frame-level Squeezeformer/CTC logits into readable English characters and words.
    Supports CTC greedy search, confidence estimation, and prediction stability analysis.
    """

    def __init__(self, vocab_path: Optional[Union[str, Path]] = None, blank_idx: int = 59):
        if vocab_path is None:
            vocab_path = Path(__file__).resolve().parent / "character_to_prediction_index.json"
        
        self.vocab_path = Path(vocab_path)
        with open(self.vocab_path, "r") as f:
            self.char_to_index: Dict[str, int] = json.load(f)

        self.blank_idx = blank_idx
        self.index_to_char: Dict[int, str] = {idx: char for char, idx in self.char_to_index.items()}
        self.index_to_char[self.blank_idx] = ""  # CTC blank token

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax along the last axis."""
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shift)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def decode_greedy(
        self,
        logits: np.ndarray,
        min_char_prob: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Perform Connectionist Temporal Classification (CTC) greedy decoding.
        
        Args:
            logits: Array of shape (L, 60) or (1, L, 60) representing frame logits.
            min_char_prob: Minimum confidence threshold for emitting a character.
            
        Returns:
            Dictionary containing:
                - text: The decoded character string.
                - characters: List of decoded character details with timestamp & confidence.
                - confidence: Mean confidence of emitted characters (0.0 to 1.0).
                - stability: Stability metric (0.0 to 1.0) based on confidence margin.
        """
        if logits.ndim == 3:
            logits = logits[0]  # Take first batch item

        # ishara_v1.tflite outputs already-normalized softmax/one-hot probabilities;
        # avoid applying softmax a second time if rows already sum to ~1 in [0, 1].
        if logits.size > 0 and np.all(logits >= -1e-4) and np.all(logits <= 1.0001) and np.allclose(logits.sum(axis=-1), 1.0, atol=1e-2):
            probs = logits
        else:
            probs = self.softmax(logits)  # (L, C)
        best_indices = np.argmax(probs, axis=-1)
        best_probs = np.max(probs, axis=-1)

        # Margin between top 1 and top 2 for stability evaluation
        sorted_probs = np.sort(probs, axis=-1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        mean_margin = float(np.mean(margins)) if len(margins) > 0 else 0.0

        decoded_chars: List[Dict[str, Any]] = []
        raw_char_list: List[str] = []
        confidences: List[float] = []

        prev_idx = -1
        for t, (idx, prob) in enumerate(zip(best_indices, best_probs)):
            idx = int(idx)
            prob = float(prob)

            if idx != prev_idx:
                if idx != self.blank_idx and prob >= min_char_prob:
                    ch = self.index_to_char.get(idx, "")
                    if ch:
                        decoded_chars.append({
                            "char": ch,
                            "index": idx,
                            "prob": prob,
                            "frame": t,
                        })
                        raw_char_list.append(ch)
                        confidences.append(prob)
            prev_idx = idx

        assembled_text = "".join(raw_char_list)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        stability = float(np.clip(mean_margin * 1.5, 0.0, 1.0))

        return {
            "text": assembled_text,
            "characters": decoded_chars,
            "confidence": avg_confidence,
            "stability": stability,
            "frame_count": int(logits.shape[0]),
        }

    @staticmethod
    def calculate_levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return FingerspellingDecoder.calculate_levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @staticmethod
    def normalized_levenshtein(ground_truth: str, prediction: str) -> float:
        """
        Calculate normalized Levenshtein score (official Kaggle metric):
        (N - D) / N where N is total GT length and D is edit distance.
        """
        N = len(ground_truth)
        if N == 0:
            return 1.0 if len(prediction) == 0 else 0.0
        D = FingerspellingDecoder.calculate_levenshtein(ground_truth, prediction)
        return max(0.0, (N - D) / N)

