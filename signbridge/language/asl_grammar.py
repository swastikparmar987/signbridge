import re
from typing import Any, Dict, List, Optional


_TIME_WORDS = {
    "YESTERDAY": "Yesterday",
    "TODAY": "Today",
    "TOMORROW": "Tomorrow",
    "NOW": "Now",
    "LATER": "Later",
    "MORNING": "This morning",
    "AFTERNOON": "This afternoon",
    "TONIGHT": "Tonight",
    "NIGHT": "Tonight",
    "BEFORE": "Before",
    "RECENTLY": "Recently",
    "ALWAYS": "Always",
    "SOMETIMES": "Sometimes",
    "NEVER": "Never",
}

_WH_WORDS = {
    "WHAT": "what",
    "WHO": "who",
    "WHERE": "where",
    "WHEN": "when",
    "WHY": "why",
    "HOW": "how",
    "WHICH": "which",
    "HOWMANY": "how many",
    "HOW-MANY": "how many",
}

_PRONOUN_MAP = {
    "IX-1P": "I",
    "ME": "I",
    "I": "I",
    "MY": "my",
    "POSS-1P": "my",
    "MINE": "mine",
    "IX-2P": "you",
    "YOU": "you",
    "YOUR": "your",
    "POSS-2P": "your",
    "IX-3P": "they",
    "HE": "he",
    "HIM": "him",
    "HIS": "his",
    "SHE": "she",
    "HER": "her",
    "WE": "we",
    "US": "us",
    "OUR": "our",
    "THEY": "they",
    "THEM": "them",
    "THEIR": "their",
    "SELF-1P": "myself",
    "SELF-2P": "yourself",
    "SELF-3P": "themselves",
}

_MODALS = {
    "FUTURE": "will",
    "WILL": "will",
    "SHOULD": "should",
    "MUST": "must",
    "NEED": "need to",
    "CAN": "can",
    "CANT": "cannot",
    "CANNOT": "cannot",
    "COULD": "could",
    "MAY": "may",
    "MIGHT": "might",
}

_ASPECT_PAST = {"FINISH", "PAST", "ALREADY"}
_PROPER_NAMES = {"JOHN", "MARY", "BILL", "SUE", "TOM", "BOB", "ALICE", "SARAH", "MIKE", "DAVID", "LISA"}
_INFINITIVE_TAKING_VERBS = {"WANT", "NEED", "TRY", "PLAN", "LIKE", "LOVE", "HATE", "HOPE", "DECIDE"}

_COMMON_IRREGULAR_PAST = {
    "BUY": "bought",
    "GO": "went",
    "COME": "came",
    "SEE": "saw",
    "EAT": "ate",
    "DRINK": "drank",
    "READ": "read",
    "WRITE": "wrote",
    "MAKE": "made",
    "TAKE": "took",
    "GIVE": "gave",
    "TELL": "told",
    "THINK": "thought",
    "MEET": "met",
    "LEAVE": "left",
    "FIND": "found",
    "WIN": "won",
    "DO": "did",
    "HAVE": "had",
}

_GERUND_VERBS = {
    "READ": "reading",
    "WRITE": "writing",
    "COOK": "cooking",
    "CLEAN": "cleaning",
    "DRIVE": "driving",
    "EAT": "eating",
    "DRINK": "drinking",
    "STUDY": "studying",
    "WORK": "working",
    "SLEEP": "sleeping",
    "WALK": "walking",
    "RUN": "running",
    "SWIM": "swimming",
    "LEARN": "learning",
    "SIGN": "signing",
    "TALK": "talking",
    "WATCH": "watching",
    "PLAY": "playing",
}

_NOUNS_NEEDING_ARTICLE = {
    "BOOK", "CAR", "HOUSE", "DOG", "CAT", "APPLE", "LETTER", "COFFEE",
    "DOCTOR", "TEACHER", "STUDENT", "FRIEND", "MOVIE", "GAME", "STORE",
    "SCHOOL", "HOSPITAL", "JOB", "CLASS", "PHONE", "COMPUTER", "CHAIR",
    "TABLE", "KEY", "BAG", "CUP", "SHIRT", "BUS", "TRAIN", "AIRPLANE",
}


def clean_gloss_token(raw: str) -> str:
    tok = str(raw).strip().upper()
    if not tok or tok in ("HOLD", "NO_SIGN", "REST", "<BLANK>"):
        return ""
    if tok.startswith("FS-"):
        tok = tok[3:]
    if tok.startswith("NS-"):
        tok = tok[3:]
    # Strip NCSLGR hand-count / variant tags like (1H), (2H), (Y)
    tok = re.sub(r"^\([A-Z0-9]+\)", "", tok)
    tok = re.sub(r"^[A-Z]:", "", tok)
    tok = re.sub(r"[+^#_/]+$", "", tok)
    if tok.startswith("IX-1P"):
        return "IX-1P"
    if tok.startswith("IX-2P"):
        return "IX-2P"
    if tok.startswith("IX-3P"):
        return "IX-3P"
    if tok.startswith("POSS-1P"):
        return "POSS-1P"
    if tok.startswith("POSS-2P"):
        return "POSS-2P"
    if tok.startswith("PART:"):
        return ""
    return tok.strip()


def normalize_asl_tokens(glosses: List[str]) -> List[str]:
    out: List[str] = []
    for raw in glosses:
        tok = clean_gloss_token(raw)
        if not tok:
            continue
        # Collapse immediate duplicate repetitions during raw signing
        if out and out[-1] == tok:
            continue
        out.append(tok)
    return out


def asl_rules_to_english(
    glosses: List[str],
    non_manual: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Deterministic ASL-to-English syntactic converter handling Time-First ordering,
    aspect markers (FINISH/FUTURE), modals, negation (NOT/NOT-WANT/NEVER),
    WH-final questions, and copula insertion when LLMs are unavailable.
    """
    tokens = normalize_asl_tokens(glosses)
    if not tokens:
        return ""

    non_manual = non_manual or {}
    is_wh = bool(non_manual.get("wh_question", False))
    is_yn = bool(non_manual.get("yn_question", False))
    is_neg = bool(non_manual.get("negative", False))

    # Extract sentence-initial time adverbials and clause-final WH words
    time_prefix: List[str] = []
    wh_words: List[str] = []
    core: List[str] = []

    for i, tok in enumerate(tokens):
        if i == 0 and tok in _TIME_WORDS and len(tokens) > 1:
            time_prefix.append(_TIME_WORDS[tok])
            continue
        if tok in _WH_WORDS:
            wh_words.append(_WH_WORDS[tok])
            is_wh = True
            continue
        if tok in ("NOT", "NEVER", "NO", "NONE", "NOTHING", "NONENOTHING"):
            is_neg = True
        core.append(tok)

    # Phrase shortcuts for ultra-common ASL idioms
    joined = " ".join(tokens)
    idioms = {
        "THANK YOU": "Thank you.",
        "THANK-YOU": "Thank you.",
        "YOU WELCOME": "You are welcome.",
        "HOW YOU": "How are you?",
        "YOU NAME WHAT": "What is your name?",
        "MY NAME": "My name is...",
        "NICE MEET YOU": "Nice to meet you.",
        "GOOD MORNING": "Good morning.",
        "GOOD NIGHT": "Good night.",
        "I LOVE YOU": "I love you.",
        "SORRY": "I am sorry.",
        "PLEASE": "Please.",
        "EXCUSE ME": "Excuse me.",
    }
    if joined in idioms:
        return idioms[joined]

    if wh_words == ["what"] and set(core) <= {"YOU", "YOUR", "NAME", "POSS-2P", "IX-2P"} and "NAME" in core:
        return "What is your name?"

    # Expand compound negative tokens like NOT-WANT
    expanded: List[str] = []
    for tok in core:
        if tok == "NOT-WANT":
            expanded.extend(["DO_NOT", "WANT"])
            is_neg = True
        elif tok == "NOT-LIKE":
            expanded.extend(["DO_NOT", "LIKE"])
            is_neg = True
        elif tok == "NOT-KNOW":
            expanded.extend(["DO_NOT", "KNOW"])
            is_neg = True
        else:
            expanded.append(tok)
    core = expanded

    # Detect PAST/FINISH and FUTURE/modal auxiliary placement
    words: List[str] = []
    modal_found: Optional[str] = None
    has_finish = False
    i = 0
    while i < len(core):
        tok = core[i]
        nxt = core[i + 1] if i + 1 < len(core) else None

        if tok in _MODALS:
            modal_word = _MODALS[tok]
            if not modal_found:
                modal_found = modal_word
            words.append(modal_word)
            i += 1
            continue

        if tok == "DO_NOT":
            words.append("does not" if words and words[0].lower() in ("mother", "father", "he", "she") else "do not")
            i += 1
            continue

        if tok == "NOT":
            if modal_found or (words and words[-1] in ("will", "should", "must", "can", "could", "may", "might")):
                words.append("not")
            else:
                subj = words[0].lower() if words else ""
                words.append("does not" if subj in ("mother", "father", "he", "she", "john", "mary") else "do not")
            i += 1
            continue

        if tok in _ASPECT_PAST:
            has_finish = True
            if nxt and nxt not in ("NOT", "FINISH"):
                gerund = _GERUND_VERBS.get(nxt)
                if gerund and not is_wh and not modal_found:
                    words.append("finished")
                    words.append(gerund)
                    i += 2
                    continue
                elif nxt in _COMMON_IRREGULAR_PAST and modal_found:
                    words.append("finish")
                    words.append(gerund or nxt.lower())
                    i += 2
                    continue
                elif nxt in _COMMON_IRREGULAR_PAST:
                    words.append(_COMMON_IRREGULAR_PAST[nxt])
                    i += 2
                    continue
            words.append("finished")
            i += 1
            continue

        mapped = _PRONOUN_MAP.get(tok)
        if mapped is not None:
            if tok in ("IX-2P", "YOU") and nxt == "NAME":
                words.append("your")
            elif tok == "IX-3P" and nxt and nxt not in _MODALS and nxt not in _ASPECT_PAST and nxt not in ("NOT", "FINISH"):
                words.append("the")
            elif mapped == "you" and not words and wh_words:
                words.append("you")
            else:
                words.append(mapped)
            i += 1
            continue

        low = tok.lower().replace("-", " ")
        if tok in _PROPER_NAMES:
            low = low.capitalize()
        elif i == len(core) - 1 and tok in _NOUNS_NEEDING_ARTICLE and i > 0 and core[i - 1] not in ("MY", "YOUR", "THIS", "THAT", "ONE", "TWO", "THREE", "FOUR", "FIVE", "MANY"):
            words.append("the" if has_finish else "a")
        if i > 0 and core[i - 1] in _INFINITIVE_TAKING_VERBS and tok not in _NOUNS_NEEDING_ARTICLE and tok not in _PRONOUN_MAP:
            words.append("to")
        words.append(low)
        i += 1

    # Handle ASL clause-final negation (e.g., COFFEE I LIKE NOT -> I do not like coffee)
    if len(words) >= 3 and words[-1] in ("do not", "does not") and not wh_words:
        neg_aux = words.pop()
        pron_idx = next(
            (idx for idx, w in enumerate(words) if w in ("I", "you", "he", "she", "we", "they", "mother", "father")),
            None,
        )
        if pron_idx is not None and pron_idx + 1 < len(words):
            subj = words[pron_idx]
            verb = words[pron_idx + 1 :]
            topic = words[:pron_idx]
            neg_form = "does not" if subj in ("mother", "father", "he", "she") else "do not"
            words = [subj, neg_form] + verb + topic
        elif len(words) >= 2:
            words = [words[0], neg_aux] + words[1:]

    # WH-word fronting with auxiliary inversion when present
    if wh_words:
        wh = wh_words[0]
        aux_idx = next((idx for idx, w in enumerate(words) if w in ("will", "should", "can", "could", "must", "do", "does", "did", "is", "are", "was", "were")), None)
        if aux_idx is not None:
            aux = words.pop(aux_idx)
            body = [wh, aux] + words
        else:
            # Simple present/copula insertion for subject + object (e.g. YOUR NAME WHAT -> What is your name?)
            if len(words) >= 2 and words[-1] in ("name", "car", "house", "job"):
                body = [wh, "is"] + words
            else:
                body = [wh] + words
    elif time_prefix:
        body = time_prefix + words
    else:
        body = words

    # Inject copula for bare Pronoun + Adj/Noun (e.g., I HAPPY -> I am happy)
    if not wh_words and len(body) >= 2 and body[0] in ("I", "you", "he", "she", "we", "they") and body[1] in ("happy", "sad", "tired", "hungry", "thirsty", "sick", "fine", "good", "bad", "ready", "student", "teacher", "doctor"):
        cop = "am" if body[0] == "I" else ("is" if body[0] in ("he", "she") else "are")
        body = [body[0], cop] + body[1:]

    sentence = " ".join(body).strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    if is_wh or is_yn:
        if not sentence.endswith("?"):
            sentence = sentence.rstrip(".!") + "?"
    elif not sentence.endswith((".", "?", "!")):
        sentence += "."
    return sentence


def build_translation_prompt(
    glosses: List[str],
    hypotheses: Optional[List[List[Dict[str, Any]]]] = None,
    non_manual: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a concise, context-rich user prompt for Groq/Qwen containing ordered ASL
    glosses, inline Top-K ambiguous candidates per slot, and non-manual cues.
    """
    clean = [str(g).strip().upper() for g in glosses if str(g).strip()]
    display_tokens: List[str] = []
    has_ambig = False

    for idx, tok in enumerate(clean):
        slot = hypotheses[idx] if (hypotheses and idx < len(hypotheses)) else None
        if isinstance(slot, list) and len(slot) >= 2:
            normed: List[tuple] = []
            for c in slot[:3]:
                cg = str(c.get("gloss", "")).strip().upper()
                if not cg:
                    continue
                cp = float(c.get("confidence", 0.0))
                if cp > 1.0:
                    cp /= 100.0
                normed.append((cg, cp))
            if len(normed) >= 2 and (normed[0][1] < 0.72 or (normed[0][1] - normed[1][1]) < 0.25):
                has_ambig = True
                joined_alts = "|".join(f"{g}({p:.2f})" for g, p in normed)
                display_tokens.append(f"[{joined_alts}]")
                continue
        display_tokens.append(tok)

    lines = [f"ASL Gloss sequence: {' '.join(display_tokens)}"]
    if has_ambig:
        lines.append(
            "Note: Bracketed [A|B|C] slots have uncertain vision scores; select the option that makes natural semantic and grammatical sense."
        )

    if non_manual:
        flags = [k for k, v in non_manual.items() if v]
        if flags:
            lines.append(f"Non-manual markers: {', '.join(flags)}")

    lines.append("Translate into one natural, fluent English sentence. Output ONLY the final sentence.")
    return "\n".join(lines)
