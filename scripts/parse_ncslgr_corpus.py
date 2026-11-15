import glob
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


def parse_eaf(path: Path) -> Dict:
    root = ET.parse(path).getroot()
    slots: Dict[str, int] = {}
    for ts in root.findall(".//TIME_SLOT"):
        sid = ts.attrib.get("TIME_SLOT_ID", "")
        val = ts.attrib.get("TIME_VALUE")
        if sid and val is not None:
            slots[sid] = int(val)

    ann_times: Dict[str, tuple] = {}
    tier_items: Dict[str, List[Dict]] = {}

    for tier in root.findall(".//TIER"):
        tid = tier.attrib.get("TIER_ID", "")
        items = []
        for ann in tier.findall(".//ANNOTATION"):
            align = ann.find("ALIGNABLE_ANNOTATION")
            ref = ann.find("REF_ANNOTATION")
            if align is not None:
                aid = align.attrib.get("ANNOTATION_ID", "")
                t0 = slots.get(align.attrib.get("TIME_SLOT_REF1", ""), 0)
                t1 = slots.get(align.attrib.get("TIME_SLOT_REF2", ""), 0)
                ann_times[aid] = (t0, t1)
                val_el = align.find("ANNOTATION_VALUE")
                txt = (val_el.text or "").strip() if val_el is not None else ""
                if txt:
                    items.append({"id": aid, "t0_ms": t0, "t1_ms": t1, "text": txt})
            elif ref is not None:
                aid = ref.attrib.get("ANNOTATION_ID", "")
                parent = ref.attrib.get("ANNOTATION_REF", "")
                t0, t1 = ann_times.get(parent, (0, 0))
                val_el = ref.find("ANNOTATION_VALUE")
                txt = (val_el.text or "").strip() if val_el is not None else ""
                if txt:
                    items.append({"id": aid, "t0_ms": t0, "t1_ms": t1, "text": txt})
        if items:
            tier_items[tid] = sorted(items, key=lambda x: (x["t0_ms"], x["t1_ms"]))

    return tier_items


def normalize_gloss_token(tok: str) -> str:
    clean = tok.strip().upper()
    if clean.startswith("FS-"):
        return clean[3:]
    clean = re.sub(r"[+^#_/()]+$", "", clean)
    clean = re.sub(r"\d+$", "", clean)
    return clean.strip()


def main() -> None:
    eaf_files = sorted(glob.glob("data/ncslgr_annotations/raw/elanBUcorpus/*.eaf"))
    out_path = Path("data/ncslgr_annotations/ncslgr_sentences.jsonl")
    vocab_path = Path("trained_models/production/class_names.json")
    vocab = set(json.loads(vocab_path.read_text())) if vocab_path.exists() else set()

    total_utts = 0
    clean_multi = 0
    matched_tokens = 0
    total_tokens = 0

    with open(out_path, "w") as out_f:
        for fp in eaf_files:
            tiers = parse_eaf(Path(fp))
            glosses = tiers.get("main gloss", [])
            engs = tiers.get("English translation", [])
            if not glosses or not engs:
                continue

            neg_spans = [(x["t0_ms"], x["t1_ms"]) for x in tiers.get("negative", [])]
            wh_spans = [(x["t0_ms"], x["t1_ms"]) for x in tiers.get("wh question", [])]
            yn_spans = [(x["t0_ms"], x["t1_ms"]) for x in tiers.get("yes-no question", [])]

            for e in engs:
                t0, t1 = e["t0_ms"], e["t1_ms"]
                if t1 > t0:
                    sent_gloss = [g for g in glosses if g["t1_ms"] >= t0 - 150 and g["t0_ms"] <= t1 + 150]
                elif len(engs) == 1:
                    sent_gloss = glosses
                else:
                    continue
                if not sent_gloss:
                    continue

                raw_tokens = [g["text"] for g in sent_gloss]
                norm_tokens = [normalize_gloss_token(t) for t in raw_tokens if normalize_gloss_token(t)]
                if not norm_tokens:
                    continue

                in_vocab = [t for t in norm_tokens if t in vocab]
                total_tokens += len(norm_tokens)
                matched_tokens += len(in_vocab)

                has_neg = any(a <= t1 and b >= t0 for a, b in neg_spans)
                has_wh = any(a <= t1 and b >= t0 for a, b in wh_spans)
                has_yn = any(a <= t1 and b >= t0 for a, b in yn_spans)

                rec = {
                    "source_file": Path(fp).name,
                    "t0_ms": t0,
                    "t1_ms": t1,
                    "raw_glosses": raw_tokens,
                    "glosses": norm_tokens,
                    "timings_ms": [[g["t0_ms"], g["t1_ms"]] for g in sent_gloss],
                    "english": e["text"],
                    "non_manual": {
                        "negative": has_neg,
                        "wh_question": has_wh,
                        "yn_question": has_yn,
                    },
                    "vocab_coverage": len(in_vocab) / len(norm_tokens),
                }
                out_f.write(json.dumps(rec) + "\n")
                total_utts += 1
                if len(norm_tokens) >= 3:
                    clean_multi += 1

    print(
        f"Parsed {len(eaf_files)} EAF files -> {total_utts} utterances "
        f"(>=3 signs: {clean_multi}) | Vocab match: {matched_tokens}/{total_tokens} "
        f"({100.0 * matched_tokens / max(1, total_tokens):.1f}%) -> {out_path}"
    )


if __name__ == "__main__":
    main()
