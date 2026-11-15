#!/usr/bin/env python3
"""
Continuous CSLR, OOD Static Rejection, and ASL Grammar Benchmark Evaluator.

Evaluates single checkpoints or EnsemblePredictor configs on held-out stitched
test.csv continuous streams (with realistic coarticulation transitions, micro-pauses,
and trailing rest auto-finalization), OOD static/jitter streams, and ASL NLP grammar.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.continuous.session import ContinuousCSLRSessionManager
from signbridge.inference.predictor import EnsemblePredictor, SignBridgePredictor
from signbridge.language.asl_grammar import asl_rules_to_english
from signbridge.preprocessing.landmarks import compute_sequence_activity


def levenshtein_sdi(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) aligning hyp to ref."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, subs, dels, ins)
    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = (dp[i - 1][j - 1][0] + 1, dp[i - 1][j - 1][1] + 1, dp[i - 1][j - 1][2], dp[i - 1][j - 1][3])
                dele = (dp[i - 1][j][0] + 1, dp[i - 1][j][1], dp[i - 1][j][2] + 1, dp[i - 1][j][3])
                ins = (dp[i][j - 1][0] + 1, dp[i][j - 1][1], dp[i][j - 1][2], dp[i][j - 1][3] + 1)
                dp[i][j] = min(sub, dele, ins, key=lambda x: x[0])

    _, s, d, ins = dp[n][m]
    return s, d, ins


def load_ensemble_from_json(config_path: Path, device: Optional[torch.device] = None) -> EnsemblePredictor:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    members = []
    temps = []
    for member in cfg.get("members", []):
        ckpt_path = Path(member["checkpoint"])
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        weight = float(member.get("weight", 1.0))
        if not ckpt_path.exists() or weight <= 0.0:
            continue
        pred = SignBridgePredictor(
            checkpoint_path=ckpt_path,
            device=device,
            ood_threshold=cfg.get("ood_threshold"),
        )
        members.append((pred, weight))
        temps.append(float(member.get("temperature", 1.0)))
    if not members:
        raise RuntimeError(f"No valid members found in {config_path}")
    return EnsemblePredictor(
        predictors=members,
        temperatures=temps,
        ood_threshold=cfg.get("ood_threshold"),
    )


def select_active_test_indices(test_x: np.ndarray, min_active: int = 12, min_motion: float = 0.08) -> np.ndarray:
    active_ids = []
    for i in range(len(test_x)):
        act = compute_sequence_activity(test_x[i])
        if act["active_frames"] >= min_active and act["motion_score"] >= min_motion:
            active_ids.append(i)
    return np.asarray(active_ids, dtype=np.int64)


def stitch_continuous_stream(
    clips: List[np.ndarray],
    transition_frames: int = 6,
    inter_rest_frames: int = 5,
    trailing_rest_frames: int = 42,
) -> np.ndarray:
    """Stitch isolated (32, 42, 3) clips into a realistic continuous stream."""
    chunks: List[np.ndarray] = []
    # Brief initial settle so segmenter sees neutral onset
    chunks.append(np.repeat(clips[0][:1], 3, axis=0))

    for idx, clip in enumerate(clips):
        chunks.append(clip)
        if idx < len(clips) - 1:
            end_pose = clip[-1:]
            next_pose = clips[idx + 1][:1]
            if inter_rest_frames > 0:
                mid_pose = 0.5 * (end_pose + next_pose)
                a1 = np.linspace(0.0, 1.0, max(2, transition_frames // 2 + 1), dtype=np.float32)[1:]
                chunks.append((1.0 - a1[:, None, None]) * end_pose + a1[:, None, None] * mid_pose)
                chunks.append(np.repeat(mid_pose, inter_rest_frames, axis=0))
                a2 = np.linspace(0.0, 1.0, max(2, transition_frames // 2 + 1), dtype=np.float32)[:-1]
                chunks.append((1.0 - a2[:, None, None]) * mid_pose + a2[:, None, None] * next_pose)
            elif transition_frames > 0:
                alphas = np.linspace(0.0, 1.0, transition_frames + 2, dtype=np.float32)[1:-1]
                trans = (1.0 - alphas[:, None, None]) * end_pose + alphas[:, None, None] * next_pose
                chunks.append(trans)

    # Trailing pause: slowly drift / drop hands to trigger pause auto-finalization cleanly
    tail_pose = clips[-1][-1:]
    chunks.append(np.repeat(tail_pose, max(4, trailing_rest_frames // 2), axis=0))
    chunks.append(np.zeros((max(4, trailing_rest_frames - trailing_rest_frames // 2), 42, 3), dtype=np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def evaluate_streams(
    name: str,
    predictor: Any,
    streams: List[Tuple[List[str], np.ndarray]],
    batch_size: int = 5,
    fps: float = 30.0,
) -> Dict[str, Any]:
    mgr = ContinuousCSLRSessionManager()
    dt_ms = 1000.0 / fps

    window_latencies: List[float] = []

    def _predict_fn(seq32: np.ndarray, k: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        out = predictor.predict_sequence(seq32, is_normalized=False, top_k=k)
        window_latencies.append((time.perf_counter() - t0) * 1000.0)
        return out

    total_ref = 0
    total_s = 0
    total_d = 0
    total_i = 0
    exact_sentences = 0
    auto_finalized = 0
    emitted_tokens = 0
    top5_hit_slots = 0

    t_start = time.perf_counter()
    for s_idx, (ref_glosses, frames) in enumerate(streams):
        sid = f"{name}_{s_idx}"
        mgr.reset(sid)
        hyp_glosses: Optional[List[str]] = None
        hyp_top_k_sets: List[set] = []

        t_ms = 0.0
        for pos in range(0, len(frames), batch_size):
            chunk = frames[pos : pos + batch_size]
            ts = [t_ms + k * dt_ms for k in range(len(chunk))]
            t_ms += len(chunk) * dt_ms
            state = mgr.push_frames(
                session_id=sid,
                frames=chunk,
                predict_fn=_predict_fn,
                timestamps_ms=ts,
                auto_finalize_on_pause=True,
                top_k=5,
            )
            if state.get("pause_finalized") and state.get("finalized_utterance"):
                fin = state["finalized_utterance"]
                hyp_glosses = [g.upper() for g in fin.get("glosses", [])]
                hyp_top_k_sets = [
                    {str(c.get("gloss", "")).upper() for c in slot}
                    for slot in fin.get("hypotheses", [])
                ]
                auto_finalized += 1
                break

        if hyp_glosses is None:
            fin = mgr.finalize(sid, predict_fn=_predict_fn, clear_after=True)
            hyp_glosses = [g.upper() for g in fin.get("glosses", [])]
            hyp_top_k_sets = [
                {str(c.get("gloss", "")).upper() for c in slot}
                for slot in fin.get("hypotheses", [])
            ]

        ref_upper = [g.upper() for g in ref_glosses]
        s, d, ins = levenshtein_sdi(ref_upper, hyp_glosses)
        total_ref += len(ref_upper)
        total_s += s
        total_d += d
        total_i += ins
        emitted_tokens += len(hyp_glosses)
        if ref_upper == hyp_glosses:
            exact_sentences += 1
        # Slot-wise Top-5 coverage in hypotheses
        for r_tok in ref_upper:
            if any(r_tok in slot_set for slot_set in hyp_top_k_sets):
                top5_hit_slots += 1

    elapsed_s = time.perf_counter() - t_start
    wer = (total_s + total_d + total_i) / max(1, total_ref)
    return {
        "streams": len(streams),
        "total_ref_tokens": total_ref,
        "total_emitted_tokens": emitted_tokens,
        "wer_pct": round(wer * 100.0, 2),
        "sub_pct": round((total_s / max(1, total_ref)) * 100.0, 2),
        "del_pct": round((total_d / max(1, total_ref)) * 100.0, 2),
        "ins_pct": round((total_i / max(1, total_ref)) * 100.0, 2),
        "top5_slot_recall_pct": round((top5_hit_slots / max(1, total_ref)) * 100.0, 2),
        "exact_sentence_pct": round((exact_sentences / max(1, len(streams))) * 100.0, 2),
        "auto_finalized_pct": round((auto_finalized / max(1, len(streams))) * 100.0, 2),
        "mean_window_latency_ms": round(float(np.mean(window_latencies)) if window_latencies else 0.0, 2),
        "p95_window_latency_ms": round(float(np.percentile(window_latencies, 95)) if window_latencies else 0.0, 2),
        "wall_sec": round(elapsed_s, 2),
    }


def evaluate_ood_streams(
    predictor: Any,
    test_x: np.ndarray,
    active_ids: np.ndarray,
    num_ood: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    mgr = ContinuousCSLRSessionManager()
    dt_ms = 1000.0 / 30.0
    false_trigger_streams = 0
    total_spurious_tokens = 0

    def _predict_fn(seq32: np.ndarray, k: int) -> Dict[str, Any]:
        return predictor.predict_sequence(seq32, is_normalized=False, top_k=k)

    chosen = rng.choice(active_ids, size=min(num_ood, len(active_ids)), replace=False)
    for idx, sample_id in enumerate(chosen):
        kind = idx % 3
        base = test_x[sample_id]
        # Pick an active mid-frame to hold statically
        valid = np.where(np.abs(base).sum(axis=(1, 2)) > 1e-4)[0]
        anchor = base[valid[len(valid) // 2]] if len(valid) > 0 else base[15]

        if kind == 0:
            # Pure static freeze (48 frames = 1.6s)
            stream = np.repeat(anchor[None, :, :], 48, axis=0).astype(np.float32)
        elif kind == 1:
            # Static hold + tiny webcam tracker micro-jitter (std=0.002)
            noise = rng.normal(0.0, 0.002, size=(48, 42, 3)).astype(np.float32)
            mask = (np.abs(anchor) > 1e-4)[None, :, :]
            stream = np.where(mask, anchor[None, :, :] + noise, 0.0).astype(np.float32)
        else:
            # All-zero / hands dropped out of frame
            stream = np.zeros((48, 42, 3), dtype=np.float32)

        sid = f"ood_{idx}"
        mgr.reset(sid)
        emitted: List[str] = []
        t_ms = 0.0
        for pos in range(0, len(stream), 6):
            chunk = stream[pos : pos + 6]
            ts = [t_ms + k * dt_ms for k in range(len(chunk))]
            t_ms += len(chunk) * dt_ms
            state = mgr.push_frames(sid, chunk, predict_fn=_predict_fn, timestamps_ms=ts, top_k=5)
            emitted.extend(tok["gloss"] for tok in state.get("newly_committed", []))
            if state.get("pause_finalized") and state.get("finalized_utterance"):
                emitted = state["finalized_utterance"].get("glosses", [])

        if not emitted:
            fin = mgr.finalize(sid, predict_fn=_predict_fn, clear_after=True)
            emitted = fin.get("glosses", [])

        if emitted:
            false_trigger_streams += 1
            total_spurious_tokens += len(emitted)

    total = len(chosen)
    return {
        "ood_streams": total,
        "false_trigger_streams": false_trigger_streams,
        "rejection_rate_pct": round((1.0 - false_trigger_streams / max(1, total)) * 100.0, 2),
        "total_spurious_tokens": total_spurious_tokens,
    }


def evaluate_asl_grammar_rules() -> Dict[str, Any]:
    cases = [
        (["TOMORROW", "I", "GO", "SCHOOL"], None, {"should_contain": ["tomorrow", "go", "school"]}),
        (["BOOK", "YOU", "WANT", "WHAT"], {"wh_question": True}, {"endswith": "?"}),
        (["YESTERDAY", "FATHER", "BUY", "CAR", "FINISH"], None, {"should_contain": ["yesterday", "car"]}),
        (["COFFEE", "I", "LIKE", "NOT"], {"negative": True}, {"should_contain": ["do not", "coffee", "like"]}),
    ]
    passed = 0
    samples = []
    for glosses, nmm, req in cases:
        eng = asl_rules_to_english(glosses, non_manual=nmm)
        ok = bool(eng.strip())
        if "endswith" in req:
            ok = ok and eng.strip().endswith(req["endswith"])
        for word in req.get("should_contain", []):
            ok = ok and (word.lower() in eng.lower())
        if ok:
            passed += 1
        samples.append({"glosses": glosses, "english": eng, "passed": ok})

    ncslgr_path = PROJECT_ROOT / "data" / "ncslgr_annotations" / "ncslgr_sentences.jsonl"
    ncslgr_checked = 0
    if ncslgr_path.exists():
        lines = ncslgr_path.read_text(encoding="utf-8").splitlines()
        for raw_line in lines:
            row = json.loads(raw_line)
            if float(row.get("vocab_coverage", 0.0)) >= 0.75 and 3 <= len(row.get("glosses", [])) <= 8:
                out = asl_rules_to_english(row["glosses"], non_manual=row.get("non_manual"))
                if out.strip():
                    ncslgr_checked += 1

    return {
        "rule_checks_passed": passed,
        "rule_checks_total": len(cases),
        "ncslgr_clean_sentences_translated": ncslgr_checked,
        "samples": samples,
    }


def build_stream_fixtures(
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_names: List[str],
    active_ids: np.ndarray,
    num_streams: int,
    seed: int,
) -> Tuple[List[Tuple[List[str], np.ndarray]], List[Tuple[List[str], np.ndarray]]]:
    rng = np.random.default_rng(seed)
    paused_streams: List[Tuple[List[str], np.ndarray]] = []
    continuous_streams: List[Tuple[List[str], np.ndarray]] = []

    for _ in range(num_streams):
        n_signs = int(rng.integers(3, 6))  # 3 to 5 signs
        picked_ids = []
        picked_labels: List[int] = []
        attempts = 0
        while len(picked_ids) < n_signs and attempts < 40:
            attempts += 1
            cid = int(active_ids[rng.integers(0, len(active_ids))])
            lbl = int(test_y[cid])
            if not picked_labels or lbl != picked_labels[-1]:
                picked_ids.append(cid)
                picked_labels.append(lbl)

        clips = [test_x[cid] for cid in picked_ids]
        glosses = [class_names[lbl] for lbl in picked_labels]
        paused_streams.append(
            (glosses, stitch_continuous_stream(clips, transition_frames=6, inter_rest_frames=5, trailing_rest_frames=42))
        )
        continuous_streams.append(
            (glosses, stitch_continuous_stream(clips, transition_frames=6, inter_rest_frames=0, trailing_rest_frames=42))
        )
    return paused_streams, continuous_streams


def run_benchmark_for_predictor(
    label: str,
    predictor: Any,
    paused_streams: List[Tuple[List[str], np.ndarray]],
    flow_streams: List[Tuple[List[str], np.ndarray]],
    test_x: np.ndarray,
    active_ids: np.ndarray,
    num_ood: int,
    seed: int,
) -> Dict[str, Any]:
    print(f"\n=== Evaluating Continuous Suite: {label} ===", flush=True)
    res_paused = evaluate_streams(f"{label}_paused", predictor, paused_streams)
    print(
        f"  [Paused Coarticulation] WER={res_paused['wer_pct']:.2f}% "
        f"(S={res_paused['sub_pct']:.1f}% D={res_paused['del_pct']:.1f}% I={res_paused['ins_pct']:.1f}%) | "
        f"Top5SlotRecall={res_paused['top5_slot_recall_pct']:.2f}% | "
        f"AutoFin={res_paused['auto_finalized_pct']:.1f}% | "
        f"Lat={res_paused['mean_window_latency_ms']:.2f}ms (p95={res_paused['p95_window_latency_ms']:.2f}ms)",
        flush=True,
    )

    res_flow = evaluate_streams(f"{label}_flow", predictor, flow_streams)
    print(
        f"  [Zero-Pause Coarticulated Flow] WER={res_flow['wer_pct']:.2f}% "
        f"(S={res_flow['sub_pct']:.1f}% D={res_flow['del_pct']:.1f}% I={res_flow['ins_pct']:.1f}%) | "
        f"Top5SlotRecall={res_flow['top5_slot_recall_pct']:.2f}% | "
        f"Lat={res_flow['mean_window_latency_ms']:.2f}ms",
        flush=True,
    )

    res_ood = evaluate_ood_streams(predictor, test_x, active_ids, num_ood=num_ood, seed=seed)
    print(
        f"  [OOD Static/Rest Rejection] {res_ood['ood_streams'] - res_ood['false_trigger_streams']}/{res_ood['ood_streams']} rejected "
        f"({res_ood['rejection_rate_pct']:.2f}%), spurious_tokens={res_ood['total_spurious_tokens']}",
        flush=True,
    )
    return {
        "paused_coarticulation": res_paused,
        "zero_pause_flow": res_flow,
        "ood_static_rest": res_ood,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous CSLR + OOD + Grammar benchmark evaluator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Candidate single model checkpoint")
    parser.add_argument("--ensemble-config", type=str, default=None, help="Candidate ensemble_config.json")
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="Optional baseline checkpoint for side-by-side comparison")
    parser.add_argument("--num-streams", type=int, default=200, help="Number of 3-5 sign test.csv streams")
    parser.add_argument("--num-ood", type=int, default=100, help="Number of OOD static/rest streams")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if not args.checkpoint and not args.ensemble_config:
        parser.error("Provide at least one of --checkpoint or --ensemble-config")

    dev = None if args.device == "auto" else torch.device(args.device)

    packed_dir = PROJECT_ROOT / "landmark_cache_packed"
    test_x = np.load(packed_dir / "test_x.npy", mmap_mode="r")
    test_y = np.load(packed_dir / "test_y.npy", mmap_mode="r")
    class_names = json.loads((packed_dir / "class_names.json").read_text(encoding="utf-8"))

    active_ids = select_active_test_indices(test_x, min_active=12, min_motion=0.08)
    print(
        f"Loaded test cache: total={len(test_x)} clips, active_eligible={len(active_ids)} "
        f"({len(active_ids)/max(1, len(test_x))*100:.1f}%)"
    )

    paused_streams, flow_streams = build_stream_fixtures(
        test_x=test_x,
        test_y=test_y,
        class_names=class_names,
        active_ids=active_ids,
        num_streams=args.num_streams,
        seed=args.seed,
    )

    report: Dict[str, Any] = {
        "num_streams": args.num_streams,
        "num_ood": args.num_ood,
        "seed": args.seed,
        "active_eligible_test_clips": int(len(active_ids)),
        "total_test_clips": int(len(test_x)),
    }

    if args.baseline_ckpt:
        base_pred = SignBridgePredictor(checkpoint_path=Path(args.baseline_ckpt), device=dev)
        report["baseline"] = run_benchmark_for_predictor(
            "baseline",
            base_pred,
            paused_streams,
            flow_streams,
            test_x,
            active_ids,
            args.num_ood,
            args.seed,
        )

    if args.ensemble_config:
        cand_pred = load_ensemble_from_json(Path(args.ensemble_config), device=dev)
        cand_label = "ensemble"
    else:
        cand_pred = SignBridgePredictor(checkpoint_path=Path(args.checkpoint), device=dev)
        cand_label = "candidate"

    report["candidate"] = run_benchmark_for_predictor(
        cand_label,
        cand_pred,
        paused_streams,
        flow_streams,
        test_x,
        active_ids,
        args.num_ood,
        args.seed,
    )

    nlp_report = evaluate_asl_grammar_rules()
    report["asl_grammar"] = nlp_report
    print(
        f"\n[ASL Grammar Rules] passed={nlp_report['rule_checks_passed']}/{nlp_report['rule_checks_total']} | "
        f"NCSLGR clean (>=75% vocab, 3-8 signs) translated={nlp_report['ncslgr_clean_sentences_translated']}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report -> {out_path}")


if __name__ == "__main__":
    main()
