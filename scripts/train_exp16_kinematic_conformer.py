import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from signbridge.models.architectures import SignConformerClassifier, SignPyramidBiGRUClassifier
from signbridge.preprocessing.kinematics import KINEMATIC_V2_DIM, build_kinematic_v2_torch
from signbridge.preprocessing.landmarks import mirror_sequence_x, stabilize_hand_sequence
from signbridge.training.config import set_seed
from signbridge.training.gpu_augment import augment_landmark_batch, synthesize_ood_negative_batch


def set_subcenter_margin(model: torch.nn.Module, m_val: float) -> None:
    cls = model.classifier
    cls.m = float(m_val)
    cls.cos_m = math.cos(m_val)
    cls.sin_m = math.sin(m_val)
    cls.th = math.cos(math.pi - m_val)
    cls.mm = math.sin(math.pi - m_val) * m_val


def apply_feature_timemask(feat: torch.Tensor, p: float = 0.25, max_width: int = 4) -> torch.Tensor:
    if p <= 0.0:
        return feat
    B, T, _ = feat.shape
    apply_mask = torch.rand(B, device=feat.device) < p
    if not apply_mask.any():
        return feat
    out = feat.clone()
    widths = torch.randint(1, max_width + 1, (B,), device=feat.device)
    starts = torch.randint(0, max(1, T - max_width), (B,), device=feat.device)
    t_idx = torch.arange(T, device=feat.device).unsqueeze(0)
    span = (t_idx >= starts.unsqueeze(1)) & (t_idx < (starts + widths).unsqueeze(1))
    span = span & apply_mask.unsqueeze(1)
    return out.masked_fill(span.unsqueeze(-1), 0.0)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p_cur.data, alpha=1.0 - decay)
    for b_ema, b_cur in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b_cur)


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    np.add.at(tp, targets[preds == targets], 1.0)
    wrong = preds != targets
    np.add.at(fp, preds[wrong], 1.0)
    np.add.at(fn, targets[wrong], 1.0)
    support = tp + fn
    active = support > 0
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros(num_classes, dtype=np.float64)
    valid = active & (denom > 0)
    f1[valid] = (2.0 * tp[valid]) / denom[valid]
    return float(f1[active].mean() * 100.0) if active.any() else 0.0


def compute_roc_auroc_and_fpr95(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Tuple[float, float]:
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = labels[order]
    n_pos = max(1, int(labels.sum()))
    n_neg = max(1, len(labels) - n_pos)
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    idx = np.searchsorted(tpr, 0.95, side="left")
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])
    return auroc, fpr95


@torch.no_grad()
def precompute_features(raw_np: np.ndarray, device: torch.device, batch_size: int = 512) -> torch.Tensor:
    feats = []
    n = len(raw_np)
    for start in range(0, n, batch_size):
        chunk = torch.from_numpy(raw_np[start : start + batch_size]).to(device=device, dtype=torch.float32)
        feats.append(build_kinematic_v2_torch(chunk).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def evaluate_features(
    model: torch.nn.Module,
    feat_cpu: torch.Tensor,
    y_np: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    return_probs: bool = False,
) -> Dict:
    model.eval()
    all_logits = []
    all_ood = []
    for start in range(0, len(feat_cpu), batch_size):
        xb = feat_cpu[start : start + batch_size].to(device)
        logits, ood_logit, _ = model.forward_with_aux(xb, None)
        all_logits.append(logits.cpu())
        all_ood.append(torch.sigmoid(ood_logit).cpu())
    logits_t = torch.cat(all_logits, dim=0)
    ood_prob = torch.cat(all_ood, dim=0).numpy()
    top5_idx = logits_t.topk(5, dim=1).indices.numpy()
    preds = top5_idx[:, 0]
    top1 = float((preds == y_np).mean() * 100.0)
    top3 = float((top5_idx[:, :3] == y_np[:, None]).any(axis=1).mean() * 100.0)
    top5 = float((top5_idx == y_np[:, None]).any(axis=1).mean() * 100.0)
    macro_f1 = compute_macro_f1(preds, y_np, logits_t.shape[1])
    out = {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "macro_f1": macro_f1,
        "ood_prob": ood_prob,
    }
    if return_probs:
        out["logits"] = logits_t.numpy()
    return out


@torch.no_grad()
def eval_ood_scores(model: torch.nn.Module, neg_feat_cpu: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(neg_feat_cpu), batch_size):
        xb = neg_feat_cpu[start : start + batch_size].to(device)
        scores.append(torch.sigmoid(model.predict_ood_logit(xb)).cpu())
    return torch.cat(scores, dim=0).numpy()


def stabilize_split(x_np: np.ndarray) -> np.ndarray:
    return np.stack([stabilize_hand_sequence(seq) for seq in x_np], axis=0).astype(np.float32)


def mirror_split(x_np: np.ndarray) -> np.ndarray:
    return np.stack([mirror_sequence_x(seq, swap_slots=True) for seq in x_np], axis=0).astype(np.float32)


def build_model(args: argparse.Namespace, num_classes: int) -> torch.nn.Module:
    if args.arch == "conformer":
        return SignConformerClassifier(
            input_size=KINEMATIC_V2_DIM,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_classes=num_classes,
            emb_dim=args.emb_dim,
            num_subcenters=args.num_subcenters,
            dropout_p=args.dropout,
            drop_path=args.drop_path,
            s=args.arc_s,
            m=args.arc_m,
        )
    if args.arch == "pyramid_bigru":
        return SignPyramidBiGRUClassifier(
            input_size=KINEMATIC_V2_DIM,
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes,
            emb_dim=args.emb_dim,
            num_subcenters=args.num_subcenters,
            dropout_p=args.dropout,
            s=args.arc_s,
            m=args.arc_m,
        )
    raise ValueError(f"Unknown arch: {args.arch}")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    class_names: list,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict,
    ood_threshold: float,
) -> None:
    arch_name = "SignConformerClassifier" if args.arch == "conformer" else "SignPyramidBiGRUClassifier"
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "architecture": arch_name,
        "feature_version": "kinematic_v2_476",
        "stabilize_landmarks": True,
        "input_size": KINEMATIC_V2_DIM,
        "d_model": args.d_model,
        "hidden_size": args.d_model if args.arch == "conformer" else args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_classes": len(class_names),
        "emb_dim": args.emb_dim,
        "num_subcenters": args.num_subcenters,
        "dropout_p": args.dropout,
        "drop_path": args.drop_path,
        "s": args.arc_s,
        "m": args.arc_m,
        "ood_threshold": ood_threshold,
        "epoch": epoch,
        "val_top1": metrics["top1"],
        "val_top3": metrics["top3"],
        "val_top5": metrics["top5"],
        "val_macro_f1": metrics["macro_f1"],
        "val_mirrored_top1": metrics["mir_top1"],
        "val_ood_auroc": metrics["ood_auroc"],
        "val_ood_fpr95": metrics["ood_fpr95"],
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="conformer", choices=["conformer", "pyramid_bigru"])
    parser.add_argument("--epochs", type=int, default=38)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ood-per-step", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.1e-3)
    parser.add_argument("--min-lr", type=float, default=2.0e-5)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.025)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--ood-weight", type=float, default=0.35)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=320)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument("--num-subcenters", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--drop-path", type=float, default=0.08)
    parser.add_argument("--arc-s", type=float, default=32.0)
    parser.add_argument("--arc-m", type=float, default=0.30)
    parser.add_argument("--margin-warmup-epochs", type=int, default=12)
    parser.add_argument("--p-mirror", type=float, default=0.35)
    parser.add_argument("--p-hold-extend", type=float, default=0.25)
    parser.add_argument("--p-timemask", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="landmark_cache_packed")
    parser.add_argument("--save-dir", default="trained_models/full_2731/exp16_kinematic_conformer")
    parser.add_argument("--teacher-ckpt", default=None, help="Optional compatible 476D teacher checkpoint for online KL distillation")
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "class_names.json") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    t0 = time.perf_counter()
    print(f"Loading packed cache from {cache_dir} (device={device})...", flush=True)
    train_x_raw = stabilize_split(np.load(cache_dir / "train_x.npy"))
    train_y_np = np.load(cache_dir / "train_y.npy").astype(np.int64)
    val_x_raw = stabilize_split(np.load(cache_dir / "val_x.npy"))
    val_y_np = np.load(cache_dir / "val_y.npy").astype(np.int64)

    train_x_cpu = torch.from_numpy(train_x_raw)
    train_y_cpu = torch.from_numpy(train_y_np)

    val_feat_cpu = precompute_features(val_x_raw, device)
    val_mir_feat_cpu = precompute_features(mirror_split(val_x_raw), device)

    # Fixed realistic val OOD set (2048 negatives across all 4 non-sign modes)
    rng = np.random.default_rng(args.seed + 99)
    val_ood_src_idx = rng.choice(len(val_x_raw), size=2048, replace=False)
    val_ood_raw = torch.from_numpy(val_x_raw[val_ood_src_idx]).to(device)
    with torch.no_grad():
        val_ood_neg_raw = synthesize_ood_negative_batch(val_ood_raw)
        val_ood_feat_cpu = build_kinematic_v2_torch(val_ood_neg_raw).cpu()
    del val_ood_raw, val_ood_neg_raw
    print(f"Loaded & stabilized in {time.perf_counter() - t0:.1f}s | train={len(train_x_cpu)} val={len(val_feat_cpu)}", flush=True)

    teacher_model = None
    if args.teacher_ckpt:
        from signbridge.inference.model import load_signbridge_model
        teacher_model, _, t_cfg = load_signbridge_model(args.teacher_ckpt, device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"Loaded teacher {args.teacher_ckpt} ({t_cfg.get('architecture')}, val_top1={t_cfg.get('val_top1')})", flush=True)

    model = build_model(args, num_classes).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model={args.arch} params={n_params / 1e6:.2f}M save_dir={save_dir}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    start_epoch = 1
    best_val_top1 = -1.0
    best_is_ema = True
    best_metrics: Dict = {}

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_top1 = float(ckpt.get("best_val_top1", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val_top1={best_val_top1:.2f}%)", flush=True)

    n_train = len(train_x_cpu)
    steps_per_epoch = math.ceil(n_train / args.batch_size)
    log_path = save_dir / "training_log.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.perf_counter()
        if epoch <= args.warmup_epochs:
            lr_now = args.lr * (0.1 + 0.9 * (epoch / max(1, args.warmup_epochs)))
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            lr_now = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        m_now = args.arc_m * min(1.0, max(0.0, (epoch - 1) / max(1, args.margin_warmup_epochs)))
        set_subcenter_margin(model, m_now)

        model.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        running_ce = 0.0
        running_ood = 0.0
        running_kd = 0.0
        n_steps = 0

        for step_idx in range(steps_per_epoch):
            b_idx = perm[step_idx * args.batch_size : (step_idx + 1) * args.batch_size]
            raw_b = train_x_cpu[b_idx].to(device, non_blocking=False)
            y_b = train_y_cpu[b_idx].to(device, non_blocking=False)
            bsz = raw_b.shape[0]
            n_neg = min(args.ood_per_step, bsz)

            with torch.no_grad():
                aug_raw = augment_landmark_batch(
                    raw_b,
                    p_mirror=args.p_mirror,
                    p_hold_extend=args.p_hold_extend,
                )
                neg_raw = synthesize_ood_negative_batch(aug_raw[:n_neg])
                feat_all = build_kinematic_v2_torch(torch.cat([aug_raw, neg_raw], dim=0))
                pos_feat = feat_all[:bsz]
                neg_feat = feat_all[bsz:]
                pos_feat_masked = apply_feature_timemask(pos_feat, p=args.p_timemask)

            # Joint forward pass on mixed [pos, neg] batch so BatchNorm statistics never leak OOD identity
            all_feat = torch.cat([pos_feat_masked, neg_feat], dim=0)
            all_emb, all_ood = model.encode_sequence(all_feat)
            pos_emb = all_emb[:bsz]
            pos_ood = all_ood[:bsz]
            neg_ood = all_ood[bsz:]
            pos_logits = model.classifier(pos_emb, label=y_b)

            ce_loss = F.cross_entropy(pos_logits, y_b, label_smoothing=args.label_smoothing)
            ood_logits_cat = torch.cat([pos_ood, neg_ood], dim=0)
            ood_targets = torch.cat(
                [torch.ones_like(pos_ood), torch.zeros_like(neg_ood)], dim=0
            )
            # Rebalance positive:negative (bsz : n_neg) weighting so OOD head attends equally to negatives
            ood_pos_w = torch.full_like(pos_ood, 1.0)
            ood_neg_w = torch.full_like(neg_ood, float(bsz) / max(1, n_neg) * 0.5)
            ood_w = torch.cat([ood_pos_w, ood_neg_w], dim=0)
            ood_loss = F.binary_cross_entropy_with_logits(ood_logits_cat, ood_targets, weight=ood_w)

            loss = ce_loss + args.ood_weight * ood_loss

            if teacher_model is not None:
                with torch.no_grad():
                    t_logits, _, _ = teacher_model.forward_with_aux(pos_feat, None)
                    t_probs = F.softmax(t_logits / args.distill_temp, dim=1)
                s_logits_clean = model.classifier(pos_emb, label=None)
                kd_loss = F.kl_div(
                    F.log_softmax(s_logits_clean / args.distill_temp, dim=1),
                    t_probs,
                    reduction="batchmean",
                ) * (args.distill_temp ** 2)
                loss = loss + args.distill_weight * kd_loss
                running_kd += float(kd_loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)

            running_loss += float(loss.item())
            running_ce += float(ce_loss.item())
            running_ood += float(ood_loss.item())
            n_steps += 1

        # Evaluate raw and EMA models on clean val, mirrored val, and OOD negatives
        raw_res = evaluate_features(model, val_feat_cpu, val_y_np, device)
        ema_res = evaluate_features(ema_model, val_feat_cpu, val_y_np, device)

        use_ema = ema_res["top1"] >= raw_res["top1"]
        eval_model = ema_model if use_ema else model
        best_cur = ema_res if use_ema else raw_res

        mir_res = evaluate_features(eval_model, val_mir_feat_cpu, val_y_np, device)
        neg_ood_scores = eval_ood_scores(eval_model, val_ood_feat_cpu, device)
        auroc, fpr95 = compute_roc_auroc_and_fpr95(best_cur["ood_prob"], neg_ood_scores)
        ood_threshold = float(np.percentile(neg_ood_scores, 95))

        ep_sec = time.perf_counter() - ep_start
        summary = {
            "epoch": epoch,
            "lr": lr_now,
            "arc_m": m_now,
            "loss": running_loss / max(1, n_steps),
            "ce": running_ce / max(1, n_steps),
            "ood_loss": running_ood / max(1, n_steps),
            "kd_loss": running_kd / max(1, n_steps),
            "raw_val_top1": raw_res["top1"],
            "raw_val_top5": raw_res["top5"],
            "ema_val_top1": ema_res["top1"],
            "ema_val_top5": ema_res["top5"],
            "top1": best_cur["top1"],
            "top3": best_cur["top3"],
            "top5": best_cur["top5"],
            "macro_f1": best_cur["macro_f1"],
            "mir_top1": mir_res["top1"],
            "ood_auroc": auroc,
            "ood_fpr95": fpr95,
            "ood_threshold_p95": ood_threshold,
            "used_ema": use_ema,
            "sec": ep_sec,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        marker = ""
        if best_cur["top1"] > best_val_top1:
            best_val_top1 = best_cur["top1"]
            best_is_ema = use_ema
            best_metrics = summary
            save_checkpoint(save_dir / "best_model.pth", eval_model, class_names, args, epoch, summary, ood_threshold)
            marker = " *BEST*"

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_top1": best_val_top1,
                },
                save_dir / "latest.pth",
            )

        print(
            f"[Ep {epoch:02d}/{args.epochs:02d}] {ep_sec:5.1f}s lr={lr_now:.1e} m={m_now:.2f} "
            f"ce={summary['ce']:.3f} ood={summary['ood_loss']:.3f} | "
            f"val_top1={raw_res['top1']:.2f}% ema={ema_res['top1']:.2f}% top5={best_cur['top5']:.2f}% "
            f"f1={best_cur['macro_f1']:.2f}% mir={mir_res['top1']:.2f}% auroc={auroc:.4f}{marker}",
            flush=True,
        )

    print(f"\nTraining complete! Best val_top1={best_val_top1:.2f}% (used_ema={best_is_ema})", flush=True)

    if args.eval_test and (save_dir / "best_model.pth").exists():
        print("Evaluating best_model.pth on untouched test.csv (32,941 samples)...", flush=True)
        from signbridge.inference.model import load_signbridge_model
        best_model, _, _ = load_signbridge_model(save_dir / "best_model.pth", device)
        test_x_raw = stabilize_split(np.load(cache_dir / "test_x.npy"))
        test_y_np = np.load(cache_dir / "test_y.npy").astype(np.int64)
        test_feat_cpu = precompute_features(test_x_raw, device)
        test_res = evaluate_features(best_model, test_feat_cpu, test_y_np, device)

        # Subset mirror test (first 2,000 samples for perturbation check)
        test_mir_sub = precompute_features(mirror_split(test_x_raw[:2000]), device)
        mir_sub_res = evaluate_features(best_model, test_mir_sub, test_y_np[:2000], device)

        report = {
            "val": best_metrics,
            "test_top1": test_res["top1"],
            "test_top3": test_res["top3"],
            "test_top5": test_res["top5"],
            "test_macro_f1": test_res["macro_f1"],
            "test_mirrored_2k_top1": mir_sub_res["top1"],
            "test_mirrored_2k_top5": mir_sub_res["top5"],
        }
        with open(save_dir / "eval_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(
            f"TEST (32,941): top1={test_res['top1']:.2f}% top3={test_res['top3']:.2f}% "
            f"top5={test_res['top5']:.2f}% macro_f1={test_res['macro_f1']:.2f}% | "
            f"mirrored_2k={mir_sub_res['top1']:.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
