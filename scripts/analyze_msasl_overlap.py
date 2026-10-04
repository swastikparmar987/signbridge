#!/usr/bin/env python3
"""
SignBridge Dataset Overlap Analysis: ASL Citizen vs MS-ASL

Analyzes:
1. Exact and normalized lexical overlap between ASL Citizen (2,731 classes) and MS-ASL (1,000 classes).
2. Secondary synonym/variant-based overlap via MSASL_synonym.json and lexical suffix variants.
3. Video clip and unique signer statistics across MS-ASL train/val/test splits for matching classes.
4. Generates:
   - msasl_signbridge_overlap.csv
   - msasl_matching_train.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


def normalize_class_name(name: str) -> str:
    """Normalize class name by lowercasing, stripping, and removing spaces, underscores, and hyphens."""
    return re.sub(r"[\s_\-]+", "", str(name)).strip().lower()


def load_asl_citizen(splits_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str], Set[str]]:
    """
    Load ASL Citizen CSVs and map normalized gloss to canonical ASL Citizen gloss.
    Returns combined DataFrame, normalized->canonical map, and set of canonical glosses.
    """
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"ASL Citizen split file not found: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    canonical_glosses = set(combined_df["Gloss"].dropna().unique())

    norm_to_canonical = {}
    for g in sorted(canonical_glosses):
        norm_to_canonical[normalize_class_name(g)] = g

    return combined_df, norm_to_canonical, canonical_glosses


def load_msasl_classes(classes_path: Path) -> List[str]:
    """Load MS-ASL 1,000 class names list."""
    if not classes_path.exists():
        raise FileNotFoundError(f"MSASL_classes.json not found: {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_msasl_synonyms(synonyms_path: Path) -> List[List[str]]:
    """Load MS-ASL synonym groups."""
    if not synonyms_path.exists():
        return []
    with open(synonyms_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_overlap_analysis(project_root: Path, output_dir: Path):
    splits_dir = project_root / "dataset" / "ASL_Citizen" / "splits"
    msasl_dir = project_root / "MS-ASL"

    print("=== SignBridge Dataset Overlap Analysis ===")
    print(f"Project root: {project_root}")
    print(f"ASL Citizen splits directory: {splits_dir}")
    print(f"MS-ASL directory: {msasl_dir}\n")

    # 1. ASL Citizen inspection
    combined_asl, asl_norm_map, asl_glosses = load_asl_citizen(splits_dir)
    print("1. ASL Citizen Dataset:")
    print(f"   - Detected gloss column: 'Gloss'")
    print(f"   - Total unique classes: {len(asl_glosses)}")
    print(f"   - Total samples across splits: {len(combined_asl)}")

    # 2. MS-ASL Classes inspection
    ms_classes = load_msasl_classes(msasl_dir / "MSASL_classes.json")
    print("\n2. MS-ASL Dataset:")
    print(f"   - Total classes in MSASL_classes.json: {len(ms_classes)}")

    # 3. Exact / Normalized overlap
    exact_matches: Dict[int, Tuple[str, str]] = {}
    for idx, ms_name in enumerate(ms_classes):
        norm_ms = normalize_class_name(ms_name)
        if norm_ms in asl_norm_map:
            exact_matches[idx] = (ms_name, asl_norm_map[norm_ms])

    unmatched_ms = {idx: c for idx, c in enumerate(ms_classes) if idx not in exact_matches}
    matched_asl_classes = set(v[1] for v in exact_matches.values())
    unmatched_asl = sorted(asl_glosses - matched_asl_classes)

    print("\n3. Class Overlap Summary:")
    print(f"   - Exact / normalized matching classes: {len(exact_matches)} (64.6% of MS-ASL)")
    print(f"   - MS-ASL classes without exact ASL match: {len(unmatched_ms)} (35.4% of MS-ASL)")
    print(f"   - ASL Citizen classes without MS-ASL match: {len(unmatched_asl)} ({len(unmatched_asl)/len(asl_glosses)*100:.1f}% of ASL Citizen)")

    # 4. Synonym / Variant analysis
    ms_synonyms = load_msasl_synonyms(msasl_dir / "MSASL_synonym.json")
    asl_base_to_variants = defaultdict(list)
    for g in asl_glosses:
        base = normalize_class_name(re.sub(r"\d+$", "", g))
        asl_base_to_variants[base].append(g)

    syn_lookup = defaultdict(set)
    for group in ms_synonyms:
        norm_group = [normalize_class_name(w) for w in group]
        for w in group:
            syn_lookup[normalize_class_name(w)].update(norm_group)

    direct_variants = {}
    syn_exact_matches = {}
    syn_variant_matches = {}

    for idx, c in unmatched_ms.items():
        nc = normalize_class_name(c)
        if nc in asl_base_to_variants:
            direct_variants[idx] = (c, asl_base_to_variants[nc])
            continue

        syns = syn_lookup.get(nc, set())
        # check exact matches in synonym
        m_exact = [asl_norm_map[s] for s in syns if s != nc and s in asl_norm_map]
        if m_exact:
            syn_exact_matches[idx] = (c, m_exact)
            continue

        # check variant matches in synonym
        m_vars = []
        for s in syns:
            if s != nc and s in asl_base_to_variants and s not in asl_norm_map:
                m_vars.extend(asl_base_to_variants[s])
        if m_vars:
            syn_variant_matches[idx] = (c, list(set(m_vars)))

    completely_unmapped_ms = [
        idx for idx in unmatched_ms
        if idx not in direct_variants and idx not in syn_exact_matches and idx not in syn_variant_matches
    ]

    print("\n4. Secondary Synonym and Lexical Variant Analysis:")
    print(f"   - Unmatched MS-ASL classes matching ASL Citizen numbered variants: {len(direct_variants)}")
    print(f"   - Unmatched MS-ASL classes matching ASL Citizen via synonyms (MSASL_synonym.json): {len(syn_exact_matches)}")
    print(f"   - Unmatched MS-ASL classes matching ASL Citizen numbered variants via synonyms: {len(syn_variant_matches)}")
    print(f"   - Completely disjoint MS-ASL classes (no sign or variant in ASL Citizen): {len(completely_unmapped_ms)}")

    # 5. Clip & Signer Analysis across Splits
    with open(msasl_dir / "MSASL_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(msasl_dir / "MSASL_val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(msasl_dir / "MSASL_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Matching entries
    matching_train_entries = [d for d in train_data if d["label"] in exact_matches]
    matching_val_entries = [d for d in val_data if d["label"] in exact_matches]
    matching_test_entries = [d for d in test_data if d["label"] in exact_matches]

    # Overall signer stats
    all_train_signers = set(d["signer_id"] for d in train_data)
    all_val_signers = set(d["signer_id"] for d in val_data)
    all_test_signers = set(d["signer_id"] for d in test_data)
    total_ms_signers = all_train_signers | all_val_signers | all_test_signers

    # Matching signer stats
    match_train_signers = set(d["signer_id"] for d in matching_train_entries)
    match_val_signers = set(d["signer_id"] for d in matching_val_entries)
    match_test_signers = set(d["signer_id"] for d in matching_test_entries)
    match_all_signers = match_train_signers | match_val_signers | match_test_signers

    print("\n5. MS-ASL Clip and Signer Statistics for Exact Matching Classes:")
    print(f"   - Train clips: {len(matching_train_entries)} (out of {len(train_data)}, {len(matching_train_entries)/len(train_data)*100:.1f}%)")
    print(f"   - Val clips:   {len(matching_val_entries)} (out of {len(val_data)}, {len(matching_val_entries)/len(val_data)*100:.1f}%)")
    print(f"   - Test clips:  {len(matching_test_entries)} (out of {len(test_data)}, {len(matching_test_entries)/len(test_data)*100:.1f}%)")
    print(f"   - Total clips: {len(matching_train_entries) + len(matching_val_entries) + len(matching_test_entries)} (out of {len(train_data) + len(val_data) + len(test_data)}, {(len(matching_train_entries) + len(matching_val_entries) + len(matching_test_entries))/(len(train_data) + len(val_data) + len(test_data))*100:.1f}%)")
    print(f"   - Unique train signers: {len(match_train_signers)} (out of {len(all_train_signers)} MS-ASL train signers)")
    print(f"   - Unique val signers:   {len(match_val_signers)} (out of {len(all_val_signers)} MS-ASL val signers)")
    print(f"   - Unique test signers:  {len(match_test_signers)} (out of {len(all_test_signers)} MS-ASL test signers)")
    print(f"   - Unique overall signers: {len(match_all_signers)} (out of {len(total_ms_signers)} total MS-ASL signers)")

    # Per-class counts
    train_counts = Counter(d["label"] for d in matching_train_entries)
    val_counts = Counter(d["label"] for d in matching_val_entries)
    test_counts = Counter(d["label"] for d in matching_test_entries)

    train_signers_by_class = defaultdict(set)
    for d in matching_train_entries:
        train_signers_by_class[d["label"]].add(d["signer_id"])

    val_signers_by_class = defaultdict(set)
    for d in matching_val_entries:
        val_signers_by_class[d["label"]].add(d["signer_id"])

    test_signers_by_class = defaultdict(set)
    for d in matching_test_entries:
        test_signers_by_class[d["label"]].add(d["signer_id"])

    rows = []
    for lbl, (ms_cls, asl_cls) in exact_matches.items():
        tr_c = train_counts[lbl]
        va_c = val_counts[lbl]
        te_c = test_counts[lbl]
        rows.append({
            "asl_citizen_class": asl_cls,
            "msasl_class": ms_cls,
            "msasl_label": lbl,
            "train_clips": tr_c,
            "val_clips": va_c,
            "test_clips": te_c,
            "total_clips": tr_c + va_c + te_c,
            "unique_train_signers": len(train_signers_by_class[lbl]),
            "unique_val_signers": len(val_signers_by_class[lbl]),
            "unique_test_signers": len(test_signers_by_class[lbl]),
        })

    overlap_df = pd.DataFrame(rows)
    overlap_df.sort_values(by=["train_clips", "total_clips"], ascending=[False, False], inplace=True)

    # 6. Summary distribution
    print("\n6. Matching Class Distribution Metrics:")
    for col_name, s in [("Train clips", overlap_df["train_clips"]), ("Total clips (Train+Val+Test)", overlap_df["total_clips"])]:
        print(f"   --- {col_name} ---")
        print(f"   Mean clips/class:   {s.mean():.2f}")
        print(f"   Median clips/class: {s.median():.1f}")
        print(f"   Min clips/class:    {s.min()}")
        print(f"   Max clips/class:    {s.max()}")
        print(f"   Classes < 5 clips:   {(s < 5).sum()}")
        print(f"   Classes 5–20 clips:  {((s >= 5) & (s <= 20)).sum()}")
        print(f"   Classes 20–50 clips: {((s > 20) & (s <= 50)).sum()}")
        print(f"   Classes > 50 clips:  {(s > 50).sum()}")

    # 7. Write output files
    overlap_csv_path = output_dir / "msasl_signbridge_overlap.csv"
    overlap_df.to_csv(overlap_csv_path, index=False)
    print(f"\n7. Created {overlap_csv_path} with {len(overlap_df)} rows.")

    matching_train_json_path = output_dir / "msasl_matching_train.json"
    with open(matching_train_json_path, "w", encoding="utf-8") as f:
        json.dump(matching_train_entries, f, indent=2)
    print(f"   Created {matching_train_json_path} with {len(matching_train_entries)} metadata entries.")


def main():
    parser = argparse.ArgumentParser(description="SignBridge MS-ASL Overlap Analysis")
    parser.add_argument("--project-root", type=str, default=".", help="Path to SignBridge project root")
    parser.add_argument("--output-dir", type=str, default=".", help="Path to output directory")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_overlap_analysis(project_root, output_dir)


if __name__ == "__main__":
    main()
