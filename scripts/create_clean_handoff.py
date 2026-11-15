"""Build a clean SignBridge release bundle for sharing with new users.

Copies project source code and base production model weights while stripping
personal custom-sign data (`data/personalization`), secrets (`.env`), cached
TTS audio (`signbridge/web/static/audio/*.wav`), local caches, and build folders.
Never modifies or deletes the local developer workspace.
"""

import argparse
from pathlib import Path
import shutil
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".kilo",
    "node_modules",
    ".next",
    "dist",
    "logs",
    "dataset",
    "landmark_cache",
    "landmark_cache_full",
    "landmark_cache_packed",
    "landmark_interpolation_cache",
    "landmark_pose_cache",
    "pose_cache_full",
    "stitch_designs",
    "SignBridge-UI-Prototype",
    "lets-scroll-main",
    "__pycache__",
}

EXCLUDED_PREFIXES = (
    "data/personalization",
    "data/ncslgr_annotations",
    "trained_models/full_2731/exp1_",
    "trained_models/full_2731/exp2_",
    "trained_models/full_2731/exp3_",
    "trained_models/full_2731/exp4_",
    "trained_models/full_2731/exp5_",
    "trained_models/full_2731/exp6_",
    "trained_models/full_2731/exp7_",
    "trained_models/full_2731/exp8_",
    "trained_models/full_2731/exp9_",
    "trained_models/full_2731/exp10_",
    "trained_models/full_2731/exp11_",
    "trained_models/full_2731/exp12_",
    "trained_models/full_2731/exp13_",
    "trained_models/full_2731/exp14",
    "trained_models/full_2731/exp15",
    "trained_models/full_2731/exp16_diag5",
)

EXCLUDED_FILENAMES = {
    ".env",
    ".DS_Store",
    "latest.pth",
    "training_log.jsonl",
}


def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    posix = rel_path.as_posix()
    if any(posix == p or posix.startswith(p + "/") for p in EXCLUDED_PREFIXES):
        return False
    name = rel_path.name
    if name in EXCLUDED_FILENAMES:
        return False
    if name.startswith(".env.") and name != ".env.example":
        return False
    if posix in {"trained_models/tts/test_speech.wav"}:
        return False
    if rel_path.suffix == ".wav" and "/static/audio/" in f"/{posix}":
        return False
    if rel_path.suffix in {".pyc", ".pyo", ".log"}:
        return False
    return True


def build_clean_bundle(output_path: Path, as_zip: bool = True) -> dict:
    files_copied = 0
    if as_zip:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(PROJECT_ROOT.rglob("*")):
                if not item.is_file():
                    continue
                rel = item.relative_to(PROJECT_ROOT)
                if not should_include(rel):
                    continue
                zf.write(item, arcname=str(Path("signbridge") / rel))
                files_copied += 1
    else:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "data" / "personalization").mkdir(parents=True, exist_ok=True)
        for item in sorted(PROJECT_ROOT.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(PROJECT_ROOT)
            if not should_include(rel):
                continue
            dest = output_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            files_copied += 1

    return {"output": str(output_path), "files_copied": files_copied, "as_zip": as_zip}


def main():
    parser = argparse.ArgumentParser(description="Create a clean new-user SignBridge handoff bundle.")
    parser.add_argument(
        "--output",
        default="dist/signbridge-new-user-release.zip",
        help="Destination zip path or folder path (with --dir).",
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        help="Write an unzipped clean directory instead of a .zip archive.",
    )
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    stats = build_clean_bundle(out, as_zip=not args.dir)
    print(f"Created clean SignBridge release at {stats['output']} ({stats['files_copied']} files).")


if __name__ == "__main__":
    main()
