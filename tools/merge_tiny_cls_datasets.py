#!/usr/bin/env python3
"""Merge Tiny CNN classification datasets without overwriting source outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASSES = ["unknown", "plastic_bottle", "foam", "buoy", "net", "ship_part"]
SPLITS = ["train", "valid", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Tiny CNN classification datasets.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Input dataset roots, each containing train/valid/test folders.")
    return parser.parse_args()


def copy_split_class(src_root: Path, out_root: Path, split: str, cls: str, prefix: str) -> int:
    src_dir = src_root / split / cls
    if not src_dir.exists():
        return 0
    out_dir = out_root / split / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob("*.jpg")):
        dst = out_dir / f"{prefix}_{src.name}"
        shutil.copy2(src, dst)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    totals = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}

    for src_root in args.inputs:
        if not src_root.exists():
            raise SystemExit(f"Input dataset not found: {src_root}")
        prefix = src_root.name.replace("tiny_cls_", "").replace("tiny_cls", "base")
        for split in SPLITS:
            for cls in CLASSES:
                totals[split][cls] += copy_split_class(src_root, args.output, split, cls, prefix)

    calib_dir = args.output / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_total = 0
    for src_root in args.inputs:
        prefix = src_root.name.replace("tiny_cls_", "").replace("tiny_cls", "base")
        src_calib = src_root / "calib"
        if not src_calib.exists():
            continue
        for src in sorted(src_calib.glob("*.jpg")):
            shutil.copy2(src, calib_dir / f"{prefix}_{src.name}")
            calib_total += 1

    print("Merged Tiny CNN dataset:")
    for split in SPLITS:
        summary = ", ".join(f"{cls}={totals[split][cls]}" for cls in CLASSES)
        print(f"  {split}: {summary}")
    print(f"  calib: {calib_total}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
