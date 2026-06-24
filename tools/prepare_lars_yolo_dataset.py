#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LaRS panoptic annotations into a small YOLO dataset.

This project only needs LaRS classes that help the Tiny CNN route:

    Buoy -> buoy
    Boat/ship, Row boats -> ship_part

LaRS test annotations do not include panoptic boxes in the downloaded package,
so this script uses LaRS train as train and splits LaRS val into valid/test.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["buoy", "ship_part"]
LARS_TO_YOLO = {
    "Buoy": 0,
    "Boat/ship": 1,
    "Row boats": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LaRS panoptic annotations to YOLO labels.")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/lars_yolo_buoy_ship"))
    parser.add_argument("--val-test-ratio", type=float, default=0.5,
                        help="Fraction of LaRS val images routed to YOLO test.")
    parser.add_argument("--min-area", type=float, default=20.0,
                        help="Discard very small panoptic segments before YOLO conversion.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def route_split(source_split: str, index: int, val_test_ratio: float) -> str:
    if source_split == "train":
        return "train"
    if source_split == "val":
        stride = max(2, round(1.0 / max(1e-6, min(0.95, val_test_ratio))))
        return "test" if index % stride == 0 else "valid"
    return source_split


def write_data_yaml(output: Path) -> None:
    text = "\n".join([
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names:",
        "  0: buoy",
        "  1: ship_part",
        "",
    ])
    (output / "data.yaml").write_text(text, encoding="utf-8")


def convert_split(args: argparse.Namespace, source_split: str, totals: dict[str, Counter]) -> None:
    panoptic = args.annotations_root / source_split / "panoptic_annotations.json"
    image_dir = args.images_root / source_split / "images"
    if not panoptic.exists():
        print(f"[skip] {source_split}: missing {panoptic}")
        return
    if not image_dir.exists():
        raise SystemExit(f"Image folder not found: {image_dir}")

    data = json.loads(panoptic.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}

    for index, ann in enumerate(data["annotations"]):
        meta = images.get(ann["image_id"])
        if not meta:
            continue
        src_image = image_dir / meta["file_name"]
        if not src_image.exists():
            print(f"[warn] missing image: {src_image}")
            continue

        width = float(meta["width"])
        height = float(meta["height"])
        lines: list[str] = []
        class_counts = Counter()
        for segment in ann.get("segments_info", []):
            name = categories.get(segment.get("category_id"))
            if name not in LARS_TO_YOLO:
                continue
            if float(segment.get("area", 0)) < args.min_area:
                continue
            x, y, w, h = [float(v) for v in segment["bbox"]]
            if w <= 1 or h <= 1:
                continue
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            nw = w / width
            nh = h / height
            values = [max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh)]
            cls_id = LARS_TO_YOLO[name]
            lines.append(f"{cls_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}")
            class_counts[CLASS_NAMES[cls_id]] += 1

        if not lines:
            continue

        split = route_split(source_split, index, args.val_test_ratio)
        out_image_dir = args.output / split / "images"
        out_label_dir = args.output / split / "labels"
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_image, out_image_dir / src_image.name)
        (out_label_dir / f"{src_image.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        totals[split]["images"] += 1
        totals[split].update(class_counts)


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        resolved_output = args.output.resolve()
        resolved_cwd = Path.cwd().resolve()
        if resolved_cwd not in resolved_output.parents and resolved_output != resolved_cwd:
            raise SystemExit(f"Refusing to overwrite outside workspace: {resolved_output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    totals = {split: Counter() for split in ("train", "valid", "test")}
    convert_split(args, "train", totals)
    convert_split(args, "val", totals)
    write_data_yaml(args.output)

    print("LaRS YOLO dataset written:")
    for split in ("train", "valid", "test"):
        summary = ", ".join(f"{key}={value}" for key, value in sorted(totals[split].items()))
        print(f"  {split}: {summary or 'empty'}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
