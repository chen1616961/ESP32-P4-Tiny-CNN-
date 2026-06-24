#!/usr/bin/env python3
"""Convert PoTATO raw images into Tiny CNN classification crops.

PoTATO stores flat YOLO labels and raw polarization images:

    potato/
      images_raw/*_raw.png
      labels/*_rgb.txt
      split_seq/train.txt
      split_seq/val.txt
      split_seq/test.txt

This script extracts only the RGB modality in memory, then writes 96x96
classification crops without materializing the full RGB image set on disk.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


TINY_CLASSES = ["unknown", "plastic_bottle", "foam", "buoy", "net", "ship_part"]


@dataclass(frozen=True)
class YoloBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Tiny CNN crops from PoTATO.")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Path to the extracted potato/potato dataset directory.")
    parser.add_argument("--output", type=Path, default=Path("data/tiny_cls_potato"))
    parser.add_argument("--positive-expand", type=float, default=1.4)
    parser.add_argument("--negatives-per-image", type=int, default=2)
    parser.add_argument("--negative-iou", type=float, default=0.04)
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--calib-per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def demosaicing(img_raw: np.ndarray) -> list[np.ndarray]:
    img_bayer_090 = img_raw[0::2, 0::2]
    img_bayer_045 = img_raw[0::2, 1::2]
    img_bayer_135 = img_raw[1::2, 0::2]
    img_bayer_000 = img_raw[1::2, 1::2]
    return [
        cv2.cvtColor(img_bayer_000, cv2.COLOR_BayerBG2BGR),
        cv2.cvtColor(img_bayer_045, cv2.COLOR_BayerBG2BGR),
        cv2.cvtColor(img_bayer_090, cv2.COLOR_BayerBG2BGR),
        cv2.cvtColor(img_bayer_135, cv2.COLOR_BayerBG2BGR),
    ]


def extract_rgb(raw_path: Path) -> Image.Image | None:
    img_raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        return None
    demosaiced = demosaicing(img_raw)
    img_a = cv2.addWeighted(demosaiced[0], 0.5, demosaiced[2], 0.5, 0.0)
    img_b = cv2.addWeighted(demosaiced[1], 0.5, demosaiced[3], 0.5, 0.0)
    bgr = cv2.addWeighted(img_a, 0.5, img_b, 0.5, 0.0)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def parse_label_file(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append(YoloBox(int(float(parts[0])), *(float(v) for v in parts[1:5])))
        except ValueError:
            continue
    return boxes


def box_xyxy(box: YoloBox, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = (box.cx - box.w / 2.0) * width
    y1 = (box.cy - box.h / 2.0) * height
    x2 = (box.cx + box.w / 2.0) * width
    y2 = (box.cy + box.h / 2.0) * height
    return x1, y1, x2, y2


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def expanded_square(box: YoloBox, width: int, height: int, expand: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy(box, width, height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1, 4.0) * expand
    left = max(0, int(round(cx - side / 2.0)))
    top = max(0, int(round(cy - side / 2.0)))
    right = min(width, int(round(cx + side / 2.0)))
    bottom = min(height, int(round(cy + side / 2.0)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def random_negative_crop(width: int, height: int, boxes: list[tuple[float, float, float, float]],
                         max_iou: float, rng: random.Random) -> tuple[int, int, int, int] | None:
    min_side = max(8, min(width, height))
    for _ in range(100):
        side = int(rng.uniform(0.20, 0.70) * min_side)
        side = max(8, min(side, width, height))
        left = rng.randint(0, max(0, width - side))
        top = rng.randint(0, max(0, height - side))
        crop = (left, top, left + side, top + side)
        if all(iou_xyxy(crop, box) <= max_iou for box in boxes):
            return crop
    return None


def save_crop(img: Image.Image, crop: tuple[int, int, int, int], path: Path,
              size: int, quality: int) -> None:
    out = ImageOps.exif_transpose(img).convert("RGB").crop(crop)
    if size > 0:
        out = ImageOps.fit(out, (size, size), method=Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "JPEG", quality=quality, optimize=True)


def read_split_tokens(root: Path, split_name: str) -> list[str]:
    path = root / "split_seq" / split_name
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*.jpg")) if p.is_file()]


def build_calib_set(output: Path, per_class: int, rng: random.Random) -> int:
    calib_dir = output / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for cls in TINY_CLASSES:
        candidates: list[Path] = []
        for split in ("train", "valid"):
            folder = output / split / cls
            if folder.exists():
                candidates.extend(iter_images(folder))
        rng.shuffle(candidates)
        for index, src in enumerate(candidates[:per_class]):
            dst = calib_dir / f"{cls}_{index:04d}.jpg"
            dst.write_bytes(src.read_bytes())
            total += 1
    return total


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    split_map = {
        "train": "train.txt",
        "valid": "val.txt",
        "test": "test.txt",
    }
    counts = {split: {cls: 0 for cls in TINY_CLASSES} for split in split_map}

    for split, list_name in split_map.items():
        tokens = read_split_tokens(args.source_root, list_name)
        for idx, token in enumerate(tokens, start=1):
            raw_path = args.source_root / "images_raw" / f"{token}_raw.png"
            label_path = args.source_root / "labels" / f"{token}_rgb.txt"
            boxes = parse_label_file(label_path)
            if not boxes:
                continue
            img = extract_rgb(raw_path)
            if img is None:
                continue
            width, height = img.size
            all_xyxy = [box_xyxy(box, width, height) for box in boxes]

            for box_index, box in enumerate(boxes):
                if box.cls != 0:
                    continue
                crop = expanded_square(box, width, height, args.positive_expand)
                out = args.output / split / "plastic_bottle" / f"{token}_pos{box_index:02d}.jpg"
                save_crop(img, crop, out, args.resize, args.quality)
                counts[split]["plastic_bottle"] += 1

            for neg_index in range(max(0, args.negatives_per_image)):
                crop = random_negative_crop(width, height, all_xyxy, args.negative_iou, rng)
                if crop is None:
                    continue
                out = args.output / split / "unknown" / f"{token}_neg{neg_index:02d}.jpg"
                save_crop(img, crop, out, args.resize, args.quality)
                counts[split]["unknown"] += 1

            if idx % 250 == 0:
                print(f"{split}: processed {idx}/{len(tokens)}")

    calib_count = build_calib_set(args.output, args.calib_per_class, rng)
    print("PoTATO Tiny classification dataset prepared:")
    for split in ("train", "valid", "test"):
        summary = ", ".join(f"{cls}={counts[split][cls]}" for cls in TINY_CLASSES)
        print(f"  {split}: {summary}")
    print(f"  calib: {calib_count}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
