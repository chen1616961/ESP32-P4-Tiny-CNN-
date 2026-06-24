#!/usr/bin/env python3
"""Build a Tiny CNN classification dataset from YOLO labels.

The output layout is:

    data/tiny_cls/
      train/unknown/*.jpg
      train/plastic_bottle/*.jpg
      valid/...
      test/...
      calib/*.jpg

Positive samples are crops around YOLO boxes. Negative samples are random crops
that avoid all boxes and are labeled as unknown. Optional hard-negative roots
can be added to populate unknown examples from water, shore, glare, sky, etc.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


TINY_CLASSES = ["unknown", "plastic_bottle", "foam", "buoy", "net", "ship_part"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO detection labels into Tiny CNN classification crops."
    )
    parser.add_argument("--source-root", type=Path, required=True,
                        help="YOLO dataset root. Expected split/images and split/labels folders.")
    parser.add_argument("--output", type=Path, default=Path("data/tiny_cls"),
                        help="Output classification dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Dataset splits to scan.")
    parser.add_argument("--class-map", action="append", default=[],
                        help="Map YOLO class to Tiny class, e.g. 0=plastic_bottle or bottle=plastic_bottle.")
    parser.add_argument("--names", type=Path, default=None,
                        help="Optional YOLO data.yaml/classes file used for name-based class maps.")
    parser.add_argument("--positive-expand", type=float, default=1.4,
                        help="Positive crop expansion around each bbox.")
    parser.add_argument("--negatives-per-image", type=int, default=2,
                        help="Number of random unknown crops per labeled image.")
    parser.add_argument("--full-image", action="store_true",
                        help="Also save one full-frame weak label per image.")
    parser.add_argument("--negative-iou", type=float, default=0.04,
                        help="Maximum IoU allowed between a negative crop and any YOLO bbox.")
    parser.add_argument("--resize", type=int, default=96,
                        help="Resize output crops to NxN. Use 0 to keep crop size.")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG output quality.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard-negative-root", action="append", type=Path, default=[],
                        help="Folder with extra images to add as unknown hard negatives.")
    parser.add_argument("--calib-per-class", type=int, default=64,
                        help="Images per class copied into output/calib for INT8 calibration.")
    return parser.parse_args()


def read_class_names(path: Path | None, source_root: Path) -> dict[int, str]:
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([source_root / "data.yaml", source_root / "classes.txt"])

    for candidate in candidates:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        if candidate.suffix.lower() == ".txt":
            return {i: line.strip() for i, line in enumerate(text.splitlines()) if line.strip()}
        inline = re.search(r"names\s*:\s*\[(.*?)\]", text, flags=re.S)
        if inline:
            names = [part.strip().strip("'\"") for part in inline.group(1).split(",")]
            return {i: name for i, name in enumerate(names) if name}
        lines = text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("names:"):
                start = i + 1
                break
        if start is not None:
            names: dict[int, str] = {}
            for line in lines[start:]:
                if not line.startswith((" ", "\t", "-")):
                    break
                match = re.search(r"(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?", line.strip())
                if match:
                    names[int(match.group(1))] = match.group(2).strip()
                    continue
                if line.strip().startswith("-"):
                    names[len(names)] = line.split("-", 1)[1].strip().strip("'\"")
            if names:
                return names
    return {}


def default_label_for_name(name: str) -> str | None:
    key = name.lower().replace("-", "_").replace(" ", "_")
    if any(token in key for token in ("plastic_bottle", "water_bottle", "bottle")):
        return "plastic_bottle"
    if "foam" in key or "styrofoam" in key:
        return "foam"
    if "buoy" in key or "float" in key:
        return "buoy"
    if "net" in key or "rope" in key:
        return "net"
    if "ship" in key or "boat" in key or "vessel" in key:
        return "ship_part"
    return None


def build_class_map(raw_maps: list[str], names: dict[int, str]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for cls_id, name in names.items():
        label = default_label_for_name(name)
        if label:
            mapping[cls_id] = label

    name_to_id = {name.lower(): cls_id for cls_id, name in names.items()}
    for item in raw_maps:
        if "=" not in item:
            raise SystemExit(f"Invalid --class-map {item!r}; expected source=target")
        src, dst = [part.strip() for part in item.split("=", 1)]
        if dst not in TINY_CLASSES:
            raise SystemExit(f"Unsupported Tiny class {dst!r}. Use one of: {', '.join(TINY_CLASSES)}")
        if src.isdigit():
            mapping[int(src)] = dst
        else:
            cls_id = name_to_id.get(src.lower())
            if cls_id is None:
                raise SystemExit(f"Class name {src!r} was not found in names file.")
            mapping[cls_id] = dst
    return mapping


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


def find_image_dir(root: Path, split: str) -> Path | None:
    for candidate in (root / split / "images", root / "images" / split, root / split):
        if candidate.exists():
            return candidate
    return None


def find_label_path(root: Path, split: str, image_path: Path) -> Path:
    candidates = [
        root / split / "labels" / f"{image_path.stem}.txt",
        root / "labels" / split / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def save_crop(img: Image.Image, crop: tuple[int, int, int, int], path: Path, size: int, quality: int) -> None:
    out = ImageOps.exif_transpose(img).convert("RGB").crop(crop)
    if size > 0:
        out = ImageOps.fit(out, (size, size), method=Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "JPEG", quality=quality, optimize=True)


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]


def prepare_output_dirs(output: Path) -> None:
    for split in ["train", "valid", "test"]:
        for cls in TINY_CLASSES:
            (output / split / cls).mkdir(parents=True, exist_ok=True)
    (output / "calib").mkdir(parents=True, exist_ok=True)


def convert_split(args: argparse.Namespace, split: str, mapping: dict[int, str],
                  rng: random.Random, counts: dict[str, dict[str, int]]) -> None:
    image_dir = find_image_dir(args.source_root, split)
    if not image_dir:
        print(f"[WARN] split {split}: no image directory found")
        return

    for image_path in iter_images(image_dir):
        label_path = find_label_path(args.source_root, split, image_path)
        boxes = parse_label_file(label_path)
        try:
            img = Image.open(image_path)
            img.load()
        except OSError:
            continue
        width, height = img.size
        all_xyxy = [box_xyxy(box, width, height) for box in boxes]
        mapped = [(box, mapping[box.cls]) for box in boxes if box.cls in mapping]

        for index, (box, label) in enumerate(mapped):
            crop = expanded_square(box, width, height, args.positive_expand)
            out = args.output / split / label / f"{image_path.stem}_pos{index:02d}.jpg"
            save_crop(img, crop, out, args.resize, args.quality)
            counts[split][label] += 1

        for neg_index in range(max(0, args.negatives_per_image)):
            crop = random_negative_crop(width, height, all_xyxy, args.negative_iou, rng)
            if crop is None:
                continue
            out = args.output / split / "unknown" / f"{image_path.stem}_neg{neg_index:02d}.jpg"
            save_crop(img, crop, out, args.resize, args.quality)
            counts[split]["unknown"] += 1

        if args.full_image:
            label = mapped[0][1] if len({label for _, label in mapped}) == 1 else "unknown"
            out = args.output / split / label / f"{image_path.stem}_full.jpg"
            save_crop(img, (0, 0, width, height), out, args.resize, args.quality)
            counts[split][label] += 1


def random_negative_crop(width: int, height: int, boxes: list[tuple[float, float, float, float]],
                         max_iou: float, rng: random.Random) -> tuple[int, int, int, int] | None:
    min_side = max(8, min(width, height))
    for _ in range(80):
        side = int(rng.uniform(0.20, 0.70) * min_side)
        side = max(8, min(side, width, height))
        left = rng.randint(0, max(0, width - side))
        top = rng.randint(0, max(0, height - side))
        crop = (left, top, left + side, top + side)
        if all(iou_xyxy(crop, box) <= max_iou for box in boxes):
            return crop
    return None


def add_hard_negatives(args: argparse.Namespace, rng: random.Random,
                       counts: dict[str, dict[str, int]]) -> None:
    images: list[Path] = []
    for root in args.hard_negative_root:
        images.extend(iter_images(root))
    rng.shuffle(images)
    if not images:
        return

    for index, image_path in enumerate(images):
        split = "train"
        if index % 10 == 8:
            split = "valid"
        elif index % 10 == 9:
            split = "test"
        try:
            img = Image.open(image_path)
            img.load()
        except OSError:
            continue
        width, height = img.size
        out = args.output / split / "unknown" / f"hard_{image_path.stem}_{index:05d}.jpg"
        save_crop(img, (0, 0, width, height), out, args.resize, args.quality)
        counts[split]["unknown"] += 1


def build_calib_set(output: Path, per_class: int, rng: random.Random) -> int:
    if per_class <= 0:
        return 0
    calib_dir = output / "calib"
    total = 0
    for cls in TINY_CLASSES:
        candidates = []
        for split in ("train", "valid"):
            candidates.extend(iter_images(output / split / cls))
        rng.shuffle(candidates)
        for index, src in enumerate(candidates[:per_class]):
            dst = calib_dir / f"{cls}_{index:04d}.jpg"
            shutil.copy2(src, dst)
            total += 1
    return total


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    names = read_class_names(args.names, args.source_root)
    mapping = build_class_map(args.class_map, names)
    if not mapping:
        print("[WARN] no class mapping was inferred; use --class-map 0=plastic_bottle etc.")

    prepare_output_dirs(args.output)
    counts = {split: {cls: 0 for cls in TINY_CLASSES} for split in ["train", "valid", "test"]}
    for split in args.splits:
        if split not in counts:
            counts[split] = {cls: 0 for cls in TINY_CLASSES}
        convert_split(args, split, mapping, rng, counts)
    add_hard_negatives(args, rng, counts)
    calib_count = build_calib_set(args.output, args.calib_per_class, rng)

    print("Tiny classification dataset prepared:")
    for split in sorted(counts):
        summary = ", ".join(f"{cls}={counts[split][cls]}" for cls in TINY_CLASSES)
        print(f"  {split}: {summary}")
    print(f"  calib: {calib_count}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
