#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantize the 96x96 Tiny CNN classifier ONNX model to ESP-DL INT8.

The exported ONNX expects NCHW float input that has already been normalized
with the same transform used during training:

    x = (rgb / 255.0 - 0.5) / 0.5

This script therefore calibrates ESP-PPQ with normalized classifier crops from
`data/tiny_cls_merged/calib` and writes an ESP32-P4 `.espdl` model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(roots: Iterable[Path], limit: int) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    files = sorted(dict.fromkeys(files))
    if limit > 0:
        return files[:limit]
    return files


class TinyClsCalibrationDataset(Dataset):
    """PTQ calibration crops for the Tiny CNN classifier."""

    def __init__(self, images: list[Path], size: int) -> None:
        self.images = images
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.images[index]).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr)


def patch_espdl_scalar_exporter() -> None:
    """Patch scalar parameter shapes before the ESP-DL FlatBuffer is written."""
    try:
        from esp_ppq.parser.espdl_exporter import EspdlExporter
    except Exception as exc:
        print(f"Skip ESP-DL scalar exporter patch: {exc}")
        return

    if getattr(EspdlExporter, "_tiny_cls_scalar_shape_patch", False):
        return

    original = EspdlExporter.build_variable_proto

    def patched(self, variable, exponent, layout, perm=None):
        value = getattr(variable, "value", None)
        if getattr(variable, "is_parameter", False) and value is not None:
            try:
                value_shape = list(value.shape)
                value_size = int(value.numel()) if hasattr(value, "numel") else int(value.size)
            except AttributeError:
                scalar = np.asarray(value)
                value_shape = list(scalar.shape)
                value_size = int(scalar.size)
                value = scalar
                variable.value = scalar
            if value_size >= 1 and (value_shape == [] or getattr(variable, "shape", None) in (None, [])):
                fixed_shape = value_shape if value_shape else [1]
                try:
                    if hasattr(value, "reshape"):
                        variable.value = value.reshape(fixed_shape)
                except Exception:
                    pass
                variable.shape = fixed_shape
        return original(self, variable, exponent, layout, perm)

    EspdlExporter.build_variable_proto = patched
    EspdlExporter._tiny_cls_scalar_shape_patch = True
    print("Patched ESP-DL exporter scalar parameter shape handling.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_classes(report_path: Path) -> list[str]:
    if not report_path.exists():
        return []
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return list(report.get("classes", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Tiny CNN classifier ONNX to ESP-DL INT8")
    parser.add_argument("--onnx", default="models/tiny_cls_96.onnx")
    parser.add_argument("--output", default="models/tiny_cls_96_s8_p4.espdl")
    parser.add_argument("--dataset", default="data/tiny_cls_merged")
    parser.add_argument("--report", default="reports/tiny_cls_report.json")
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--calib-limit", type=int, default=384)
    parser.add_argument("--calib-steps", type=int, default=48)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Keep 1 for the fixed-batch ONNX exported by train_tiny_cnn_classifier.py.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target", default="esp32p4")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    seed_everything(2026)
    patch_espdl_scalar_exporter()

    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx

    onnx_path = ROOT / args.onnx
    output_path = ROOT / args.output
    dataset_root = ROOT / args.dataset
    if not onnx_path.exists():
        raise SystemExit(f"ONNX not found: {onnx_path}")
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")

    roots = [
        dataset_root / "calib",
        dataset_root / "valid",
        dataset_root / "train",
    ]
    images = iter_images(roots, args.calib_limit)
    if not images:
        raise SystemExit(f"No calibration images found under: {dataset_root}")
    if args.batch_size != 1:
        print("Warning: this ONNX has a fixed batch dimension. Use --batch-size 1 unless it was re-exported as dynamic.")

    dataset = TinyClsCalibrationDataset(images, args.input_size)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    calib_steps = min(args.calib_steps, len(dataloader))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Quantizing Tiny CNN: onnx={onnx_path}, output={output_path}, "
        f"images={len(dataset)}, steps={calib_steps}, device={args.device}"
    )
    setting = QuantizationSettingFactory.espdl_setting()

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(args.device).type(torch.float32)

    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(output_path),
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=[1, 3, args.input_size, args.input_size],
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        setting=setting,
        device=args.device,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=args.verbose,
        inputs=None,
    )

    meta_path = output_path.with_suffix(".json")
    metadata = {
        "onnx": str(onnx_path.relative_to(ROOT)),
        "espdl": str(output_path.relative_to(ROOT)),
        "target": args.target,
        "bits": args.bits,
        "input_shape": [1, 3, args.input_size, args.input_size],
        "input_layout": "NCHW",
        "normalization": "rgb_float32 = (rgb / 255.0 - 0.5) / 0.5",
        "classes": load_classes(ROOT / args.report),
        "calibration_images": len(dataset),
        "calibration_steps": calib_steps,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Metadata written to {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
