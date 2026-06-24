#!/usr/bin/env python3
"""Train a small depthwise CNN classifier for ESP32-P4 experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install it with: python -m pip install torch"
    ) from exc


ALL_CLASSES = ["unknown", "plastic_bottle", "foam", "buoy", "net", "ship_part"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny CNN classification model.")
    parser.add_argument("--dataset", type=Path, default=Path("data/tiny_cls_merged"))
    parser.add_argument("--output", type=Path, default=Path("models/tiny_cls_96.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/tiny_cls_report.json"))
    parser.add_argument("--onnx", type=Path, default=Path("models/tiny_cls_96.onnx"))
    parser.add_argument("--imgsz", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-onnx", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def discover_classes(root: Path) -> list[str]:
    classes = []
    for cls in ALL_CLASSES:
        folder = root / "train" / cls
        if folder.exists() and any(folder.glob("*.jpg")):
            classes.append(cls)
    if len(classes) < 2:
        raise SystemExit(f"Need at least two non-empty classes in {root / 'train'}")
    return classes


def collect_samples(root: Path, split: str, classes: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for label, cls in enumerate(classes):
        folder = root / split / cls
        if not folder.exists():
            continue
        samples.extend(Sample(path, label) for path in sorted(folder.glob("*.jpg")))
    return samples


class TinyClsDataset(Dataset):
    def __init__(self, samples: list[Sample], imgsz: int, train: bool):
        self.samples = samples
        self.imgsz = imgsz
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        img = Image.open(sample.path).convert("RGB")
        img = ImageOps.fit(img, (self.imgsz, self.imgsz), method=Image.Resampling.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.18))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.18))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.90, 1.12))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, sample.label


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DSConv(16, 24, stride=2),
            DSConv(24, 32, stride=2),
            DSConv(32, 48, stride=2),
            DSConv(48, 64, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def counts_by_class(samples: list[Sample], num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for sample in samples:
        counts[sample.label] += 1
    return counts


def class_weights(counts: list[int]) -> torch.Tensor:
    total = sum(counts)
    weights = []
    for count in counts:
        weights.append(total / max(1, count))
    mean = sum(weights) / len(weights)
    return torch.tensor([w / mean for w in weights], dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int) -> dict:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        loss_sum += float(loss.item()) * int(labels.numel())
        for truth, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[int(truth), int(pred)] += 1
    per_class = []
    for cls_id in range(num_classes):
        support = int(confusion[cls_id].sum())
        acc = float(confusion[cls_id, cls_id] / support) if support else 0.0
        per_class.append({"support": support, "accuracy": acc})
    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def choose_device(text: str) -> torch.device:
    if text == "cuda":
        return torch.device("cuda")
    if text == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes = discover_classes(args.dataset)
    train_samples = collect_samples(args.dataset, "train", classes)
    valid_samples = collect_samples(args.dataset, "valid", classes)
    test_samples = collect_samples(args.dataset, "test", classes)
    if not train_samples or not valid_samples:
        raise SystemExit("train and valid splits must be non-empty")

    device = choose_device(args.device)
    train_set = TinyClsDataset(train_samples, args.imgsz, train=True)
    valid_set = TinyClsDataset(valid_samples, args.imgsz, train=False)
    test_set = TinyClsDataset(test_samples, args.imgsz, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers)

    model = TinyCNN(len(classes)).to(device)
    counts = counts_by_class(train_samples, len(classes))
    criterion = nn.CrossEntropyLoss(weight=class_weights(counts).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_valid = -math.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        scheduler.step()
        valid = evaluate(model, valid_loader, device, len(classes))
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "valid_loss": valid["loss"],
            "valid_accuracy": valid["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_acc={row['valid_accuracy']:.4f}"
        )
        if valid["accuracy"] > best_valid:
            best_valid = valid["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    valid = evaluate(model, valid_loader, device, len(classes))
    test = evaluate(model, test_loader, device, len(classes)) if test_samples else {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "imgsz": args.imgsz,
        "model": "TinyCNN",
    }, args.output)

    exported_onnx = False
    if not args.no_onnx:
        try:
            args.onnx.parent.mkdir(parents=True, exist_ok=True)
            dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
            torch.onnx.export(
                model,
                dummy,
                args.onnx,
                input_names=["input"],
                output_names=["logits"],
                opset_version=18,
            )
            exported_onnx = True
        except Exception as exc:
            print(f"ONNX export skipped: {exc}")

    report = {
        "dataset": str(args.dataset),
        "classes": classes,
        "imgsz": args.imgsz,
        "train_counts": dict(zip(classes, counts)),
        "valid": valid,
        "test": test,
        "history": history,
        "output": str(args.output),
        "onnx": str(args.onnx) if exported_onnx else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")
    print(f"report: {args.report}")
    if exported_onnx:
        print(f"onnx: {args.onnx}")


if __name__ == "__main__":
    main()
