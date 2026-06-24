#!/usr/bin/env python3
"""Create a self-contained Tiny CNN delivery package for the ESP32-P4 project."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FIRMWARE_FILES = {
    ROOT / "build" / "bootloader" / "bootloader.bin": "firmware/bootloader.bin",
    ROOT / "build" / "partition_table" / "partition-table.bin": "firmware/partition-table.bin",
    ROOT / "build" / "esp32p4_buoy_vision_lab.bin": "firmware/esp32p4_buoy_vision_lab.bin",
    ROOT / "build" / "flash_args": "firmware/flash_args",
    ROOT / "build" / "flasher_args.json": "firmware/flasher_args.json",
}


PROJECT_FILES = [
    "CMakeLists.txt",
    "dependencies.lock",
    "flash_p4.ps1",
    "partitions.csv",
    "README.md",
    "sdkconfig",
    "sdkconfig.defaults",
    "sdkconfig.defaults.esp32p4",
]


SOURCE_DIRS = [
    "main",
    "components/example_video_common",
    "models",
    "docs",
    "tools",
]


SKIP_NAMES = {
    "__pycache__",
    ".pytest_cache",
}


SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "release" / "tinycls_marine_project"),
        help="Output package directory.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before creating the package.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def ignore_generated(dir_path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(dir_path) / name
        if name in SKIP_NAMES:
            ignored.add(name)
        elif path.is_file() and path.suffix in SKIP_SUFFIXES:
            ignored.add(name)
    return ignored


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore_generated)


def write_release_flash_script(out: Path) -> None:
    text = r'''param(
    [string]$Port = "COM3",
    [int]$Baud = 460800,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Firmware = Join-Path $Here "..\firmware"

$Bootloader = Join-Path $Firmware "bootloader.bin"
$Partitions = Join-Path $Firmware "partition-table.bin"
$App = Join-Path $Firmware "esp32p4_buoy_vision_lab.bin"

if (-not (Test-Path $Bootloader)) { throw "missing $Bootloader" }
if (-not (Test-Path $Partitions)) { throw "missing $Partitions" }
if (-not (Test-Path $App)) { throw "missing $App" }

& $Python -m esptool --chip esp32p4 -p $Port -b $Baud `
    --before default-reset --after hard-reset write-flash `
    --flash-mode dio --flash-size 16MB --flash-freq 80m `
    0x2000 $Bootloader `
    0x8000 $Partitions `
    0x10000 $App
'''
    path = out / "scripts" / "flash_release_p4.ps1"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_quickstart(out: Path) -> None:
    text = """# Tiny CNN Marine ESP32-P4 交付包

这个目录是从 ESP32-P4 智能无线视觉浮标工程整理出来的 Tiny CNN 轻量化部署包。

## 目录

```text
firmware/                 可以直接烧进 ESP32-P4 的固件产物
  bootloader.bin
  partition-table.bin
  esp32p4_buoy_vision_lab.bin
  flash_args
  flasher_args.json
scripts/
  flash_release_p4.ps1    只用 release 包内 bin 文件烧录 P4
tools/                    PC 端辅助脚本：数据集、训练、量化、板端采样
models/                   Tiny CNN ONNX / ESP-DL 模型和说明
main/                     板端主程序源码
components/               项目本地组件
docs/                     中文说明文档
```

## 1. 烧录到 ESP32-P4

在 Windows PowerShell 中进入本目录：

```powershell
cd release\\tinycls_marine_project
```

烧录 P4：

```powershell
.\\scripts\\flash_release_p4.ps1 -Port COM3
```

如果 `python` 不是可用命令，指定 ESP-IDF Python：

```powershell
.\\scripts\\flash_release_p4.ps1 -Port COM3 -Python C:\\Users\\cyj\\.espressif\\python_env\\idf6.0_py3.14_env\\Scripts\\python.exe
```

烧录后重启开发板。手机或电脑连接板子热点，或连接同一路由器，打开：

```text
http://<board-ip>/
```

常用接口：

```text
/api/status
/api/config?inference_interval_ms=0
/api/recognition?method=tinycls
/api/validate/run?sample=demo_01&method=tinycls&box_min_score=50
```

## 2. 上板后测 FPS 和 p95

从本目录运行：

```powershell
python tools\\sample_status_latency.py 192.168.4.1 --duration-min 1 --interval-ms 200 --method tinycls --wake --set-interval-zero
```

5 分钟测试：

```powershell
python tools\\sample_status_latency.py 192.168.4.1 --duration-min 5 --interval-ms 200 --method tinycls --wake --set-interval-zero --output reports\\tinycls_status_latency_5min.csv
```

判定目标：

```text
p95 analysis_ms < 200
inference_fps >= 5
```

## 3. 重新训练与量化流程

数据集整理：

```powershell
python tools\\prepare_tiny_cls_dataset.py --help
python tools\\prepare_potato_tiny_cls_dataset.py --help
python tools\\prepare_lars_yolo_dataset.py --help
python tools\\merge_tiny_cls_datasets.py --help
```

训练 Tiny CNN：

```powershell
python tools\\train_tiny_cnn_classifier.py --help
```

ESP-DL INT8 量化：

```powershell
python tools\\quantize_tiny_cls_espdl.py --help
```

生成新固件后，回到源工程运行：

```powershell
python tools\\make_tinycls_release.py --clean
```

它会刷新这个交付包里的 firmware、models、tools、docs 和源码。

## 4. 注意

- Python 脚本是 PC 端工具，不是在 ESP32-P4 上执行。
- 真正写入板子的是 `firmware/*.bin`。
- 当前固件默认识别方法是 `tinycls`，默认推理间隔是 `0 ms`。
- `/api/validate/run` 已支持 `method=tinycls`。
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def write_manifest(out: Path) -> None:
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    lines = ["# Release Manifest", "", f"root: {out}", "", "## Files", ""]
    lines.extend(f"- `{name}`" for name in files)
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = Path(args.output).resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for src, rel in FIRMWARE_FILES.items():
        copy_file(src, out / rel)

    for rel in PROJECT_FILES:
        src = ROOT / rel
        if src.exists():
            copy_file(src, out / rel)

    for rel in SOURCE_DIRS:
        copy_tree(ROOT / rel, out / rel)

    write_release_flash_script(out)
    write_quickstart(out)
    write_manifest(out)
    print(f"Release package written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
