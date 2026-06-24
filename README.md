# Tiny CNN Marine ESP32-P4 交付包

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
cd release\tinycls_marine_project
```

烧录 P4：

```powershell
.\scripts\flash_release_p4.ps1 -Port COM3
```

如果 `python` 不是可用命令，指定 ESP-IDF Python：

```powershell
.\scripts\flash_release_p4.ps1 -Port COM3 -Python C:\Users\cyj\.espressif\python_env\idf6.0_py3.14_env\Scripts\python.exe
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
python tools\sample_status_latency.py 192.168.4.1 --duration-min 1 --interval-ms 200 --method tinycls --wake --set-interval-zero
```

5 分钟测试：

```powershell
python tools\sample_status_latency.py 192.168.4.1 --duration-min 5 --interval-ms 200 --method tinycls --wake --set-interval-zero --output reports\tinycls_status_latency_5min.csv
```

判定目标：

```text
p95 analysis_ms < 200
inference_fps >= 5
```

## 3. 重新训练与量化流程

数据集整理：

```powershell
python tools\prepare_tiny_cls_dataset.py --help
python tools\prepare_potato_tiny_cls_dataset.py --help
python tools\prepare_lars_yolo_dataset.py --help
python tools\merge_tiny_cls_datasets.py --help
```

训练 Tiny CNN：

```powershell
python tools\train_tiny_cnn_classifier.py --help
```

ESP-DL INT8 量化：

```powershell
python tools\quantize_tiny_cls_espdl.py --help
```

生成新固件后，回到源工程运行：

```powershell
python tools\make_tinycls_release.py --clean
```

它会刷新这个交付包里的 firmware、models、tools、docs 和源码。

## 4. 注意

- Python 脚本是 PC 端工具，不是在 ESP32-P4 上执行。
- 真正写入板子的是 `firmware/*.bin`。
- 当前固件默认识别方法是 `tinycls`，默认推理间隔是 `0 ms`。
- `/api/validate/run` 已支持 `method=tinycls`。
