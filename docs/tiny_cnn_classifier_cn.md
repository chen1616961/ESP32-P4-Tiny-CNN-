# Tiny CNN 分类器任务定义

本文档对应轻量化路线的第一步：先把部署任务从 YOLO 检测任务改成分类任务。后续训练、量化和固件接入都应遵守这里定义的类别、输出字段和验收指标。

## 1. 任务边界

当前 YOLO 路线输出：

```text
bbox + class + score
```

Tiny CNN 分类路线输出：

```text
label + score + top_k + inference_ms + analysis_ms
```

因此 Tiny CNN 不再承诺目标框坐标，也不再运行 YOLO head、DFL 解码、NMS 或 bbox 坐标映射。它只回答当前画面或候选裁剪区域属于哪一类，以及分类置信度是多少。

## 2. 类别定义

板端类别定义保存在：

```text
main/tiny_cls_task.h
```

当前 6 类训练模型类别为：

```text
unknown
plastic_bottle
foam
buoy
net
ship_part
```

`buoy` 来自 LaRS 的 `Buoy` 标注；`ship_part` 当前使用 LaRS 的 `Boat/ship` 与 `Row boats` 作为代理数据。若后续需要严格识别船体局部，需要继续补充人工局部裁剪样本。

如果后续数据集不足，也允许临时退化为二分类：

```text
unknown
target
```

但固件接口仍建议保留 `top_k` 数组，方便从二分类平滑升级到多分类。

## 3. 输入定义

第一版 Tiny CNN 输入固定为：

```text
96x96 RGB
```

训练与板端预处理必须保持一致：

```text
resize/crop -> RGB -> normalize -> int8 quantized model input
```

如果目标在画面中很小，不建议只做整图分类。推荐使用：

```text
整图分类 + 中心 crop + 多尺度 crop 分类
```

最终仍只输出分类结果，不输出检测框。

## 4. JSON 输出契约

板端 API 和历史记录建议使用以下字段：

```json
{
  "method": "tinycls",
  "model": "tiny-cnn-cls-96-6cls-p4",
  "label": "plastic_bottle",
  "score": 92,
  "top_k": [
    {"label": "plastic_bottle", "score": 92},
    {"label": "foam", "score": 4},
    {"label": "unknown", "score": 3}
  ],
  "inference_ms": 38,
  "analysis_ms": 55
}
```

字段含义：

```text
label         Top-1 分类标签
score         Top-1 分类置信度，0~100
top_k         前 K 个分类结果，默认 K=3
inference_ms  模型本体推理耗时
analysis_ms   预处理 + 推理 + 后处理总耗时
```

## 5. 数据集转换规则

从检测数据集迁移到分类数据集时：

```text
正样本   从 bbox 周围裁剪，裁剪框可扩大 1.2~1.6 倍
负样本   从无 bbox 区域随机裁剪，标为 unknown
整图样本 有目标图可作为 target/multi-class 弱标签样本
难负样本 海浪、反光、泡沫、天空、岸线、船体局部、模糊目标
```

建议目录结构：

```text
data/tiny_cls/
  train/unknown/
  train/plastic_bottle/
  train/foam/
  train/net/
  valid/...
  test/...
  calib/...
```

`calib/` 用于 INT8 量化校准，必须使用真实或代表性的摄像头图片，不使用随机输入。

本工程已经提供转换脚本：

```text
tools/prepare_tiny_cls_dataset.py
tools/prepare_potato_tiny_cls_dataset.py
tools/merge_tiny_cls_datasets.py
```

海洋视觉数据集来源和组合建议见 [marine_vision_datasets_cn.md](marine_vision_datasets_cn.md)。

当前 6 类合并训练数据在：

```text
data/tiny_cls_merged6/
```

训练和 ESP-DL INT8 量化命令：

```powershell
C:\Users\cyj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe tools\train_tiny_cnn_classifier.py --dataset data\tiny_cls_merged6 --output models\tiny_cls_96_6cls.pt --onnx models\tiny_cls_96_6cls.onnx --report reports\tiny_cls_6cls_report.json --epochs 12 --batch 128 --device cpu
C:\Users\cyj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -u tools\quantize_tiny_cls_espdl.py --onnx models\tiny_cls_96_6cls.onnx --output models\tiny_cls_96_6cls_s8_p4.espdl --dataset data\tiny_cls_merged6 --report reports\tiny_cls_6cls_report.json --input-size 96 --calib-limit 672 --calib-steps 200 --batch-size 1 --device cpu
```

训练产物：

```text
models/tiny_cls_96_6cls.pt
models/tiny_cls_96_6cls.onnx
reports/tiny_cls_6cls_report.json
```

ESP-DL INT8 量化产物：

```text
models/tiny_cls_96_6cls_s8_p4.espdl
models/tiny_cls_96_6cls_s8_p4.info
models/tiny_cls_96_6cls_s8_p4.json
```

当前 6 类 PC 侧测试指标：

```text
valid_acc=0.9288
test_acc=0.9125
test per-class:
  unknown=0.9052
  plastic_bottle=0.9240
  foam=1.0000
  buoy=0.7907
  net=1.0000
  ship_part=0.8243
```

## 6. 验收指标

Tiny CNN 主路线的验收指标应从检测框指标改为分类和延时指标：

```text
analysis_ms p95 < 200
inference_fps > 5
model_bytes < 500 KB
unknown false positive rate < 5%
target recall >= 85%
```

如果项目阶段更看重实时性，可以先接受较低召回率，用 `unknown` 阈值控制误报；如果更看重漏检率，则需要补充更多真实场景正样本和 hard negative 后再降低阈值。

## 7. 与现有路线关系

建议保留现有路线用于对照：

```text
off       纯图传测试
mlp       超轻量 baseline
tinycls   Tiny CNN 主部署路线
coco      COCO YOLO 对照
yolo11    自训练 YOLO11 对照
yolo26    自训练 YOLO26 对照
```

当前 PC 侧已经完成 6 类分类数据集准备、Tiny CNN 训练、ONNX 导出、ESP-DL INT8 量化和板端 `tinycls` 推理桥接层接入。后续步骤是在 ESP-IDF 环境中编译烧录，并在 ESP32-P4 上实测 `analysis_ms` 是否稳定低于 200 ms。
