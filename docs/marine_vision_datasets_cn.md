# 海洋视觉数据集建议

本文档记录 Tiny CNN 分类路线可使用的公开数据源。使用前请再次核对各数据集的许可证、引用要求和下载条款。

## 当前已落地数据

第二阶段当前使用 OceanCV `PlasticInWater / TeachingTankPlastic`：

- 链接：[OceanCV PlasticInWater](https://huggingface.co/datasets/OceanCV/PlasticInWater)
- 格式：YOLOv8
- 原始图片：3,837 张
- 源数据声明许可证：CC BY 4.0
- 已生成分类图片：11,691 张，全部为 `96x96`
- 输出目录：`data/tiny_cls`
- 详细映射和数量：`data/tiny_cls/DATASET_INFO.md`

当前已生成 `unknown/plastic_bottle/foam/net` 四类；源数据没有 `buoy/ship_part`，这两类暂时为空。

PoTATO 也已处理为独立分类集：

- 本地来源：`C:\Users\cyj\Downloads\potato\potato`
- 转换脚本：`tools/prepare_potato_tiny_cls_dataset.py`
- 输出目录：`data/tiny_cls_potato`
- 输出规模：21,708 张 `96x96` 图片
- 类别：`unknown/plastic_bottle`
- 说明文件：`data/tiny_cls_potato/DATASET_INFO.md`

## 首选数据集

### 1. PoTATO

- 链接：[PoTATO GitHub](https://github.com/luisfelipewb/PoTATO/tree/eccv2024)
- 论文：[PoTATO: A Dataset for Analyzing Polarimetric Traces of Afloat Trash Objects](https://arxiv.org/abs/2409.12659)
- 场景：水面漂浮塑料瓶，USV 前视视角。
- 规模：论文摘要说明包含 12,380 个标注塑料瓶，并提供彩色/偏振相关原始数据。
- 适合用途：本项目优先用作 `plastic_bottle` 正样本和水面反光场景样本。
- 注意：数据集包含偏振 raw 数据，仓库说明提到提取图片约需 25GB 空间；本项目训练 Tiny CNN 时可优先使用 RGB 或导出的普通图像通道。

### 2. TrashCan

- 论文：[TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris](https://arxiv.org/abs/2007.08097)
- 场景：水下 ROV 视角海洋垃圾。
- 规模：论文中说明数据集包含 7,212 张标注图像，带 segmentation，并转换为 COCO 格式。
- 适合用途：补充 `plastic_bottle`、`net`、`ship_part` 或其他垃圾材质/实例外观。
- 注意：它更偏水下，不完全等同海面漂浮物；建议不要单独作为主训练集。

### 3. TACO

- 链接：[TACO GitHub](https://github.com/pedropro/TACO)
- 论文：[TACO: Trash Annotations in Context for Litter Detection](https://arxiv.org/abs/2003.06975)
- 场景：通用生活垃圾自然场景。
- 规模：论文摘要说明约 1,500 张图像和 4,784 个标注。
- 适合用途：补充塑料瓶、瓶罐、包装垃圾的外观多样性。
- 注意：不是海面数据，必须和真实海面 hard negative 混合使用，避免模型只学到陆地背景。

## Hard Negative 数据集

### 4. ATLANTIS

- 论文：[ATLANTIS: A Benchmark for Semantic Segmentation of Waterbody Images](https://arxiv.org/abs/2111.11567)
- 场景：水体、岸线、自然物、人造物语义分割。
- 规模：论文摘要说明包含 5,195 张水体图像，56 个语义类别。
- 适合用途：抽取海面、岸线、天空、船体局部、反光等 `unknown` hard negative。
- 注意：它不是漂浮垃圾数据集，更适合压低误报。

### 5. Marine Debris FLS

- 链接：[marine-debris-fls-datasets GitHub](https://github.com/mvaldenegro/marine-debris-fls-datasets/)
- 论文：[The Marine Debris Dataset for Forward-Looking Sonar Semantic Segmentation](https://arxiv.org/abs/2108.06800)
- 场景：前视声呐水下垃圾。
- 适合用途：不建议直接用于 RGB Tiny CNN；可作为“不要混用传感器域”的对照案例。

## 推荐组合

第一版 Tiny CNN 建议：

```text
plastic_bottle 正样本：PoTATO 为主，TACO/TrashCan 为辅
foam/net/ship_part：TrashCan + 自采样本
unknown/hard negative：ATLANTIS + 自采海面空场景
calib：必须从真实 ESP32-P4 摄像头或目标部署场景抽样
```

如果短期只有少量数据，先做二分类：

```text
unknown
target
```

等海面样本足够后再扩展成：

```text
unknown
plastic_bottle
foam
buoy
net
ship_part
```

## 转换脚本

本工程新增脚本：

```powershell
python tools\prepare_tiny_cls_dataset.py ^
  --source-root data\marine_yolo ^
  --output data\tiny_cls ^
  --names data\marine_yolo\data.yaml ^
  --class-map bottle=plastic_bottle ^
  --class-map foam=foam ^
  --class-map buoy=buoy ^
  --class-map net=net ^
  --class-map boat=ship_part ^
  --hard-negative-root data\hard_negative_water ^
  --full-image
```

如果 YOLO 类别是数字 ID，也可以写：

```powershell
python tools\prepare_tiny_cls_dataset.py ^
  --source-root data\marine_yolo ^
  --output data\tiny_cls ^
  --class-map 0=plastic_bottle ^
  --class-map 1=foam ^
  --class-map 2=buoy
```

输出目录：

```text
data/tiny_cls/
  train/unknown/
  train/plastic_bottle/
  train/foam/
  train/buoy/
  train/net/
  train/ship_part/
  valid/...
  test/...
  calib/
```

`calib/` 会从 train/valid 中抽样，用于后续 ESP-DL INT8 量化校准。
