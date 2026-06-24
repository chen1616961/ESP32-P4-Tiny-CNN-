# ESP32-P4 Tiny CNN 海洋视觉轻量化说明

本工程是在 ESP32-P4 智能无线视觉浮标实验项目基础上完成的轻量化优化版本。原项目使用 YOLO 系列检测模型作为主要识别路线，输出结果包含检测框、类别和置信度。由于 YOLO 检测模型在 ESP32-P4 端需要执行较重的特征提取、检测头后处理、候选框筛选和 NMS，端侧推理耗时较高，不利于达到连续实时识别所需的 5 FPS 以上目标。

本版本将主要识别路线切换为 Tiny CNN 海洋目标分类器。模型不再输出 bbox，也不再执行 YOLO 后处理，而是直接对摄像头 raw frame 做轻量分类，输出 `label / score / top_k / inference_ms / analysis_ms`。这种方式更适合当前海面漂浮物识别任务：系统只需要判断画面中是否存在目标及其类别，而不是必须给出精确检测框。因此，分类模型可以显著降低板端计算量，并保留原项目的网页控制台、NVS 配置、状态 API、历史记录和网络框架。

## 硬件环境

- 开发板：Waveshare ESP32-P4-WIFI6-DEV-KIT-A。
- 摄像头：MIPI-CSI 摄像头，实时采集 raw frame，固件内部再编码为 MJPEG 供网页预览。
- 存储：支持 TF 卡记录历史识别、快照和分段录像；TF 不可用时可使用内部 flash fallback。
- 网络：默认 AP+STA 模式，手机可连接板子热点访问网页，也可以通过路由器 STA 地址访问。
- 模型运行：Tiny CNN 通过 ESP-DL INT8 `.espdl` 模型嵌入固件，在 ESP32-P4 端直接加载运行。

## Tiny CNN 优化路线

本次优化的核心是用 Tiny CNN 分类器替代 YOLO 实时检测路径。

```text
原 YOLO 路线：
camera raw frame -> JPEG encode -> inference queue -> JPEG decode -> YOLO -> bbox/NMS -> detections

当前 Tiny CNN 路线：
camera raw frame -> tiny_cls_classify_frame() -> label/top_k/score -> vision_result_t
```

实时摄像头帧不再进入 YOLO 异步 JPEG 队列，也不再走检测框、NMS 和 JPEG decode 路径。Tiny CNN 通过 `tiny_cls_espdl_classify_frame()` 接收 raw frame，同步完成预处理、ESP-DL 推理和 top-k 后处理。最终结果统一写回 `vision_result_t`，因此网页和 API 不需要感知底层模型差异。

当前分类类别为：

```text
unknown
plastic_bottle
foam
buoy
net
ship_part
```

其中 `unknown` 用于背景、海浪、天空、岸线、船体局部等 hard negative 场景，降低误报；其余类别对应海面漂浮物和目标物类别。

## 模型和固件状态

- 默认识别方法：`tinycls`。
- 默认推理间隔：`0 ms`，即尽可能每帧推理。
- 模型文件：`models/tiny_cls_96_6cls_s8_p4.espdl`。
- 模型输入：`96x96 RGB`。
- 模型大小：约 `25 KB`。
- 固件内嵌模型符号：`tiny_cls_96_6cls_s8_p4_espdl`。
- 实时推理入口：`tiny_cls_espdl_classify_frame()`。
- 验证接口入口：`tiny_cls_espdl_classify_validation_jpeg()`，仅用于 `/api/validate/run` 的内嵌 JPEG 样例验证。

当前 `idf.py build` 已通过，固件可直接烧录到 ESP32-P4。由于没有实体板时无法测得真实板端 FPS 和 p95 延迟，本工程同时提供 `/api/status` 采样脚本，用于上板后连续采样 `vision.analysis_ms` 并统计 p95。

## Web 页面和 API

固件保留原项目的统一 Web 控制台和 API 结构。

```text
/                                  控制台首页：视频流、识别状态、调参和运行信息
/validate                          板端样例验证页面
/stream                            MJPEG 实时视频流
/api/status                        实时状态 JSON
/api/config                        读取或设置阈值、FPS、推理间隔等参数
/api/recognition?method=tinycls    切换到 Tiny CNN 分类路线
/api/validate/run?sample=demo_01&method=tinycls&box_min_score=50
                                   使用内嵌样例图测试 Tiny CNN 板端推理
/api/validate/overlay.svg?id=<id>  返回验证结果可视化图
```

`/api/status` 中重点关注：

```json
{
  "recognition_method": "tinycls",
  "inference_fps_x100": 520,
  "vision": {
    "model": "tiny_cls_96_6cls_s8_p4",
    "label": "plastic_bottle",
    "object": "plastic_bottle",
    "object_score": 92,
    "top_k": [
      {"label": "plastic_bottle", "score": 92},
      {"label": "foam", "score": 4},
      {"label": "unknown", "score": 3}
    ],
    "inference_ms": 38,
    "analysis_ms": 55
  }
}
```

判断指标：

```text
inference_fps_x100 / 100 >= 5
vision.analysis_ms p95 < 200 ms
```

## 板端验证方法

烧录固件后，先确认板子已启动 Web 服务，然后访问：

```text
http://<board-ip>/api/status
```

切换到 Tiny CNN 并设置每帧推理：

```text
http://<board-ip>/api/recognition?method=tinycls
http://<board-ip>/api/config?inference_interval_ms=0
```

运行样例验证：

```text
http://<board-ip>/api/validate/run?sample=demo_01&method=tinycls&box_min_score=50
```

返回 JSON 中应包含：

```text
ok=true
method=tinycls
vision.label
vision.object_score
vision.top_k
vision.inference_ms
vision.analysis_ms
```

与 YOLO 验证不同，Tiny CNN 分类结果不要求 `detections` 和 bbox。分类网络的主要输出是 `label / score / top_k`，`detection_count` 可以为 0。

## p95 和 FPS 采样

项目提供脚本：

```text
tools/sample_status_latency.py
```

上板后连续采样 5 分钟：

```powershell
python tools\sample_status_latency.py 192.168.4.1 --duration-min 5 --interval-ms 200 --method tinycls --wake --set-interval-zero --output reports\tinycls_status_latency_5min.csv
```

脚本会每 200ms 请求一次 `/api/status`，记录 `vision.analysis_ms`，并输出：

```text
analysis_ms avg / p50 / p95 / p99 / max
inference_ms avg / p95 / max
HTTP request latency
inference_fps
target check: p95_analysis<200ms
target check: latest_inference_fps>=5
```

CSV 结果可用于报告、论文附录或后续曲线绘制。

## 训练与量化流程

数据集由 YOLO 标注裁剪、PoTATO 海洋漂浮物数据、LaRS 海洋场景数据和 hard negative 样本合并得到。处理流程包括：

```text
1. 准备 Tiny CNN 分类数据集
2. 合并 unknown / plastic_bottle / foam / buoy / net / ship_part 六类样本
3. 训练 96x96 Tiny CNN 分类器
4. 导出 ONNX
5. 使用 ESP-DL 工具链做 INT8 量化
6. 将 .espdl 嵌入固件
7. 在 Web/API 中切换默认识别方法为 tinycls
8. 上板采样 FPS 和 p95 延迟
```

常用脚本：

```text
tools/prepare_tiny_cls_dataset.py
tools/prepare_potato_tiny_cls_dataset.py
tools/prepare_lars_yolo_dataset.py
tools/merge_tiny_cls_datasets.py
tools/train_tiny_cnn_classifier.py
tools/quantize_tiny_cls_espdl.py
tools/sample_status_latency.py
```

## 与 YOLO 路线对比

YOLO 路线适合需要检测框的目标定位任务，但在 ESP32-P4 端会引入较高的模型计算和后处理成本。Tiny CNN 路线放弃 bbox，只保留分类结果，更适合需要“是否发现海面目标”和“目标属于哪一类”的轻量场景。

本版本没有删除原项目的 Web、NVS、history、网络和 API 框架，而是在统一方法切换体系中新增并默认使用 `tinycls`。这样既保留了原工程的可调试性和可扩展性，又能最大化发挥 Tiny CNN 的速度优势。

## 当前结论

本优化版本已经完成从 YOLO 检测到 Tiny CNN 分类的主路径切换。固件默认使用 Tiny CNN，模型已嵌入 `.espdl`，实时推理路径直接处理 raw frame，验证接口支持 `method=tinycls`，并提供 p95/FPS 采样脚本。下一步只需要在实体 ESP32-P4 板上运行采样脚本，即可得到真实板端 `analysis_ms p95` 和 `inference_fps` 指标。
