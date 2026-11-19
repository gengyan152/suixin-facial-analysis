# 本地人脸分析后端

Windows / Python 3.13 / PyTorch 2.8.0+cu128 / RTX 4060 Laptop GPU。服务实现 API Contract v1.1 的六个路由，原图及结果图仅保留在任务内存中。实现范围为 `backend/`。

## 启动

建议直接在**仓库根目录**执行提供的启动脚本：

```powershell
# USB 回环模式（监听 127.0.0.1:8000）
.\run_usb_mode.ps1
# 另一窗口执行端口转发：
adb reverse tcp:8000 tcp:8000
```

热点模式在仓库根目录执行 `.\run_hotspot_mode.ps1`，监听 `0.0.0.0:8000`；手机端填入电脑热点/局域网网卡的 IPv4 地址，并按需放行防火墙 `8000` 端口入站规则。脚本固定 `workers=1`、关闭热重载，不同时启动两份服务。

服务启动后在单个 GPU 线程内加载模型，并以随模型缓存保存的 NASA 公开人像完成检测、前置复核、质量/姿态检查、数值评分和六字段描述预热。HTTP 健康检查在加载期间可用，等 `warmed_up=true` 再演示。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

当前若已挂载 SCUT-FBP5500 正式 Fold 1 第 90 轮权重：`beauty_2dfap.weights_status=trained_fold1`、`weights_ready=true`；未挂载时则保持 `untrained_init` / `weights_ready=false`。年龄使用官方 APPA-REAL 预训练权重。训练配置、五折指标与耗时见 [训练实测报告](../docs/TRAINING_ASSESSMENT.md)。

## 在新环境中准备

```powershell
# 在 backend/ 目录下执行（建议使用虚拟环境 python）
pip install -r requirements.txt
# 如所在网络需代理访问 Hugging Face / GitHub，可先设置 $env:HTTPS_PROXY / $env:HTTP_PROXY
python scripts/download_weights.py
python scripts/prepare_age.py
```

模型下载在准备阶段执行，服务推理使用 `local_files_only=True`，不访问外部推理服务。`HF_HOME`、`HF_HUB_CACHE`、`TORCH_HOME`、`XDG_CACHE_HOME` 均在导入模型库前固定到本目录的 `weights/` 下。下载脚本对大模型使用断点分段请求，校验固定 SHA-256，并写入 `weights/manifest.json`。初次下载约 4.2 GB InternVL3-2B 基座和 17 MB LoRA，另有约 309 MB 原始年龄检查点与约 104 MB 纯年龄权重。

`requirements.txt` 是部署所需的直接依赖；`requirements.lock.txt` 是实跑时整个 torch_env 的冻结记录。新环境使用前者，后者用于核对复现版本。OpenCV 使用 contrib 包，它提供 `cv2` 和 YuNet；无需再叠装另一个 OpenCV wheel。

| 文件 | 用途 |
|---|---|
| `weights/blaze_face_short_range.tflite` | MediaPipe 短距检测器 |
| `weights/face_detection_yunet_2023mar.onnx` | YuNet 双检关联 |
| `weights/face_landmarker.task` | 478 点、5 脸上限、3D 姿态矩阵 |
| `weights/InternVL3-2B-hf/` | 真人门禁与描述共用的基座 |
| `weights/InternVL3-2B-face-lora/` | 六字段描述适配器，由 v1.2 Release 提供 |
| `weights/2dfap_fold1_state_dict.pt` | 可选正式首折训练权重；缺失时确定性初始化 |
| `weights/appa_real_se_resnext50_state_dict.pt` | 官方 101 类年龄权重 |
| `weights/age_retrained_state_dict.pt` | 可选本地重训年龄权重，优先于官方预训练 |
| `weights/warmup.jpg` | 公开 NASA 宇航员照片，实际内容为 PNG，通过字节解码 |

InternVL3-2B 固定版本 `cb57a075cb75a2e6d1b668b128d48bb00ae321d2`。门禁使用基座推理，六字段描述启用 LoRA。两阶段共用同一个常驻模型，每张脸使用一个 448×448 图块。前置判别最多生成 12 个 token，时限为 20 秒。后置一次生成六字段 JSON，最多 220 个 token，时限为 45 秒。每一步生成都检查取消状态和时限。

## 模型迁移契约

- **2D-FAP**：MobileNetV2 的 `features.*`、`classifier.1.*` 键与官方及当前训练侧一致；40 维 `Sigmoid → L1` 概率输出；档中心为 `1.05 + 0.1*i`。五档聚合严格为 `[5,10,10,10,5]`；期望分在 40 档上计算。前处理为 RGB、Resize(256)、CenterCrop(224)、ImageNet 标准化。
- **年龄**：Cadene `se_resnext50_32x4d`，`last_linear` 输出 101 类，AdaptiveAvgPool2d(1)，Softmax 后对 0—100 加权求期望。输入始终是 **OpenCV BGR、224×224、float32 0—255**；不除 255，不做 ImageNet 归一化。
- **权重加载**：全部 `torch.load(..., weights_only=True)`、`strict=True`。接受纯张量字典或顶层 `state_dict`；统一去除 `module.` 前缀。官方年龄发布件的 argparse 元数据仅通过 `safe_globals([argparse.Namespace])` 白名单读取，再导出纯张量字典；没有 `weights_only=False` 路径。
- **替换权重**：把训练侧导出的纯 `state_dict` 放入约定文件名，重启服务；缺键、形状不符会明确反映为 startup_error，不默默部分加载。初始化模型不保存成“训练权重”。
- **推理**：两个数值模型始终在 `no_grad()` 与 `eval()` 下执行，直接输出期望分、五档概率分布与表观年龄。

## 门禁与文字

原始字节 → EXIF 转正 → 长边最多 1280 → 每个检测器内 NMS → 跨检测器一对一 IoU ≥ 0.3 → 候选短边 ≥ 80 → 用 InternVL3-2B 逐候选判别。

只保留 `HUMAN`；合法 `UNKNOWN` 也不放行。保留 0 人返回 `NO_HUMAN_FACE / 未识别到人脸`，分数、几何、解释与文字均为空。返回内容不是固定枚举、模型异常或超时则 `GATE_UNAVAILABLE`。人宠同框逐候选处理，动物不会导致已有真人候选被整图拒绝；多名真人只返回真人候选供点选。

选定目标在带边距的独立裁剪内提取网格，并与所选框再次匹配。初始质量阈值：绝对 yaw ≤ 35°、pitch ≤ 30°，目标框灰度均值 35—225，拉普拉斯方差 ≥ 35。阈值尚未经过独立锁定数据集标定。遮挡不做未经验证的硬拒绝。

模型为每个部位生成一条可见外观的中文短句或 `null`。后端逐项校验并组装 `label`、`description`、`visible`，缺失、无效或不可见的部位显示“无法判断”。最终段落拼接表观年龄、评分数值和通过校验的观察。JSON 无法解析、所有值无效或生成超时时，任务进入 `partial_error` 并保留数值结果。健康检查的 `warmup_metrics.text_ok` 记录描述预热状态。

### LoRA 评测

`scripts/eval_description.py` 在相同裁剪、提示词、单图块及贪心解码配置下比较 InternVL3-2B 基座及其 LoRA。评测会保存逐图原始文本和业务解析结果，并汇总六字段覆盖、严格 JSON、有标签部位的模板匹配和时延。模板匹配衡量模型复现特定几何属性的情况。对自由文本和同义表达，结合 `crop` 与 `raw` 逐图核对事实。

从仓库根目录在本地后端 Python 环境执行以下命令。清单中的 `image` 指向相对该清单的已裁剪图像，`sha256` 校验裁剪内容，`fields` 和 `field_codes` 分别记录参考短句与属性类别。

```powershell
$Data = 'training/artifacts/face_lora/data_final'
$Adapter = (Resolve-Path -LiteralPath 'backend/weights/InternVL3-2B-face-lora').Path
python backend/scripts/eval_description.py --manifest "$Data/test.jsonl" `
  --exclude-manifest "$Data/train.jsonl" --exclude-manifest "$Data/dev.jsonl" `
  --adapter $Adapter --output 'training/artifacts/face_lora_eval/v1.2-heldout40' --limit 40
```

`health.models.face_gate` 报告前置模型，`health.models.description` 和 `model_versions.text` 报告后置模型。准备脚本会把适配器安装到默认目录，启动时一并加载。

## 接口与内存生命周期

| 路由 | 行为 |
|---|---|
| `GET /api/v1/health` | 模型状态、预热、设备、队列容量/长度、TTL |
| `POST /api/v1/tasks` | 原始 image/jpeg、image/png 或 application/octet-stream，202 返回任务 |
| `POST /api/v1/tasks/{id}/select-face` | JSON `image_id + face_id` 精确续跑 |
| `GET /api/v1/tasks/{id}` | `queued → processing → scores_ready → completed`；或 rejected / need_selection / partial_error |
| `GET /api/v1/tasks/{id}/artifacts/{name}` | `normalized.jpg`（供多人选人与原图预览）；`Cache-Control: no-store` |
| `DELETE /api/v1/tasks/{id}` | 删除任务及资源，后续 404 |

仅图片资源是二进制；请求从 `Request.stream()` 累加，超过 8,388,608 字节立即 413，不使用 UploadFile、临时图片或磁盘数据库。解码额外限制 4000 万像素，防止小压缩文件展开成超大图。缩放/EXIF 映射随任务保留。

队列最多 8 个等待任务，所有任务合计最多 32 个（包括等待点选和已完成资源），避免 120 秒窗口内无界堆积。重复 client_request_id 对仍在处理或等待点选的任务幂等。TTL 从创建时刻起算，不因轮询刷新；1 秒周期清理，查询时也立即检查过期。删除后立刻撤销 URL、清除任务容器与待处理项；正在执行的模型调用在下一个取消检查点释放其局部张量，不会重新发布结果。任务完成后释放解码原图/裁剪，只保留短期 JPEG 资源；拒绝任务不保留图像资源。

API Contract 中没有独立的“数值模型执行失败”状态：此类故障保守终止为 `rejected/GATE_UNAVAILABLE`，`detail` 注明 inference failure 并清空结果。`partial_error` 专用于后置文字失败。

## 验证与复现

```powershell
# 1. 仅契约与几何单元测试（无需加载 GPU 大模型）
pytest -q

# 2. 全量测试（包含真实 CUDA 门禁、数值模型和文字模型；需已下载权重）
pytest -q --live

# 3. 实际启动 Uvicorn，完成预热、HTTP 上传/轮询/取图/删除冒烟验证
python scripts/http_smoke.py

# 或依次完成 --live 测试与 HTTP 冒烟验证：
python scripts/validate.py
```

`tests/test_contract.py` 通过依赖注入测试 API 状态机。`tests/test_live.py` 使用真实模型和公开照片，`--live` 启用三项集成测试。`http_smoke.py` 使用独立端口 18000，结束后关闭测试服务。

公开样本缓存由 `scripts/download_weights.py` 准备（详情见下节）。小样本用于工程回归与提示词开发，不构成门禁独立锁定测试集；尚无分门类大样本误放行率、误拒率或六字段属性准确率报告。初始评分权重的可解释图只验证计算实现，正式解释需要替换训练权重。

## 文件与来源

- `app/main.py`、`schemas.py`：六接口与 Pydantic 字段类型。
- `app/service.py`：有界队列、单 GPU 线程、TTL、分阶段发布。
- `app/pipeline.py`、`vlm.py`：双检测、姿态匹配、前后置模型。
- `app/numeric.py`、`quality.py`：模型迁移、数值推理、姿态/亮度/清晰度门禁。
- [2D-FAP 官方实现](https://github.com/enquan/2D_FAP)、[InternVL3-2B 权重及加载说明](https://huggingface.co/OpenGVLab/InternVL3-2B-hf)、[年龄模型官方发布](https://github.com/yu4u/age-estimation-pytorch/releases/tag/v1.0)。
- [MediaPipe 模型说明](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)、[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。
- 回归样本：scikit-image 的 astronaut（NASA Eileen Collins 公开照片）、chelsea（猫）；Matplotlib 的 Grace Hopper（美国海军公开照片）；PyTorch Hub 示例 dog.jpg。人宠与双人测试由这些公开样本在内存拼接，没有保存上传者照片。来源和哈希记录于模型缓存 manifest。
