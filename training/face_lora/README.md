# InternVL3-2B 五官描述 LoRA

模型输入是一张裁切人脸，使用单个 448×448 图块。六字段任务沿用后端的中文提示词。输出 JSON 的键为 `brows`、`eyes`、`nose`、`mouth`、`contour`、`expression`，每个部位是一句简体中文观察，无法标注时为 `null`。

## 数据准备

`prepare_scut.py` 读取 SCUT-FBP5500 的 `Images` 原图、官方 Fold 1 划分文件首列和 MediaPipe 几何结果。关键点决定眉弧、眼裂开口、鼻翼与内眼角距离、唇厚、下颌收窄等观察。表情使用嘴角模型系数与二维位置，或唇间开口。各字段的模板、阈值和测量代码都在脚本中，阈值参照训练折的几何分布。数据集正面且少遮挡，主要覆盖上述可测形态。自动生成的字段属于弱标签，可逐图在 `fields`、`field_codes` 和 `measurements` 中核对。 [SCUT-FBP5500 数据说明及非商业研究条款](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) [MediaPipe Face Landmarker 输出说明](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

运行 `prepare_scut.py --help` 查看位置参数。`--geometry` 接收按原图转正后计算的 JSONL，包含 `filename`、`status`、`width`、`height`、`landmarks`（归一化坐标）和 `blendshapes`。分割优先保留官方测试折，再保留训练折内由固定种子选取的开发集。去重同时检查原图与裁剪图 SHA-256、DCT 感知哈希和左右镜像。SCUT 未提供此包可用于隔离同一受试者的身份 ID，重复检查按照片执行。

现有数据位于 `training/artifacts/face_lora/data_final/`。其中 `train.jsonl` 有 3609 张，`dev.jsonl` 有 375 张，`test.jsonl` 有 1057 张。云端基础训练包 `face_lora_remote.tar` 含 train/dev 图片和 dev/test 排除指纹。`regression_overlay.tar.gz` 追加六张既有对照照片的指纹，并更新 train/dev 索引和元数据。每个 manifest 中的 `image` 相对于该 manifest 的目录。

额外语料通过重复传入 `--extra-train file.jsonl` 混入。下面是一条局部中文监督。程序自动生成单部位问题与同键 JSON 目标。

```json
{"id":"source:001","image":"faces/001.jpg","source":"source","fields":{"nose":"鼻梁较直。"}}
```

额外 manifest 也可使用六键 `fields`，或使用 `prompt` 与 JSON 对象 `answer` 明确指定局部问答。每行只能提供实际可追溯的观察，只有已知不可见或证据不足的部位使用六键任务中的 `null`。训练会重算图片的感知哈希，逐一核对准备阶段的指纹，并排除与已接纳语料或留出集合相交的外部图。

## GPU 训练

在 Linux 单卡 CUDA BF16 环境执行。先安装与驱动匹配的 PyTorch 2.8 和 torchvision 0.23，再 `python -m pip install -r requirements.txt`。使用 [OpenGVLab/InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf/tree/cb57a075cb75a2e6d1b668b128d48bb00ae321d2) 固定提交。传入 `--model /path/to/local-snapshot` 时从本地读取模型。

```bash
mkdir -p data/scut
tar -xf face_lora_remote.tar -C data/scut
tar -xzf regression_overlay.tar.gz -C data/scut

python train.py --data data/scut --output runs/smoke \
  --batch-size 16 --gradient-accumulation 1 --train-samples 64 --dev-samples 16 --max-steps 4

python train.py --data data/scut --output runs/one-epoch \
  --batch-size 16 --gradient-accumulation 1 --extra-train data/public/public_train.jsonl \
  --extra-heldout data/public/public_heldout.jsonl
```

每次运行使用新的输出目录。正式训练一轮，语言骨干 28 层自注意力的 q/k/v/o 共 112 个线性层使用 rank 16 LoRA。优化器仅更新 LoRA 参数。损失覆盖 assistant JSON 与结束标记，用户输入、图像占位符和 padding 均掩蔽。可加 `--gradient-checkpointing` 降低语言侧激活显存。

`adapter/` 保存 `adapter_config.json` 与 `adapter_model.safetensors`，可用 `PeftModel.from_pretrained(base_model, adapter_path)` 加载。`training_stats.json` 保存训练及开发集 loss、GPU 显存与时间、样本来源计数、数据哈希和排除记录。测试图单独用离线 A/B 工具比较通用基模和适配器。
