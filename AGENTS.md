# AGENTS.md

## 定位
基于深度学习的人脸颜值评分与多维面部特征分析系统（Android 客户端 + 本地单 GPU FastAPI 推理后端 + 复现训练工程）。

## 常用启动与测试命令
- 启动后端（USB 回环 `127.0.0.1:8000`）：`.\run_usb_mode.ps1`
- 启动后端（热点/局域网 `0.0.0.0:8000`）：`.\run_hotspot_mode.ps1`
- 后端单元测试：`cd backend && pytest -q`（加 `--live` 跑真实 CUDA/FaceLLM 集成测试）
- Android 单测 / 构建：`.\android-app\build_apk.ps1`（自动通过 NTFS junction 规避 Windows 非 ASCII 路径问题）

## 技术栈
- `android-app/`：Kotlin、Jetpack Compose、Material 3、CameraX、端侧 MediaPipe BlazeFace、OkHttp、Coil。
- `backend/`：Python、FastAPI、Uvicorn（固定 `--workers 1` 单 GPU 线程）、PyTorch、Transformers（`Idiap/FaceLLM-1B`）、MediaPipe、OpenCV YuNet。
- `training/`：PyTorch 复现训练（`2D-FAP` MobileNetV2、`APPA-REAL` SE-ResNeXt-50、MediaPipe 478 点统计）。

## 目录与契约约定
- 前后端接口契约见 `docs/API_CONTRACT.md`。
- 图片上传使用原始二进制体（`<= 8 MB`），纯内存处理并设 `120s` TTL，不落盘临时图片。
- 模型权重统一放置于 `backend/weights/`（不入库）；`backend/app/config.py` 启动前固定 `HF_HOME`、`TORCH_HOME` 至该目录且推理禁用在线请求。
- `2D-FAP` / `APPA-REAL` 权重交接仅使用纯 `state_dict`（`weights_only=True, strict=True`）；年龄模型输入保持 BGR `0-255 float32`。

## 当前状态
- v1.1 已移除已弃用的 478 点像素平面几何、Grad-CAM 与局部模糊敏感性模块；保留人脸门禁、质量校验、颜值/表观年龄评估与 FaceLLM 六字段中文描述主链路。
