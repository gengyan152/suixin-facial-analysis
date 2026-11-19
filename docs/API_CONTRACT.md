# 前后端接口与数据结构契约（API Contract v1.1）

本文档为 `backend/`（Python FastAPI 单进程推理服务）与 `android-app/`（Android 客户端）的统一接口契约 v1.1，前后端实现与自动化测试均以此契约为准。

---

## 1. 通信基础约定

- **默认服务监听**：
  - USB 有线回环模式（主模式）：后端监听 `127.0.0.1:8000`，Android 端通过 `adb reverse tcp:8000 tcp:8000` 访问 `http://127.0.0.1:8000`。
  - 手机热点局域网模式（备选模式）：后端监听 `0.0.0.0:8000`，Android 端通过右上角配置弹窗填入笔记本热点局域网 `http://<IP>:8000`。
- **并发与隐私契约**：
  - 后端固定单进程 `workers=1`，内部采用 **有界内存任务队列（默认容量 8）+ 单个 GPU 工作线程** 串行执行所有 GPU 推理。
  - **应用层不落盘**：图片上传不使用 `multipart/form-data` 的 `UploadFile` 临时文件机制，统一采用限制体积上限（`<= 8 MB`）的原始图片二进制请求体（`Content-Type: image/jpeg`、`image/png` 或 `application/octet-stream`），在内存中直接解码。
  - **内存生命周期（TTL）**：所有内存中的图像资源绑定具体 `task_id`，默认保留 `120` 秒 TTL 后由后台清理器自动释放；客户端也可主动调用 `DELETE /api/v1/tasks/{task_id}` 立即销毁。

---

## 2. 接口定义

### 2.1 健康检查与预热状态：`GET /api/v1/health`

用于 Android 右上角“服务地址配置与一键连通性自检”弹窗。

**Response `200 OK`**：
```json
{
  "status": "ok",
  "warmed_up": true,
  "device": "cuda:0 (NVIDIA GeForce RTX 4060 Laptop GPU)",
  "queue_length": 0,
  "queue_capacity": 8,
  "ttl_seconds": 120,
  "models": {
    "gate_detectors": {
      "loaded": true,
      "version": "mediapipe_face_detector + opencv_yunet"
    },
    "landmarks": {
      "loaded": true,
      "version": "mediapipe_face_landmarker_pose"
    },
    "face_gate": {
      "loaded": true,
      "version": "OpenGVLab/InternVL3-2B-hf",
      "precision": "bf16"
    },
    "description": {
      "loaded": true,
      "version": "OpenGVLab/InternVL3-2B-hf + LoRA",
      "precision": "bf16"
    },
    "beauty_2dfap": {
      "loaded": true,
      "version": "2d_fap_mobilenetv2",
      "weights_status": "trained_fold1",
      "weights_ready": true
    },
    "apparent_age": {
      "loaded": true,
      "version": "se_resnext50_appa_real",
      "weights_status": "appa_real_pretrained"
    }
  }
}
```
> 注：若正式五折权重尚未放入指定路径，`beauty_2dfap.weights_status` 必须诚实返回 `"untrained_init"`（`weights_ready: false`），严禁伪造已训练状态。

---

### 2.2 提交图像分析任务：`POST /api/v1/tasks`

- **Query 参数（可选）**：
  - `client_request_id`（string）：客户端防抖幂等标识；若同一 `client_request_id` 任务仍在队列或处理中，直接返回已有任务信息。
- **Headers**：
  - `Content-Type`: `image/jpeg` | `image/png` | `application/octet-stream`
- **Body**：原始图片字节流（上限 `8,388,608` 字节，即 8 MB）。

**Response `202 Accepted`**：
```json
{
  "task_id": "tsk_a1b2c3d4e5f6",
  "image_id": "img_a1b2c3d4e5f6",
  "status": "queued",
  "queue_position": 0,
  "created_at": "2026-11-17T11:40:00+08:00"
}
```

**错误响应**：
- `400 Bad Request`：`{"error_code": "INVALID_IMAGE", "message": "无法解码图像数据，请选择有效的照片"}`
- `413 Payload Too Large`：`{"error_code": "IMAGE_TOO_LARGE", "message": "图片大小超过 8 MB 限制，请压缩后重试"}`
- `429 Too Many Requests`：`{"error_code": "QUEUE_FULL", "message": "当前分析任务队列已满，请稍后再试"}`

---

### 2.3 多人候选点选目标人脸：`POST /api/v1/tasks/{task_id}/select-face`

当 `GET /api/v1/tasks/{task_id}` 返回 `status == "need_selection"` 时，用户在客户端点击其中一个人类候选框后调用，凭 `image_id + face_id` 续跑后续分析，无需重传图像。

**Request JSON**：
```json
{
  "image_id": "img_a1b2c3d4e5f6",
  "face_id": "face_0"
}
```

**Response `202 Accepted`**：
```json
{
  "task_id": "tsk_a1b2c3d4e5f6",
  "status": "processing",
  "selected_face_id": "face_0"
}
```

**错误响应**：
- `404 Not Found` / `410 Gone`：`{"error_code": "TASK_EXPIRED", "message": "临时图像已过期销毁，请重新选择或拍摄照片"}`
- `400 Bad Request`：`{"error_code": "INVALID_FACE_ID", "message": "所选目标人脸编号无效"}`

---

### 2.4 查询任务状态与分阶段结果：`GET /api/v1/tasks/{task_id}`

任务状态 `status` 取值集合固定为：
`"queued"` | `"processing"` | `"rejected"` | `"need_selection"` | `"scores_ready"` | `"completed"` | `"partial_error"`

#### （A）门禁拦截态（`status == "rejected"`）
- 当经候选框关联与逐候选 `InternVL3-2B` 前置枚举判别（`HUMAN / NON_HUMAN / UNKNOWN`）后，**保留的有效 `HUMAN` 候选数为 0**（包括狗脸、猫脸、猴子脸、毛绒玩偶、卡通动漫头像、日常风景物体、或漏检图像）时：
  - `rejection.code` 必须为 `"NO_HUMAN_FACE"`
  - `rejection.message` 必须为 **`"未识别到人脸"`**
  - `scores`、`text_analysis` 必须全部为 `null`，绝不出分。
- 当选定目标人类面孔因姿态、清晰度、光线、尺寸未达标，或前置门禁模型异常时：
  - `POSE_TOO_LARGE`：`"人脸偏转角度过大，请正对镜头重拍"`
  - `IMAGE_TOO_BLURRY`：`"画面清晰度不足，请保持稳定后重拍"`
  - `POOR_LIGHTING`：`"画面过暗或过曝，请调整光线后重拍"`
  - `FACE_TOO_SMALL`：`"人脸占画面比例过小，请靠近镜头重拍"`
  - `GATE_UNAVAILABLE`：`"前置人脸复核暂不可用，无法确认有效人脸"`

```json
{
  "task_id": "tsk_dog_001",
  "image_id": "img_dog_001",
  "status": "rejected",
  "is_offline_replay": false,
  "original_image_url": null,
  "rejection": {
    "code": "NO_HUMAN_FACE",
    "message": "未识别到人脸",
    "detail": "0 human candidates remained after per-candidate gate"
  },
  "selection": null,
  "target_face": null,
  "scores": null,
  "text_analysis": null,
  "ethics_notice": "评分与表观年龄仅反映特定公开数据集标注群体的主观统计趋势，不代表客观美丑标准或个人价值；图像仅在本地设备内存处理并定时销毁。",
  "timings_ms": {
    "gate_ms": 85.4,
    "total_elapsed_ms": 91.2
  }
}
```

#### （B）多人待点选态（`status == "need_selection"`）
人宠同框（如 1 人 + 1 狗）在剔除狗脸后仅剩 1 名有效人类候选，自动进入单目标处理；仅当有效 `HUMAN` 候选数 `>= 2` 时进入此状态：
```json
{
  "task_id": "tsk_multi_001",
  "image_id": "img_multi_001",
  "status": "need_selection",
  "is_offline_replay": false,
  "original_image_url": null,
  "rejection": null,
  "selection": {
    "image_width": 960,
    "image_height": 1280,
    "preview_image_url": "/api/v1/tasks/tsk_multi_001/artifacts/normalized.jpg",
    "candidates": [
      {
        "face_id": "face_0",
        "bbox_px": [140, 220, 410, 560],
        "bbox_norm": [0.1458, 0.1719, 0.4271, 0.4375],
        "score": 0.95
      },
      {
        "face_id": "face_1",
        "bbox_px": [520, 240, 790, 580],
        "bbox_norm": [0.5417, 0.1875, 0.8229, 0.4531],
        "score": 0.92
      }
    ]
  },
  "target_face": null,
  "scores": null,
  "text_analysis": null,
  "ethics_notice": "检测到多张人脸，请点击选择单人目标继续分析。",
  "timings_ms": {
    "gate_ms": 142.0,
    "total_elapsed_ms": 148.5
  }
}
```

#### （C）分阶段分析与完成态（`scores_ready` -> `completed` / `partial_error`）
```json
{
  "task_id": "tsk_ok_001",
  "image_id": "img_ok_001",
  "status": "completed",
  "is_offline_replay": false,
  "model_versions": {
    "beauty": "2d_fap_mobilenetv2 (trained_fold1)",
    "age": "se_resnext50_appa_real (appa_real_pretrained)",
    "text": "OpenGVLab/InternVL3-2B-hf + LoRA"
  },
  "generated_at": "2026-11-17T11:40:03+08:00",
  "original_image_url": "/api/v1/tasks/tsk_ok_001/artifacts/normalized.jpg",
  "rejection": null,
  "selection": null,
  "target_face": {
    "face_id": "face_0",
    "image_width": 960,
    "image_height": 1280,
    "bbox_px": [260, 210, 700, 790],
    "bbox_norm": [0.2708, 0.1641, 0.7292, 0.6172],
    "pose_deg": {
      "yaw": -3.2,
      "pitch": 2.1,
      "roll": 1.4
    },
    "blur_laplacian_var": 245.6,
    "brightness_mean": 134.2
  },
  "scores": {
    "beauty_score": 3.84,
    "beauty_display": 3.8,
    "scale_min": 1,
    "scale_max": 5,
    "distribution_label": "模型预测评分分布（1—5 档）",
    "rating_probabilities": [0.02, 0.08, 0.24, 0.51, 0.15],
    "distribution_std": 0.79,
    "apparent_age": 22.4,
    "apparent_age_display": 22,
    "weights_ready": true,
    "weights_status": "trained_fold1"
  },
  "text_analysis": {
    "status": "ok",
    "error_message": null,
    "fields": {
      "brows": {"label": "眉部", "description": "眉形呈弧形。", "visible": true},
      "eyes": {"label": "眼部", "description": "眼睛睁开，眼裂轮廓可见。", "visible": true},
      "nose": {"label": "鼻部", "description": "可见鼻梁线条较直。", "visible": true},
      "mouth": {"label": "嘴部", "description": "双唇闭合。", "visible": true},
      "contour": {"label": "脸部轮廓", "description": "可见面部轮廓偏椭圆。", "visible": true},
      "expression": {"label": "可见表情", "description": "可见嘴角上扬，呈微笑表情。", "visible": true}
    },
    "narrative": "画面中的人物表观年龄约 22 岁，颜值评分为 3.8 分（满分 5.0 分），五官比例舒展协调，整体气质清爽耐看。细看五官细节，脸部轮廓柔和偏椭圆，眉形柔和呈自然弧度，双眸自然舒展、眼型清晰，鼻梁线条平直挺括，双唇轻合，唇角轻扬带着浅浅笑意，整体面容舒展和谐。"
  },
  "ethics_notice": "说明：颜值评分（基于 SCUT-FBP5500 数据集）与表观年龄（基于 APPA-REAL）仅反映特定人群统计趋势，不代表客观美丑标准或个人价值；图像仅在本地设备内存处理并定时销毁，不上传公网云端。",
  "timings_ms": {
    "gate_ms": 92.5,
    "scores_ms": 38.2,
    "text_ms": 1820.4,
    "total_elapsed_ms": 1951.1
  }
}
```

---

### 2.5 获取任务内存图像资源：`GET /api/v1/tasks/{task_id}/artifacts/{artifact_name}`

- `artifact_name` 支持：
  - `normalized.jpg`：EXIF 转正并适度缩放后的原图（用于多人点选预览与结果展示）
- Response `200 OK`: `image/jpeg` 二进制字节流（`Cache-Control: no-store`）。

---

### 2.6 立即销毁任务内存数据：`DELETE /api/v1/tasks/{task_id}`

- Response `200 OK`: `{"task_id": "tsk_ok_001", "deleted": true}`
