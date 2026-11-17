# 前后端接口与数据结构契约（API Contract v1.0）

本文档为 `backend/`（Python FastAPI 单进程推理服务）与 `android-app/`（Android 客户端）的统一接口契约 v1.0，前后端实现与自动化测试均以此契约为准。

---

## 1. 通信基础约定

- **默认服务监听**：
  - USB 有线回环模式（主模式）：后端监听 `127.0.0.1:8000`，Android 端通过 `adb reverse tcp:8000 tcp:8000` 访问 `http://127.0.0.1:8000`。
  - 手机热点局域网模式（备选模式）：后端监听 `0.0.0.0:8000`，Android 端通过右上角配置弹窗填入笔记本热点局域网 `http://<IP>:8000`。
- **并发与隐私契约**：
  - 后端固定单进程 `workers=1`，内部采用 **有界内存任务队列（默认容量 8）+ 单个 GPU 工作线程** 串行执行所有 GPU 推理与 Grad-CAM 反向传播。
  - **应用层不落盘**：图片上传不使用 `multipart/form-data` 的 `UploadFile` 临时文件机制，统一采用限制体积上限（`<= 8 MB`）的原始图片二进制请求体（`Content-Type: image/jpeg`、`image/png` 或 `application/octet-stream`），在内存中直接解码。
  - **内存生命周期（TTL）**：所有内存中的图像与可视化叠图绑定具体 `task_id`，默认保留 `120` 秒 TTL 后由后台清理器自动释放；客户端也可主动调用 `DELETE /api/v1/tasks/{task_id}` 立即销毁。

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
      "version": "mediapipe_face_landmarker_478"
    },
    "facellm": {
      "loaded": true,
      "version": "Idiap/FaceLLM-1B",
      "precision": "bf16"
    },
    "beauty_2dfap": {
      "loaded": true,
      "version": "2d_fap_mobilenetv2",
      "weights_status": "trained_fold1"
    },
    "apparent_age": {
      "loaded": true,
      "version": "se_resnext50_appa_real",
      "weights_status": "appa_real_pretrained"
    },
    "gender_stats": {
      "loaded": true,
      "source": "scut_fbp5500_train_stats"
    }
  }
}
```
> 注：若正式五折权重尚未放入指定路径，`beauty_2dfap.weights_status` 必须诚实返回 `"untrained_init"`（用于纵向最小闭环联调），严禁伪造已训练状态。

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
  "created_at": "2026-11-14T12:00:00+08:00"
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
`"queued"` | `"processing"` | `"rejected"` | `"need_selection"` | `"scores_ready"` | `"explanation_ready"` | `"completed"` | `"partial_error"`

#### （A）门禁拦截态（`status == "rejected"`）
- 当经候选框关联与逐候选 `FaceLLM-1B` 前置枚举判别（`HUMAN / NON_HUMAN / UNKNOWN`）后，**保留的有效 `HUMAN` 候选数为 0**（包括狗脸、猫脸、猴子脸、毛绒玩偶、卡通动漫头像、日常风景物体、或漏检图像）时：
  - `rejection.code` 必须为 `"NO_HUMAN_FACE"`
  - `rejection.message` 必须包含并突出显示 **`"未识别到人脸"`**（标准文案：`"未识别到人脸"`，可在 `detail` 字段补充内部原因供调试，前端主提示展示 `"未识别到人脸"`）
  - `scores`、`explanation`、`text_analysis` 必须全部为 `null`，绝不出分！
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
  "rejection": {
    "code": "NO_HUMAN_FACE",
    "message": "未识别到人脸",
    "detail": "0 human candidates remained after per-candidate gate"
  },
  "selection": null,
  "target_face": null,
  "scores": null,
  "geometry": null,
  "explanation": null,
  "text_analysis": null,
  "gender_views": null,
  "ethics_notice": "评分与参考维度仅反映特定公开数据集标注群体的主观统计偏好，不代表客观美丑标准或个体价值；图像仅在本地设备内存处理并定时销毁。",
  "timings_ms": {
    "gate_ms": 85.4,
    "total_elapsed_ms": 91.2
  }
}
```

#### （B）多人待点选态（`status == "need_selection"`）
人宠同框（如 1 人 + 1 狗）在剔除狗脸后仅剩 1 名有效人类候选，自动进入单目标处理，不进入此状态；仅当有效 `HUMAN` 候选数 `>= 2` 时进入此状态：
```json
{
  "task_id": "tsk_multi_001",
  "image_id": "img_multi_001",
  "status": "need_selection",
  "is_offline_replay": false,
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
  "geometry": null,
  "explanation": null,
  "text_analysis": null,
  "gender_views": null,
  "ethics_notice": "检测到多张人脸，请点击选择单人目标继续分析。",
  "timings_ms": {
    "gate_ms": 142.0,
    "total_elapsed_ms": 148.5
  }
}
```

#### （C）分阶段分析与完成态（`scores_ready` -> `explanation_ready` -> `completed` / `partial_error`）
```json
{
  "task_id": "tsk_ok_001",
  "image_id": "img_ok_001",
  "status": "completed",
  "is_offline_replay": false,
  "model_versions": {
    "beauty": "2d_fap_mobilenetv2 (trained_fold1)",
    "age": "se_resnext50_appa_real (appa_real_pretrained)",
    "text": "Idiap/FaceLLM-1B"
  },
  "generated_at": "2026-11-14T12:00:03+08:00",
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
    "weights_ready": true
  },
  "geometry": {
    "measurement_note": "基于转正像素平面并经双眼连线 Roll 水平对齐计算；首版测量中下庭比例（不含发际线上庭）。",
    "mid_to_lower_face_ratio": 0.96,
    "five_eyes": {
      "left_eye_to_face_ratio": 0.198,
      "intercanthal_to_face_ratio": 0.212,
      "right_eye_to_face_ratio": 0.196,
      "inner_canthus_to_eye_width_ratio": 1.07
    },
    "brow_to_eye_ratio": 0.58,
    "nose_wing_to_intercanthal_ratio": 0.99,
    "upper_to_lower_lip_ratio": 0.68,
    "mirror_symmetry_score": 0.942,
    "mirror_deviation_norm": 0.029
  },
  "explanation": {
    "original_image_url": "/api/v1/tasks/tsk_ok_001/artifacts/normalized.jpg",
    "landmarks_overlay_url": "/api/v1/tasks/tsk_ok_001/artifacts/landmarks_overlay.jpg",
    "gradcam_overlay_url": "/api/v1/tasks/tsk_ok_001/artifacts/gradcam_overlay.jpg",
    "random_baseline_mean_abs_delta": 0.031,
    "region_sensitivity": [
      {
        "region_key": "eyes",
        "region_name": "眼部区域",
        "perturbed_score": 3.69,
        "delta_score": 0.15,
        "abs_delta": 0.15,
        "relative_to_random": 0.119,
        "is_sensitive": true
      },
      {
        "region_key": "brows",
        "region_name": "眉部区域",
        "perturbed_score": 3.78,
        "delta_score": 0.06,
        "abs_delta": 0.06,
        "relative_to_random": 0.029,
        "is_sensitive": false
      },
      {
        "region_key": "nose",
        "region_name": "鼻部区域",
        "perturbed_score": 3.74,
        "delta_score": 0.10,
        "abs_delta": 0.10,
        "relative_to_random": 0.069,
        "is_sensitive": true
      },
      {
        "region_key": "mouth",
        "region_name": "嘴唇区域",
        "perturbed_score": 3.79,
        "delta_score": 0.05,
        "abs_delta": 0.05,
        "relative_to_random": 0.019,
        "is_sensitive": false
      },
      {
        "region_key": "contour",
        "region_name": "脸部轮廓与下颌",
        "perturbed_score": 3.72,
        "delta_score": 0.12,
        "abs_delta": 0.12,
        "relative_to_random": 0.089,
        "is_sensitive": true
      }
    ]
  },
  "text_analysis": {
    "status": "ok",
    "error_message": null,
    "fields": {
      "brows": {"label": "眉部", "description": "双眉轮廓清晰，眉形平缓自然，浓密度适中。", "visible": true},
      "eyes": {"label": "眼部", "description": "双眼皮形态清晰，眼裂适中，目光平视镜头。", "visible": true},
      "nose": {"label": "鼻部", "description": "鼻梁线条平直，鼻翼宽度与内眼角间距基本协调。", "visible": true},
      "mouth": {"label": "嘴部", "description": "唇部轮廓清晰，双唇自然闭合，嘴角微扬。", "visible": true},
      "contour": {"label": "脸部轮廓", "description": "鹅蛋型轮廓，下颌边缘过渡柔和对称。", "visible": true},
      "expression": {"label": "可见表情", "description": "神态放松，呈自然浅笑表情。", "visible": true}
    }
  },
  "gender_views": {
    "general": {
      "title": "通用综合参考",
      "focus_keys": ["eyes", "nose", "mouth", "brows", "contour", "expression"],
      "summary": "综合展示全部六项可见五官观察、像素平面几何比例（中下庭比 0.96、镜像对称度 94.2%）及模型敏感区域（眼部、轮廓、鼻部）。",
      "evidence_note": "客观几何测量 + 模型区域敏感性 + 可见特征描述"
    },
    "female": {
      "title": "女性侧重参考维度",
      "focus_keys": ["eyes", "mouth", "contour"],
      "summary": "按课程任务书建议维度重点关注眼型、唇形比例（上下唇比 0.68）与面部轮廓柔和度，结合眼部与轮廓区域的评分敏感性供参考。",
      "evidence_note": "侧重维度取自课程任务书建议；几何与评分关联参考 SCUT-FBP5500 女性子集（2,750 张）统计结果。"
    },
    "male": {
      "title": "男性侧重参考维度",
      "focus_keys": ["contour", "brows", "nose"],
      "summary": "按课程任务书建议维度重点关注面部轮廓与下颌线条、眉眼间距比（0.58）及中下庭结构比例（0.96），结合轮廓与鼻部敏感性供参考。",
      "evidence_note": "侧重维度取自课程任务书建议；几何与评分关联参考 SCUT-FBP5500 男性子集（2,750 张）统计结果。"
    }
  },
  "ethics_notice": "说明：颜值评分（基于 SCUT-FBP5500 男女各 2,750 张图像、60 名标注者）与表观年龄（基于 APPA-REAL）仅反映特定数据集的人群主观统计趋势，不代表客观美丑标准或个人价值。本系统不做任何疾病诊断、性格命运推断或性取向预测；图像仅在演示电脑本地内存处理并按短生存期自动销毁，不上传公网云端。",
  "timings_ms": {
    "gate_ms": 92.5,
    "scores_ms": 38.2,
    "explanation_ms": 64.7,
    "text_ms": 1820.4,
    "total_elapsed_ms": 2015.8
  }
}
```

---

### 2.5 获取任务内存图像资源：`GET /api/v1/tasks/{task_id}/artifacts/{artifact_name}`

- `artifact_name` 支持：
  - `normalized.jpg`：EXIF 转正并适度缩放后的原图
  - `landmarks_overlay.jpg`：绘制了目标人脸 478 点五官分区网格与对称中轴线的叠图
  - `gradcam_overlay.jpg`：将 2D-FAP 标量 Grad-CAM 热力图映射回原图目标人脸区域的叠图
- Response `200 OK`: `image/jpeg` 二进制字节流（`Cache-Control: no-store`）。

---

### 2.6 立即销毁任务内存数据：`DELETE /api/v1/tasks/{task_id}`

- Response `200 OK`: `{"task_id": "tsk_ok_001", "deleted": true}`
