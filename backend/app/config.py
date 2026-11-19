import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
# Set before importing transformers, huggingface_hub or torch.
for key, directory in {"HF_HOME": "hf_cache", "HF_HUB_CACHE": "hf_cache/hub",
                       "TORCH_HOME": "torch_cache", "XDG_CACHE_HOME": "cache"}.items():
    os.environ[key] = str(WEIGHTS / directory)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass(frozen=True)
class Settings:
    weights: Path = WEIGHTS
    queue_capacity: int = 8
    ttl_seconds: float = 120
    max_body_bytes: int = 8_388_608
    max_tasks: int = 32
    max_image_pixels: int = 40_000_000
    max_long_side: int = 1280
    min_face_side: int = 80
    detector_iou: float = 0.3
    detector_score: float = 0.6
    max_yaw: float = 35
    max_pitch: float = 30
    min_blur: float = 35
    min_brightness: float = 35
    max_brightness: float = 225
    gate_timeout: float = 20
    text_timeout: float = 45
    cleanup_interval: float = 1


ETHICS = "说明：颜值评分与参考维度仅反映特定公开数据集标注群体的主观统计偏好，不代表客观美丑标准或个体价值；表观年龄为模型估计。本系统不做疾病诊断、性格命运推断或性取向预测。图像仅在本地演示设备内存处理并定时销毁，不上传公网云端。"
REJECTIONS = {
    "NO_HUMAN_FACE": "未识别到人脸",
    "POSE_TOO_LARGE": "人脸偏转角度过大，请正对镜头重拍",
    "IMAGE_TOO_BLURRY": "画面清晰度不足，请保持稳定后重拍",
    "POOR_LIGHTING": "画面过暗或过曝，请调整光线后重拍",
    "FACE_TOO_SMALL": "人脸占画面比例过小，请靠近镜头重拍",
    "GATE_UNAVAILABLE": "前置人脸复核暂不可用，无法确认有效人脸",
}


class GateRejected(Exception):
    def __init__(self, code, detail=""):
        super().__init__(detail)
        self.code = code
        self.detail = detail
