from .config import WEIGHTS

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class BeautyModel(nn.Module):
    """State keys identical to official MobileNetV2; 40 bins at 1.05 .. 4.95."""
    def __init__(self):
        super().__init__()
        network = mobilenet_v2(weights=None, num_classes=40)
        self.features = network.features
        self.classifier = nn.Sequential(*network.classifier, nn.Sigmoid())

    def forward(self, tensor):
        features = self.features(tensor).mean((2, 3))
        return nn.functional.normalize(self.classifier(features), p=1, dim=1)


def load_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Expected a pure tensor state_dict: {path.name}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def age_tensor(bgr):
    # Official yu4u contract: BGR float32 0..255, never ImageNet normalization.
    return torch.from_numpy(cv2.resize(bgr, (224, 224)).astype(np.float32).transpose(2, 0, 1).copy())


class NumericalModels:
    def __init__(self, weights=WEIGHTS, device="cuda:0"):
        import pretrainedmodels
        self.device = torch.device(device)
        with torch.random.fork_rng():
            torch.manual_seed(20261114)
            self.beauty = BeautyModel()
        beauty_path = weights / "2dfap_fold1_state_dict.pt"
        self.beauty_status = "trained_fold1" if beauty_path.exists() else "untrained_init"
        if beauty_path.exists():
            self.beauty.load_state_dict(load_state(beauty_path), strict=True)
        self.beauty.to(self.device).eval()
        self.age = pretrainedmodels.se_resnext50_32x4d(pretrained=None)
        self.age.last_linear = nn.Linear(self.age.last_linear.in_features, 101)
        self.age.avg_pool = nn.AdaptiveAvgPool2d(1)
        retrained = weights / "age_retrained_state_dict.pt"
        age_path = retrained if retrained.exists() else weights / "appa_real_se_resnext50_state_dict.pt"
        if not age_path.exists():
            raise FileNotFoundError("APPA-REAL weights missing; run scripts/prepare_age.py")
        self.age.load_state_dict(load_state(age_path), strict=True)
        self.age_status = "local_retrained" if retrained.exists() else "appa_real_pretrained"
        self.age.to(self.device).eval()
        self.transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize([.485,.456,.406],[.229,.224,.225])])
        self.rank = torch.arange(40, device=self.device, dtype=torch.float32) * .1 + 1.05

    def beauty_tensor(self, image):
        return self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    def score(self, image):
        with torch.no_grad():
            distribution = self.beauty(self.beauty_tensor(image).unsqueeze(0).to(self.device))[0]
            score = (distribution * self.rank).sum().item()
            bins = torch.stack([chunk.sum() for chunk in distribution.split([5,10,10,10,5])]).cpu().tolist()
            age = (self.age(age_tensor(image).unsqueeze(0).to(self.device)).softmax(1)[0] * torch.arange(101, device=self.device)).sum().item()
            coarse_mean = sum((i+1)*p for i,p in enumerate(bins))
            std = sum(p*((i+1)-coarse_mean)**2 for i,p in enumerate(bins)) ** .5
        return {"beauty_score": score, "beauty_display": round(score, 1), "scale_min": 1, "scale_max": 5,
                "distribution_label": "模型预测评分分布（1—5 档）", "rating_probabilities": bins,
                "distribution_std": std, "apparent_age": age, "apparent_age_display": int(age+.5),
                "weights_ready": self.beauty_status != "untrained_init", "weights_status": self.beauty_status}
