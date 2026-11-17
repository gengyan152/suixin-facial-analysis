"""Exact upstream architecture/state_dict keys, with modern torch imports."""
from runtime import ROOT
import hashlib
import importlib.util
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNetV2


class TwoDFAP(MobileNetV2):
    def __init__(self, pretrained=False):
        super().__init__(num_classes=1000)
        self.classifier.add_module("2", nn.Sigmoid())
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                map_location="cpu", check_hash=True, weights_only=True), strict=True)
        self.classifier[1] = nn.Linear(self.last_channel, 40)

    def _forward_impl(self, x):
        x = self.features(x)
        x = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        # FP32 probability normalization prevents FP16 loss underflow.
        return F.normalize(self.classifier(x).float(), p=1, dim=1)


def age_model(pretrained=False):
    # Import only Cadene's original SENet module, avoiding obsolete package imports.
    source = ROOT / "_deps/pretrainedmodels/models/senet.py"
    spec = importlib.util.spec_from_file_location("training_senet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.se_resnext50_32x4d(pretrained=None)
    if pretrained:
        # The original lip6 host currently fails TLS / HTTP 502 here. MONAI mirrors the identical file.
        path = ROOT/'artifacts/torch_cache/hub/checkpoints/se_resnext50_32x4d-a260b3a4.pth'
        digest = 'a260b3a40f82dfe37c58d26a612bcf7bef0d27c6fed096226b0e4e9fb364168e'
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                'https://huggingface.co/datasets/MONAI/testing_data/resolve/main/se_resnext50_32x4d-a260b3a4.pth',
                path, hash_prefix=digest)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'ImageNet SENet weight checksum mismatch: {path}')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    model.last_linear = nn.Linear(model.last_linear.in_features, 101)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def score_outputs(probabilities):
    ranks = torch.tensor([1.05 + 0.1*i for i in range(40)], device=probabilities.device)
    # Exactly the upstream [5, 10, 10, 10, 5] aggregation, including its tie convention.
    levels = torch.zeros(probabilities.shape[0], 5, dtype=torch.float32, device=probabilities.device)
    indices = torch.tensor([0]*5 + [1]*10 + [2]*10 + [3]*10 + [4]*5, device=probabilities.device)
    levels.index_add_(1, indices, probabilities)
    return (probabilities * ranks).sum(1), levels


def fap_loss(probabilities, target, mean, levels, alpha=1., beta=1., gamma=1., score_loss="exp"):
    score, predicted_levels = score_outputs(probabilities)
    fine = torch.pow(torch.sum(torch.pow(probabilities-target, 2), dim=1), .5).mean()
    coarse = torch.pow(torch.sum(torch.pow(predicted_levels-levels, 2), dim=1), .5).mean()
    expectation = (torch.exp(torch.abs(score-mean))-1).sum() if score_loss == "exp" else F.l1_loss(score, mean)
    # Upstream accumulates in-place into the FP32 distribution loss, even when
    # DataLoader collates Python mean-score labels as float64.
    total = alpha*fine
    total += beta*coarse
    total += gamma*expectation
    return total


def make_optimizer(model, name):
    if name == "2dfap":
        return torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                 weight_decay=.01, amsgrad=True)
    return torch.optim.Adam(model.parameters(), lr=.001)
