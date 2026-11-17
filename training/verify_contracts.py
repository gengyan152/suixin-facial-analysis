"""CPU migration parity and input/geometry contracts; no real dataset required."""
import importlib.util
import argparse
import json
from pathlib import Path
import types
from runtime import ROOT, seed_everything, write_json
import cv2
import numpy as np
import torch
from torch import nn
from datasets import APPADataset, appa_augmentation, fine_target
from models import TwoDFAP, age_model, score_outputs, fap_loss
from geometry import LANDMARKS, measure_geometry


def fixture_landmarks():
    points = np.full((478, 2), [150., 180.])
    oval = LANDMARKS['face_oval']
    angle = np.linspace(-np.pi/2, 3*np.pi/2, len(oval), endpoint=False)
    points[oval] = np.stack([150+100*np.cos(angle), 180+140*np.sin(angle)], 1)
    overrides = {33: [80, 140], 133: [125, 140], 362: [175, 140], 263: [220, 140],
                 159: [102, 133], 145: [102, 147], 386: [198, 133], 374: [198, 147],
                 52: [100, 115], 65: [110, 115], 282: [200, 115], 295: [190, 115],
                 9: [150, 110], 2: [150, 200], 152: [150, 320], 168: [150, 120], 6: [150, 160],
                 98: [128, 190], 327: [172, 190], 0: [150, 226], 13: [150, 234],
                 14: [150, 237], 17: [150, 249], 61: [118, 237], 291: [182, 237]}
    for index, value in overrides.items():
        points[index] = value
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    seed_everything(20)
    result = {'device': args.device, 'torch': str(torch.__version__), 'reference_environment': 'upstream source running on modern torch; no legacy torch 1.8 environment'}
    # Only replace the removed import; preserve the full upstream model body.
    source = (ROOT/'vendor/2D_FAP/models/MobileNetv2.py').read_text(encoding='utf-8')
    source = source.replace('from torchvision.models.utils import load_state_dict_from_url', 'from torch.hub import load_state_dict_from_url')
    namespace = {}
    exec(compile(source, 'upstream_MobileNetv2.py', 'exec'), namespace)
    reference = namespace['MobileNetV2']()
    reference.classifier[1] = nn.Linear(1280, 40)
    migrated = TwoDFAP()
    migrated.load_state_dict(reference.state_dict(), strict=True)
    reference.to(args.device).eval()
    migrated.to(args.device).eval()
    x = torch.randn(2, 3, 224, 224).to(args.device)
    with torch.no_grad():
        a, b = reference(x), migrated(x)
    assert torch.equal(a, b), (a-b).abs().max().item()
    targets = torch.stack([fine_target(2.5, .7), fine_target(3.4, .5)]).to(args.device)
    means = torch.tensor([2.5, 3.4], device=args.device, dtype=torch.float64)
    levels = torch.tensor([[.1, .2, .3, .3, .1], [.05, .1, .2, .4, .25]], device=args.device)
    rank = torch.tensor(np.arange(1.05, 5.05, .1), dtype=torch.float32, device=args.device)
    plevel = torch.zeros(2, 5, device=args.device)
    idx = torch.repeat_interleave(torch.arange(5), torch.tensor([5, 10, 10, 10, 5])).to(args.device)
    plevel.index_add_(1, idx, a)
    ref_loss = torch.pow(torch.sum((a-targets)**2, dim=1), .5).mean()
    ref_loss += torch.pow(torch.sum((plevel-levels)**2, dim=1), .5).mean()
    ref_loss += (torch.exp(torch.abs((a*rank).sum(1)-means))-1).sum()
    new_loss = fap_loss(b, targets, means, levels)
    assert torch.equal(ref_loss, new_loss), (ref_loss.item(), new_loss.item())
    score, coarse = score_outputs(b)
    assert torch.equal(plevel, coarse)
    assert torch.equal(score, (a*rank).sum(1))
    result['2dfap'] = {'vector_exact': True, 'loss_exact': True, 'expectation_exact': True,
                        'aggregation_exact': True, 'max_abs_error': 0., 'loss': new_loss.item()}
    # Execute the official factory using its original SENet dependency, without unrelated obsolete imports.
    spec = importlib.util.spec_from_file_location('reference_senet', ROOT/'_deps/pretrainedmodels/models/senet.py')
    senet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(senet)
    namespace = {'pretrainedmodels': types.SimpleNamespace(se_resnext50_32x4d=senet.se_resnext50_32x4d)}
    source = (ROOT/'vendor/age-estimation-pytorch/model.py').read_text(encoding='utf-8')
    source = source.replace('import pretrainedmodels.utils', '').replace('import pretrainedmodels', '')
    exec(compile(source, 'upstream_age_model.py', 'exec'), namespace)
    reference = namespace['get_model'](pretrained=None).to(args.device).eval()
    migrated = age_model().to(args.device).eval()
    migrated.load_state_dict(reference.state_dict(), strict=True)
    x = (torch.rand(2, 3, 224, 224)*255).to(args.device)
    with torch.no_grad():
        a, b = reference(x), migrated(x)
    assert torch.equal(a, b)
    result['appa'] = {'logits_exact': True, 'probabilities_exact': torch.equal(a.softmax(1), b.softmax(1)), 'max_abs_error': 0.}
    fixture = ROOT/'artifacts/contracts_bgr.png'
    _, encoded = cv2.imencode('.png', np.full((224, 224, 3), [17, 99, 233], dtype=np.uint8))
    encoded.tofile(fixture)
    dataset = APPADataset([{'name': 'sentinel', 'path': str(fixture), 'mean': 22.7, 'std': 2.}])
    x, label, mean, _ = dataset[0]
    assert x.dtype == torch.float32 and x.shape == (3, 224, 224) and x[:, 0, 0].tolist() == [17., 99., 233.]
    assert label == 23 and abs(float(mean)-22.7) < 1e-5
    augmentation = appa_augmentation()
    augmentation.seed_(20)
    for _ in range(30):
        augmented = augmentation.augment_image(np.full((224, 224, 3), [17, 99, 233], dtype=np.uint8))
        assert augmented.shape == (224, 224, 3)
    points = fixture_landmarks()
    original = measure_geometry(points/[300, 400], 300, 400)
    angle = .21
    c, s = np.cos(angle), np.sin(angle)
    rotated = (points-[150, 180]) @ np.array([[c, s], [-s, c]]) + [150, 180]
    transformed = measure_geometry(rotated/[300, 400], 300, 400)
    error = max(abs(original[k]-transformed[k]) for k in original)
    assert error < 1e-12, error
    result['input_and_geometry'] = {'bgr_0_255_exact': True, 'rounded_label': label,
        'augmentation_draws': 30, 'roll_invariance_max_abs_error': error, 'non_square_image': [300, 400]}
    write_json(ROOT/f'artifacts/contracts_verification{"_cuda" if args.device == "cuda" else ""}.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
