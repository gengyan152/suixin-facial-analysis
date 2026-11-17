"""Shared reproducible runtime; all downloads and local dependencies stay here."""
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_deps"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "artifacts" / "torch_cache"))
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, batch, workers, shuffle=False, seed=20, drop_last=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0,
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_weights(model, path):
    """Tensor-only state_dict payload in the agreed checkpoint envelope."""
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_weights(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)


def verify_roundtrip(model, factory, path, inputs, output):
    import hashlib
    model.eval()
    with torch.no_grad():
        expected = model(inputs).detach().cpu()
    restored = factory().to(inputs.device).eval()
    load_weights(restored, path, inputs.device)
    with torch.no_grad():
        actual = restored(inputs).detach().cpu()
    result = {'weights_only': True, 'strict': True, 'exact_equal': torch.equal(expected, actual),
              'max_abs_error': (expected-actual).abs().max().item(), 'shape': list(actual.shape),
              'output_sha256': hashlib.sha256(actual.numpy().tobytes()).hexdigest()}
    write_json(output, result)
    if not result['exact_equal']:
        raise AssertionError(f'Checkpoint roundtrip mismatch: {result}')
    del restored
    return result


def autocast(precision, device="cuda"):
    return torch.autocast(device_type=device, enabled=precision != "fp32",
                          dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def regression_metrics(target, predicted):
    target, predicted = np.asarray(target), np.asarray(predicted)
    pc = float(np.corrcoef(target, predicted)[0, 1]) if np.std(target) > 0 and np.std(predicted) > 0 else None
    return {"n": len(target), "PC": pc, "MAE": float(np.abs(target-predicted).mean()),
            "RMSE": float(np.sqrt(np.square(target-predicted).mean()))}
