"""Extract the official APPA-REAL release without unrestricted pickle loading."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import WEIGHTS
from scripts.download_weights import download
import torch


def main():
    name = "age_release/epoch044_0.02343_3.9984.pth"
    download((name, "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"))
    # Official checkpoint contains an argparse.Namespace with training options.
    # Only allow this specific inert metadata type, keeping weights_only=True.
    import argparse
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(WEIGHTS / name, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key,value in checkpoint["state_dict"].items()}
    if state["last_linear.weight"].shape != (101,2048):
        raise ValueError("Unexpected age architecture")
    torch.save(state, WEIGHTS / "appa_real_se_resnext50_state_dict.pt")
    print("Extracted official 101-class APPA-REAL state_dict", flush=True)


if __name__ == "__main__":
    main()
