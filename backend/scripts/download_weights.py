"""Download public model assets and fixtures into backend/weights/. Run before offline service startup."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import WEIGHTS
import requests

MODEL_REVISION = "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
MODEL_URL = f"https://huggingface.co/OpenGVLab/InternVL3-2B-hf/resolve/{MODEL_REVISION}"
RELEASE_URL = "https://github.com/gengyan152/suixin-facial-analysis/releases/download/v1.2"
MODEL_FILES = (
    "added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json",
    "merges.txt", "model.safetensors", "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
)
EXPECTED_SHA256 = {
    "blaze_face_short_range.tflite":"b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    "face_landmarker.task":"64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "face_detection_yunet_2023mar.onnx":"8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "InternVL3-2B-hf/model.safetensors": "8e2c302719a13916a4d276e60fad4fb1c92a57b39dd2ad51af5a63dcf7a16f4a",
    "InternVL3-2B-face-lora/adapter_config.json": "eb16f4b8cc107d22fa0893d2f86a3f8026462126b7b7cdabca60b05dae159c0a",
    "InternVL3-2B-face-lora/adapter_model.safetensors": "714144a59af9b8ecca4841bce6bc19efdd7101a5a90589df85dc4c27ce659a1e",
    "age_release/epoch044_0.02343_3.9984.pth":"ae49c7f7d65ada34ac60adb610997e7cf267bf3072d9ee6549105f754c580807",
}
ASSETS = {
    **{f"InternVL3-2B-hf/{name}": f"{MODEL_URL}/{name}" for name in MODEL_FILES},
    "InternVL3-2B-face-lora/adapter_config.json": f"{RELEASE_URL}/adapter_config.json",
    "InternVL3-2B-face-lora/adapter_model.safetensors": f"{RELEASE_URL}/adapter_model.safetensors",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "warmup.jpg": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png",
    "fixtures/dog.jpg":"https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "fixtures/grace_hopper.jpg":"https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg",
    "fixtures/cat.png":"https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/chelsea.png",
    "upstream/MobileNetv2.py":"https://raw.githubusercontent.com/enquan/2D_FAP/main/models/MobileNetv2.py",
}


def download(item):
    name, url = item
    path = WEIGHTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        if name == "InternVL3-2B-hf/model.safetensors":
            # Some local proxies buffer whole large objects; bounded HTTP ranges
            # also make retries resume rather than restart the full checkpoint.
            offset = part.stat().st_size if part.exists() else 0
            total = offset+1
            while offset < total:
                for attempt in range(4):
                    try:
                        response = requests.get(url+f"?download=true&chunk={offset}",headers={"Range":f"bytes={offset}-{offset+8*1024*1024-1}"},timeout=(20,90))
                        response.raise_for_status()
                        if response.status_code != 206 or not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                            raise ValueError("Server did not honor requested byte range")
                        total = int(response.headers["Content-Range"].split("/")[-1])
                        break
                    except requests.RequestException:
                        if attempt==3:
                            raise
                        time.sleep(1)
                with part.open("ab") as output:
                    output.write(response.content)
                offset += len(response.content)
                if offset%(64*1024*1024)==0 or offset==total:
                    print(f"DOWNLOAD {name}: {offset}/{total}",flush=True)
        else:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with part.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
    candidate = path if path.is_file() else part
    with candidate.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if name in EXPECTED_SHA256 and digest != EXPECTED_SHA256[name]:
        raise ValueError(f"SHA-256 mismatch for {name}; inspect the file before replacing it")
    if candidate != path:
        candidate.replace(path)
    print(f"READY {name} {path.stat().st_size} {digest}", flush=True)
    return name, {"url": url, "sha256": digest, "bytes": path.stat().st_size}


def main():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    assets = dict(ASSETS)
    # Public release assets, rather than an assumed filename or an ImageNet checkpoint.
    release = requests.get("https://github.com/yu4u/age-estimation-pytorch/releases/expanded_assets/v1.0", timeout=60)
    release.raise_for_status()
    links = re.findall(r'href="([^" ]+/releases/download/[^" ]+)"', release.text)
    for link in links:
        if any(link.endswith(ext) for ext in (".pth", ".pt")):
            assets["age_release/" + link.rsplit("/", 1)[1]] = "https://github.com" + link
    print("Age release assets:", links, flush=True)
    results, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, item): item[0] for item in assets.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                name, data = future.result()
                results[name] = data
            except Exception as error:
                failures[futures[future]] = str(error)
                print("FAILED", futures[future], str(error), flush=True)
    (WEIGHTS / "manifest.json").write_text(json.dumps({"internvl_revision": MODEL_REVISION, "files": results, "failures": failures}, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
