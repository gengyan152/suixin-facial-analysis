"""Download public model assets and fixtures into backend/weights/. Run before offline service startup."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import WEIGHTS
import requests

REVISION = "f253c5c2f3b6622998ba5b1ba2fb13dd33dc7474"
EXPECTED_SHA256 = {
    "blaze_face_short_range.tflite":"b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    "face_landmarker.task":"64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "face_detection_yunet_2023mar.onnx":"8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "FaceLLM-1B/model/model-00001-of-00002.safetensors":"fadee7acba57f151b9e4e2fdef2c7532507e3041db7800ec4ce7602a366ae990",
    "FaceLLM-1B/model/model-00002-of-00002.safetensors":"09a8c1fcdc08d64b2217d828518632fbf811d4d1168faf4c9a062a03bb5af5ff",
    "age_release/epoch044_0.02343_3.9984.pth":"ae49c7f7d65ada34ac60adb610997e7cf267bf3072d9ee6549105f754c580807",
}
ASSETS = {
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
        if name.endswith(".safetensors"):
            # Some local proxies buffer whole large objects; bounded HTTP ranges
            # also make retries resume rather than restart a 1 GB shard.
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
        part.replace(path)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if name in EXPECTED_SHA256 and digest != EXPECTED_SHA256[name]:
        raise ValueError(f"SHA-256 mismatch for {name}; inspect the file before replacing it")
    print(f"READY {name} {path.stat().st_size} {digest}", flush=True)
    return name, {"url": url, "sha256": digest, "bytes": path.stat().st_size}


def main():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    info = requests.get(f"https://huggingface.co/api/models/Idiap/FaceLLM-1B/revision/{REVISION}", timeout=60)
    info.raise_for_status()
    assets = dict(ASSETS)
    for file in info.json()["siblings"]:
        name = file["rfilename"]
        if name.startswith("model/"):
            assets["FaceLLM-1B/" + name] = f"https://huggingface.co/Idiap/FaceLLM-1B/resolve/{REVISION}/{name}"
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
    (WEIGHTS / "manifest.json").write_text(json.dumps({"facellm_revision": REVISION, "files": results, "failures": failures}, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
