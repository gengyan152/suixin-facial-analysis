"""Build six-field Chinese weak labels from SCUT photos and MediaPipe geometry."""

import argparse
from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image

from common import DuplicateIndex, FIELD_NAMES, HASH_KEYS, PHASH_RADIUS, fingerprint, open_rgb, read_rows, sha256, write_rows


PHRASES = {
    "brows": {
        "small_arch": "两侧眉线弯曲幅度较小。",
        "curved": "两侧眉毛中段明显拱起，眉尾回落。",
        "slightly_curved": "两侧眉线略带弧度。",
    },
    "eyes": {
        "narrow_opening": "两眼的眼裂偏细长。",
        "wide_opening": "两眼上下眼睑之间的开口较大。",
        "moderate_opening": "两眼的眼裂中间较宽，两端收窄。",
    },
    "nose": {
        "narrow_alar": "鼻翼宽度小于两内眼角之间的距离。",
        "wide_alar": "鼻翼宽度大于两内眼角之间的距离。",
        "similar_alar": "鼻翼宽度与两内眼角之间的距离接近。",
    },
    "mouth": {
        "thin_lips": "上下唇的唇红偏薄。",
        "full_lips": "唇红区域较厚，嘴唇显得饱满。",
        "upper_thinner": "上唇比下唇薄。",
        "lower_thinner": "下唇比上唇薄。",
        "similar_thickness": "上下唇的厚度接近。",
    },
    "contour": {
        "strong_taper": "脸颊向下颌明显收窄。",
        "slight_taper": "脸颊到下颌的宽度变化较小。",
        "moderate_taper": "脸颊较宽，轮廓向下颌收窄。",
    },
    "expression": {
        "raised_corners": "两侧嘴角上扬。",
        "open_lips": "嘴唇张开。",
        "parted_lips": "嘴唇微张。",
        "closed_lips": "嘴唇合拢。",
    },
}
OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
BROWS = ((70, 63, 105, 66, 107), (336, 296, 334, 293, 300))
EYES = ((33, 133, 159, 145), (362, 263, 386, 374))
NOSE = (98, 327, 133, 362)
MOUTH = (61, 291, 0, 13, 14, 17)
CONTOUR = (234, 454, 172, 397, 152)


def third(value, lower, upper, codes):
    if value < lower:
        return codes[0]
    if value > upper:
        return codes[2]
    return codes[1]


def describe_geometry(points, size, blend):
    # Rotate measurements to the eye-corner axis before comparing projected sizes.
    axis = points[263] - points[33]
    eye_span = np.linalg.norm(axis)
    if eye_span < 50:
        return None
    axis /= eye_span
    x = points @ axis
    y = points @ np.array([-axis[1], axis[0]])
    cheek_width = x[454] - x[234]
    mouth_width = x[291] - x[61]
    eye_gap = x[362] - x[133]
    if min(cheek_width, mouth_width, eye_gap) < 12:
        return None
    # Reject strongly asymmetric face projection. These comparisons assume a frontal face.
    side_a = x[1] - x[33]
    side_b = x[263] - x[1]
    if min(side_a, side_b) < 0.30 * eye_span:
        return None

    def on_image(indices):
        probe = points[list(indices)]
        return (probe >= 1).all() and (probe <= np.asarray(size) - 2).all()

    arches = []
    for start, a, b, c, end in BROWS:
        width = x[end] - x[start]
        if width < 10:
            return None
        internal = np.array([a, b, c])
        chord = y[start] + (x[internal] - x[start]) / width * (y[end] - y[start])
        arches.append(float(np.max(chord - y[internal]) / width))
    apertures = [(y[bottom] - y[top]) / max(1, x[end] - x[start]) for start, end, top, bottom in EYES]
    upper = y[13] - y[0]
    lower = y[17] - y[14]
    opening = float((y[14] - y[13]) / mouth_width)
    corner_raise = float(((y[0] + y[17]) - (y[61] + y[291])) / (2 * mouth_width))
    smile = min(blend["mouthSmileLeft"], blend["mouthSmileRight"])

    codes = dict.fromkeys(FIELD_NAMES)
    if on_image(sum(BROWS, ())) and abs(arches[0] - arches[1]) < 0.14:
        codes["brows"] = third(np.mean(arches), 0.137, 0.161, ("small_arch", "slightly_curved", "curved"))
    if on_image(sum(EYES, ())) and abs(apertures[0] - apertures[1]) < 0.16:
        codes["eyes"] = third(np.mean(apertures), 0.275, 0.36, ("narrow_opening", "moderate_opening", "wide_opening"))
    if on_image(NOSE):
        codes["nose"] = third((x[327] - x[98]) / eye_gap, 0.91, 1.09, ("narrow_alar", "similar_alar", "wide_alar"))
    if on_image(MOUTH) and min(upper, lower) >= 2:
        thickness = third(upper / lower, 0.83, 1.20, ("upper_thinner", "similar_thickness", "lower_thinner"))
        codes["mouth"] = third((upper + lower) / mouth_width, 0.25, 0.365, ("thin_lips", thickness, "full_lips"))
    if on_image(CONTOUR):
        codes["contour"] = third((x[397] - x[172]) / cheek_width, 0.76, 0.813, ("strong_taper", "moderate_taper", "slight_taper"))
    if on_image(MOUTH):
        if smile > 0.38 and corner_raise > 0.036:
            codes["expression"] = "raised_corners"
        else:
            codes["expression"] = third(opening, 0.026, 0.115, ("closed_lips", "parted_lips", "open_lips"))
    measurements = dict(
        brow_arch_to_width=float(np.mean(arches)), eye_opening_to_width=float(np.mean(apertures)),
        alar_to_inner_eye_gap=float((x[327] - x[98]) / eye_gap),
        upper_to_lower_lip=float(upper / lower) if lower > 0 else None, lip_thickness_to_width=float((upper + lower) / mouth_width),
        jaw_to_cheek_width=float((x[397] - x[172]) / cheek_width),
        lip_opening_to_width=opening, mouth_corner_raise_to_width=corner_raise, bilateral_smile_score=smile,
    )
    return codes, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Choose a new output folder: {args.output}")
    folds = {
        split: {line.split(maxsplit=1)[0] for line in (args.fold1 / f"{split}_1.txt").read_text().splitlines() if line.strip()}
        for split in ("train", "test")
    }
    if len(folds["train"]) != 4400 or len(folds["test"]) != 1100 or folds["train"] & folds["test"]:
        raise ValueError("Expected original 4,400/1,100 SCUT Fold 1 filenames")
    dev_names = set(random.Random(args.seed).sample(sorted(folds["train"]), 400))
    split_names = {"test": folds["test"], "dev": dev_names, "train": folds["train"] - dev_names}
    group = {name: split for split, names in split_names.items() for name in names}
    geometry = {row["filename"]: row for row in read_rows(args.geometry)}
    if set(geometry) != set(group):
        raise ValueError("Geometry filenames do not match all 5,500 Fold 1 images")

    splits = {split: [] for split in split_names}
    archived = []
    excluded = []
    duplicates = DuplicateIndex()
    for split, names in split_names.items():
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        split_archive = []
        for name in sorted(names):
            row_id = f"scut-fbp5500:{name}"
            g = geometry.pop(name)
            if g["status"] != "ok":
                excluded.append({"id": row_id, "split": split, "reason": g["status"]})
                continue
            path = args.images / name
            source_contents = path.read_bytes()
            image = open_rgb(path)
            if image.size != (g["width"], g["height"]):
                raise ValueError(f"{name}: dimensions differ from geometry input")
            points = np.asarray(g["landmarks"])[:, :2] * image.size
            if not np.isfinite(points).all():
                raise ValueError(f"{name}: invalid landmark coordinates")
            result = describe_geometry(points, image.size, g["blendshapes"])
            if result is None or sum(code is not None for code in result[0].values()) < 4:
                excluded.append({"id": row_id, "split": split, "reason": "insufficient_visible_geometry"})
                if split != "train":
                    split_archive.append({"id": row_id, "split": split, **fingerprint(image, source_contents)})
                continue
            codes, measurements = result
            contour = points[list(OVAL)]
            left, top = contour.min(axis=0)
            right, bottom = contour.max(axis=0)
            dx, dy = (right - left) * 0.16, (bottom - top) * 0.16
            bounds = [
                max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
                min(image.width, math.ceil(right + dx)), min(image.height, math.ceil(bottom + dy)),
            ]
            crop = image.crop(bounds)
            crop.thumbnail((448, 448), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            crop.save(buffer, format="JPEG", quality=92)
            contents = buffer.getvalue()
            # Compute the actual on-disk crop fingerprint after JPEG quantization.
            with Image.open(BytesIO(contents)) as decoded:
                hashes = fingerprint(decoded, contents, image, source_contents)
            index_entry = {"id": row_id, "split": split, **hashes}
            if split != "train":
                split_archive.append(index_entry)
            duplicate = duplicates.find(index_entry)
            if duplicate:
                excluded.append({"id": row_id, "split": split, "reason": "duplicate", "of": duplicate})
                continue
            duplicates.add(index_entry)
            relative = f"images/{split}/{name}"
            (args.output / relative).write_bytes(contents)
            splits[split].append({
                **index_entry, "image": relative, "source": "SCUT-FBP5500 original Images",
                "crop_bounds": bounds, "fields": {key: PHRASES[key].get(code) for key, code in codes.items()},
                "field_codes": codes, "measurements": measurements,
            })
        for index_entry in split_archive:
            if index_entry["id"] not in duplicates.ids:
                duplicates.add(index_entry)
        archived.extend(split_archive)

    for split, rows in splits.items():
        write_rows(args.output / f"{split}.jsonl", rows)
    write_rows(args.output / "heldout_index.jsonl", archived)
    write_rows(args.output / "excluded.jsonl", excluded)
    summary = {
        "split": "Official Fold 1 test set, 400 seeded dev filenames from Fold 1 train",
        "source": "https://github.com/HCIILAB/SCUT-FBP5500-Database-Release",
        "labels": "MediaPipe landmark geometry and bilateral mouthSmile coefficients",
        "seed": args.seed,
        "geometry_sha256": sha256(args.geometry.read_bytes()),
        "hashes": HASH_KEYS,
        "perceptual_duplicate_hamming_radius": PHASH_RADIUS,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "excluded": dict(Counter(row["reason"] for row in excluded)),
        "field_counts": {
            split: {key: dict(Counter(row["field_codes"][key] for row in rows)) for key in FIELD_NAMES}
            for split, rows in splits.items()
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
