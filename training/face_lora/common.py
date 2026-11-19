import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MODEL_ID = "OpenGVLab/InternVL3-2B-hf"
MODEL_REVISION = "cb57a075cb75a2e6d1b668b128d48bb00ae321d2"
FIELD_NAMES = {
    "brows": "眉毛", "eyes": "眼睛", "nose": "鼻子",
    "mouth": "嘴唇", "contour": "脸型", "expression": "可见表情",
}
TEXT_PROMPT = (
    "请用简体中文描述照片中的五官。每个部位写一句话，说清看得见的外形。看不清的部位写 null。"
    "只输出一个 JSON 对象，六个键分别是 brows（眉毛）、eyes（眼睛）、nose（鼻子）、"
    "mouth（嘴唇）、contour（脸型）、expression（可见表情）。"
    "眉部写双眼上方的眉毛。可见表情写嘴角或眉间的动作。"
    "省略审美评价以及对身份、年龄、性别、族裔、健康、情绪和性格的推测。"
)
HASH_KEYS = ("sha256", "source_sha256", "phash", "phash_mirror", "source_phash", "source_phash_mirror")
PHASH_RADIUS = 4
DCT = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32) + 0.5))
DCT[0] /= np.sqrt(2.0)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def phash(image):
    grayscale = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    hashes = []
    for view in (grayscale, grayscale[:, ::-1]):
        frequencies = (DCT @ view @ DCT.T).ravel()[1:]
        code = 0
        for bit in frequencies > np.median(frequencies):
            code = (code << 1) | int(bit)
        hashes.append(f"{code:016x}")
    return hashes


def fingerprint(image, contents, source=None, source_contents=None):
    forward, mirror = phash(image)
    hashes = {"sha256": sha256(contents), "phash": forward, "phash_mirror": mirror}
    if source is not None:
        forward, mirror = phash(source)
        hashes.update(source_sha256=sha256(source_contents), source_phash=forward, source_phash_mirror=mirror)
    return hashes


class DuplicateIndex:
    def __init__(self):
        self.ids = set()
        self.exact = {}
        self.bands = {}

    def find(self, row):
        if row["id"] in self.ids:
            return row["id"]
        for key in HASH_KEYS[:2]:
            if row.get(key) in self.exact:
                return self.exact[row[key]]
        for key in HASH_KEYS[2:]:
            if key not in row:
                continue
            code = int(row[key], 16)
            # Within four bit errors, at least one of these five disjoint bands is identical.
            for part in range(PHASH_RADIUS + 1):
                for previous, sample_id in self.bands.get((part, (code >> (13 * part)) & 8191), ()):
                    if (code ^ previous).bit_count() <= PHASH_RADIUS:
                        return sample_id
        return None

    def add(self, row):
        self.ids.add(row["id"])
        for key in HASH_KEYS[:2]:
            if key in row:
                self.exact.setdefault(row[key], row["id"])
        for code in {int(row[key], 16) for key in HASH_KEYS[2:] if key in row}:
            for part in range(PHASH_RADIUS + 1):
                self.bands.setdefault((part, (code >> (13 * part)) & 8191), []).append((code, row["id"]))


def sample_prompt_and_target(row):
    fields = row["fields"]
    if len(fields) == 6:
        if set(fields) != set(FIELD_NAMES):
            raise ValueError(f"{row['id']}: six-field sample has unknown keys")
        ordered = {key: fields[key] for key in FIELD_NAMES}
        prompt = TEXT_PROMPT
    elif len(fields) == 1:
        field, = fields
        ordered = fields
        prompt = (
            f"请用简体中文写一句话，描述照片中{FIELD_NAMES[field]}看得见的外形。"
            f"只输出一个 JSON 对象，键为 {field}。省略审美评价和对人物的推测。"
        )
    else:
        raise ValueError(f"{row['id']}: expected one or six fields")

    def short_observation(value):
        return (
            isinstance(value, str) and 1 <= len(value) <= 64 and len(value.splitlines()) == 1
            and any("\u4e00" <= char <= "\u9fff" for char in value)
            and not any(char in value[:-1] for char in "。！？!?")
        )

    if all(value is None for value in ordered.values()) or any(
        value is not None and not short_observation(value) for value in ordered.values()
    ):
        raise ValueError(f"{row['id']}: empty or invalid observation")
    return prompt, json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def open_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")
