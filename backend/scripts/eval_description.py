"""Compare local description models on identical held-out face crops."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.vlm import FIELD_LABELS, TEXT_PROMPT, TEXT_TOKENS, InternVL, parse_fields
from app.imaging import decode_image
import torch


VARIANTS = ("base", "lora")


def score_output(raw, phrases):
    try:
        pairs = json.loads(raw, object_pairs_hook=list)
        strict_json = (
            isinstance(pairs, list) and len(pairs) == len(FIELD_LABELS)
            and all(isinstance(pair, tuple) and (pair[1] is None or isinstance(pair[1], str)) for pair in pairs)
            and {key for key, _ in pairs} == FIELD_LABELS.keys()
        )
    except ValueError:
        strict_json = False
    try:
        fields = parse_fields(raw)
    except ValueError:
        fields = None

    # Score recognized reference templates separately from the human image audit.
    template_codes = {}
    for key, field in (fields or {}).items():
        if field["visible"]:
            content = compact(field["description"])
            codes = {code for phrase, code in phrases[key].items() if phrase in content}
            if len(codes) == 1:
                template_codes[key] = codes.pop()
    return {
        "strict_six_field_json": strict_json,
        "fields": fields,
        "template_codes": template_codes,
    }


def compact(text):
    return re.sub(r"[\s，。！？、；：,.!?;:]", "", text)


def summarize(records):
    visible = Counter()
    reference_codes = {key: Counter() for key in FIELD_LABELS}
    correct_codes = {key: Counter() for key in FIELD_LABELS}
    recognized_codes = {key: Counter() for key in FIELD_LABELS}
    for row in records:
        visible.update(key for key, field in (row["fields"] or {}).items() if field["visible"])
        for key, code in (row["reference_codes"] or {}).items():
            if code:
                reference_codes[key][code] += 1
                if code == row["template_codes"].get(key):
                    correct_codes[key][code] += 1
        for key, code in row["template_codes"].items():
            recognized_codes[key][code] += 1
    return {
        "images": len(records),
        "generation_errors": sum(row["error"] is not None for row in records),
        "strict_six_field_json": sum(row["strict_six_field_json"] for row in records),
        "app_parseable": sum(row["fields"] is not None for row in records),
        "visible_fields": sum(visible.values()),
        "total_fields": len(records) * len(FIELD_LABELS),
        "all_six_visible": sum(
            row["fields"] is not None and all(field["visible"] for field in row["fields"].values())
            for row in records
        ),
        "visible_by_field": dict.fromkeys(FIELD_LABELS, 0) | visible,
        "reference_template_match": {
            key: {
                "matches": sum(correct_codes[key].values()),
                "reference_slots": sum(ref.values()),
                "majority_guess_matches": max(ref.values()),
                "macro_class_recall": round(statistics.mean(correct_codes[key][code] / count for code, count in ref.items()), 4),
                "expected_by_code": ref,
                "matched_by_code": correct_codes[key],
                "recognized_by_code": recognized_codes[key],
            }
            for key, ref in reference_codes.items() if ref
        },
        "precision": records[0]["precision"],
        "gpu": records[0]["gpu"],
        "latency_ms_median": round(statistics.median(row["elapsed_ms"] for row in records), 1),
        "batch_peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in records),
        "adapter": records[0]["adapter"],
    }


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL records with id, image, sha256 and fields")
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[], help="Check for overlap with a train/dev manifest")
    parser.add_argument("--base-model", type=Path, help="Local InternVL3-2B-hf directory, defaults to backend/weights")
    parser.add_argument("--adapter", type=Path, help="Local directory with adapter config and safetensors")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--limit", type=int, help="Size of the deterministic held-out subset")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if "lora" in args.models and args.adapter is None:
        parser.error("--adapter is required to evaluate lora")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for local description evaluation")
    torch.set_num_threads(4)
    settings = Settings()
    base_model = args.base_model or settings.weights / "InternVL3-2B-hf"
    rows = sorted(
        (json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()),
        key=lambda row: row["id"],
    )
    if not rows or len({row["id"] for row in rows}) != len(rows):
        parser.error("manifest must contain unique sample IDs")
    phrases = {key: {} for key in FIELD_LABELS}
    for row in rows:
        for key, code in row.get("field_codes", {}).items():
            if code:
                phrases[key][compact(row["fields"][key])] = code
    if args.exclude_manifest:
        test_hashes = {row[key] for row in rows for key in ("sha256", "source_sha256")}
        for manifest in args.exclude_manifest:
            excluded = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if test_hashes.intersection(row[key] for row in excluded for key in ("sha256", "source_sha256")):
                parser.error(f"held-out manifest overlaps with {manifest}")
    if args.limit is not None:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    adapter = None
    if "lora" in args.models:
        adapter = {
            "path": str(args.adapter.resolve()),
            "config_sha256": file_hash(args.adapter / "adapter_config.json"),
            "weights_sha256": file_hash(args.adapter / "adapter_model.safetensors"),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    protocol = {
        "manifest_sha256": file_hash(args.manifest),
        "ids": [row["id"] for row in rows],
        "text_prompt": TEXT_PROMPT,
        "decoder": "greedy",
        "max_new_tokens": TEXT_TOKENS,
        "max_patches": 1,
        "timeout_s": settings.text_timeout,
        "seed": args.seed,
    }
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        parser.error("this output directory contains a different evaluation protocol")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    for variant in args.models:
        print(f"Loading {variant}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        gpu = torch.cuda.get_device_name(0)
        model = InternVL(settings, "cuda:0", base_model, args.adapter if variant == "lora" else None)
        pending = args.output / f"{variant}.jsonl.tmp"
        with pending.open("w", encoding="utf-8", buffering=1) as output:
            for index, row in enumerate(rows, 1):
                image_path = args.manifest.parent / row["image"]
                image_bytes = image_path.read_bytes()
                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError(f"Held-out crop hash mismatch: {image_path}")
                image = decode_image(image_bytes, settings).bgr
                start = time.monotonic()
                raw, error = "", None
                try:
                    raw = model.generate(image, TEXT_PROMPT, TEXT_TOKENS, settings.text_timeout)
                except (TimeoutError, RuntimeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                record = {
                    "id": row["id"], "crop": str(image_path.resolve()), "crop_sha256": digest,
                    "reference_fields": row["fields"], "reference_codes": row.get("field_codes"),
                    "raw": raw, "error": error, "elapsed_ms": round((time.monotonic() - start) * 1000, 1),
                    "precision": model.precision, "gpu": gpu,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "adapter": adapter if variant == "lora" else None,
                    **score_output(raw, phrases),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 10 == 0 or index == len(rows):
                    print(f"{variant} {index}/{len(rows)}", flush=True)
        pending.replace(args.output / f"{variant}.jsonl")
        del model
        torch.cuda.empty_cache()

    summary = {
        variant: summarize([json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()])
        for variant in VARIANTS if (result := args.output / f"{variant}.jsonl").is_file()
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
