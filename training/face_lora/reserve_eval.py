"""Generate replacement train/dev manifests while preserving existing image paths."""

import argparse
from collections import Counter
import json
from pathlib import Path

from common import DuplicateIndex, FIELD_NAMES, HASH_KEYS, fingerprint, open_rgb, read_rows, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reserve-id", action="append", default=[])
    parser.add_argument("--reserve-image", metavar="ID=PATH", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() or not (args.reserve_id or args.reserve_image):
        parser.error("Use a new output directory and specify held-out samples")

    manifests = {split: list(read_rows(args.data / f"{split}.jsonl")) for split in ("train", "dev", "test")}
    all_rows = {row["id"]: row for rows in manifests.values() for row in rows}
    heldout = list(read_rows(args.data / "heldout_index.jsonl"))
    existing_test_ids = {row["id"] for row in heldout if row["split"] == "test"}
    reserved = []
    for reserved_id in args.reserve_id:
        if reserved_id not in existing_test_ids:
            row = all_rows[reserved_id]
            reserved.append({"id": reserved_id, "split": "test", **{key: row[key] for key in HASH_KEYS if key in row}})
    for value in args.reserve_image:
        identity, separator, filename = value.partition("=")
        if not separator or not identity or not filename:
            parser.error("--reserve-image expects ID=PATH")
        path = Path(filename)
        reserved.append({"id": identity, "split": "test", **fingerprint(open_rgb(path), path.read_bytes())})
    guard = DuplicateIndex()
    for row in reserved:
        guard.add(row)

    summary = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    summary["regression_excluded"] = {}
    summary["regression_reserved_ids"] = sorted(set(args.reserve_id) | {row["id"] for row in reserved})
    summary["excluded"]["regression_duplicate"] = 0
    args.output.mkdir(parents=True)
    for split in ("train", "dev"):
        kept = []
        rejected = []
        for row in manifests[split]:
            match = guard.find(row)
            if match:
                rejected.append({"id": row["id"], "duplicate_of": match})
            else:
                kept.append(row)
        summary["regression_excluded"][split] = rejected
        summary["excluded"]["regression_duplicate"] += len(rejected)
        summary["counts"][split] = len(kept)
        summary["field_counts"][split] = {
            key: dict(Counter(row["field_codes"][key] for row in kept)) for key in FIELD_NAMES
        }
        write_rows(args.output / f"{split}.jsonl", kept)

    write_rows(args.output / "heldout_index.jsonl", [*heldout, *reserved])
    (args.output / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("counts", "regression_excluded", "regression_reserved_ids")}))


if __name__ == "__main__":
    main()
