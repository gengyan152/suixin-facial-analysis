"""Train a one-tile InternVL3-2B-hf language-attention LoRA from face descriptions."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, set_seed

from common import (
    DuplicateIndex, FIELD_NAMES, HASH_KEYS, MODEL_ID, MODEL_REVISION, fingerprint, open_rgb,
    read_rows, sample_prompt_and_target, sha256,
)


TARGETS = r"model\.language_model\.layers\.[0-9]+\.self_attn\.[qkvo]_proj"


class FaceCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.image_processor.max_patches = 1
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "right"
        self.assistant_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.image_token = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.assistant_prefix = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids

    def __call__(self, rows):
        messages = [
            [
                {"role": "user", "content": [
                    {"type": "image", "image": open_rgb(row["image"])},
                    {"type": "text", "text": row["prompt"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]},
            ]
            for row in rows
        ]
        batch = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
            padding=True,
        )
        if batch["pixel_values"].shape[0] != len(rows):
            raise ValueError("Expected one image tile per row")
        labels = torch.full_like(batch["input_ids"], -100)
        for index, tokens in enumerate(batch["input_ids"].tolist()):
            user_start, assistant_start = [position for position, token in enumerate(tokens) if token == self.assistant_start]
            start = assistant_start + len(self.assistant_prefix)
            end = tokens.index(self.eos, start)
            if (
                tokens[assistant_start:start] != self.assistant_prefix or start >= end
                or tokens[user_start:assistant_start].count(self.image_token) != self.processor.image_seq_length
                or self.processor.tokenizer.decode(tokens[start:end], skip_special_tokens=False) != rows[index]["target"]
            ):
                raise ValueError(f"{rows[index]['id']}: invalid prompt boundary, image tokens, or answer")
            labels[index, start:end + 1] = batch["input_ids"][index, start:end + 1]
        batch["labels"] = labels
        return batch


def input_records(manifest, index, *, extra=False, split=None, forbidden=None):
    kept = []
    excluded = []
    for row in read_rows(manifest):
        identity = row["id"]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{manifest}: expected a nonempty string id")
        if split is not None and row.get("split") != split:
            raise ValueError(f"{identity}: manifest contains a row outside {split}")
        path = Path(row["image"])
        if path.is_absolute():
            raise ValueError(f"{identity}: image path must be relative to {manifest.name}")
        path = (manifest.parent / path).resolve()
        if extra and "fields" not in row:
            answer = row["answer"]
            row["fields"] = json.loads(answer) if isinstance(answer, str) else answer
            prompt, target = sample_prompt_and_target(row)
            if not isinstance(row["prompt"], str) or not row["prompt"].strip() or len(row["prompt"]) > 400:
                raise ValueError(f"{identity}: invalid part-specific prompt")
            prompt = row["prompt"]
        else:
            prompt, target = sample_prompt_and_target(row)
        if any(marker in prompt or marker in target for marker in ("<|im_", "<IMG_CONTEXT>", "<image>", "<video>")):
            raise ValueError(f"{identity}: reserved chat marker in sample text")

        hashes = fingerprint(open_rgb(path), path.read_bytes())
        for key, value in hashes.items():
            if key in row and row[key] != value:
                raise ValueError(f"{identity}: {key} does not match image file")
        for key, length in (("source_sha256", 64), ("source_phash", 16), ("source_phash_mirror", 16)):
            if key in row and not re.fullmatch(f"[0-9a-f]{{{length}}}", row[key]):
                raise ValueError(f"{identity}: invalid {key} format")
        entry = {key: row[key] for key in HASH_KEYS if key in row}
        entry.update(hashes, id=identity)
        if forbidden is not None:
            blocked_by = forbidden.find(entry)
            if blocked_by:
                excluded.append({"id": identity, "duplicate_of": blocked_by, "manifest": str(manifest)})
                continue
        duplicate = index.find(entry)
        if duplicate:
            if not extra:
                raise ValueError(f"{identity}: prepared {split} manifest conflicts with {duplicate}")
            excluded.append({"id": identity, "duplicate_of": duplicate, "manifest": str(manifest)})
            continue
        index.add(entry)
        kept.append({
            "id": identity, "image": path, "prompt": prompt, "target": target,
            "source": row.get("source", manifest.stem), "field_count": len(row["fields"]),
        })
    return kept, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with train.jsonl, dev.jsonl and heldout_index.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID, help="Pinned Hub model or a local snapshot")
    parser.add_argument("--extra-train", type=Path, action="append", default=[])
    parser.add_argument("--extra-heldout", type=Path, action="append", default=[], help="Exclude photos in external eval manifests")
    parser.add_argument("--max-steps", type=int, help="Run a shortened smoke test")
    parser.add_argument("--train-samples", type=int, help="Cap the shuffled train set before training")
    parser.add_argument("--dev-samples", type=int, help="Cap the seeded dev set before evaluation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--check-batch", action="store_true", help="Audit all manifests and encode a real batch on CPU")
    args = parser.parse_args()
    if min(args.batch_size, args.gradient_accumulation, args.save_steps) < 1:
        parser.error("Batch size, gradient accumulation and save steps must be positive")
    if any(value is not None and value < 1 for value in (args.max_steps, args.train_samples, args.dev_samples)):
        parser.error("Set positive limits for smoke training and evaluation")
    if args.output.exists():
        parser.error("Choose an unused output path")

    train_path, dev_path, holdout_path = (args.data / f"{name}.jsonl" for name in ("train", "dev", "heldout_index"))
    holdout_rows = list(read_rows(holdout_path))
    if not holdout_rows or not any(row["split"] == "test" for row in holdout_rows):
        raise ValueError("Held-out test fingerprints are required")
    index = DuplicateIndex()
    for row in holdout_rows:
        if row["split"] == "test":
            index.add(row)
    external_holdout = None
    if args.extra_heldout:
        external_holdout = DuplicateIndex()
        for manifest in args.extra_heldout:
            input_records(manifest, external_holdout, extra=True)
    dev, excluded_dev = input_records(dev_path, index, split="dev", forbidden=external_holdout)
    for row in holdout_rows:
        if row["split"] == "dev":
            index.add(row)
    train, excluded_train = input_records(train_path, index, split="train", forbidden=external_holdout)
    rejected = []
    for manifest in args.extra_train:
        rows, excluded = input_records(manifest, index, extra=True, forbidden=external_holdout)
        train.extend(rows)
        rejected.extend(excluded)

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed).shuffle(dev)
    train = train[:args.train_samples]
    dev = dev[:args.dev_samples]
    if not train or not dev:
        raise ValueError("Training and validation both need at least one sample")
    max_epoch_steps = math.ceil(len(train) / (args.batch_size * args.gradient_accumulation))
    if args.max_steps is not None and args.max_steps > max_epoch_steps:
        parser.error("A smoke run cannot exceed one epoch")

    args.output.mkdir(parents=True)
    stats = {
        "base_model": MODEL_ID, "base_revision": MODEL_REVISION, "seed": args.seed,
        "train": len(train), "dev": len(dev), "heldout_reference_rows": len(holdout_rows),
        "train_sources": dict(Counter(row["source"] for row in train)),
        "train_field_counts": dict(Counter(row["field_count"] for row in train)),
        "excluded_extra_duplicates": rejected,
        "excluded_scut_as_external_heldout": [*excluded_dev, *excluded_train],
        "manifest_sha256": {
            str(path): sha256(path.read_bytes())
            for path in [train_path, dev_path, holdout_path, *args.extra_train, *args.extra_heldout]
        },
        "parameters": {
            "rank": 16, "alpha": 32, "dropout": 0.05, "image_tiles": 1, "epochs": 1,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size, "max_steps": args.max_steps,
            "gradient_accumulation": args.gradient_accumulation,
            "gradient_checkpointing": args.gradient_checkpointing, "targets": TARGETS,
        },
        "train_ids": [row["id"] for row in train], "dev_ids": [row["id"] for row in dev],
    }
    print(json.dumps({
        key: stats[key] for key in ("train", "dev", "heldout_reference_rows", "train_sources", "train_field_counts")
    }, ensure_ascii=False), flush=True)
    model_path = Path(args.model)
    if args.model != MODEL_ID and not model_path.is_dir():
        raise ValueError("Provide the pinned model identifier or a valid local snapshot")
    model_kwargs = {"local_files_only": True} if model_path.is_dir() else {"revision": MODEL_REVISION}
    collate = FaceCollator(AutoProcessor.from_pretrained(args.model, **model_kwargs))
    if args.check_batch:
        # Exercise both forms when an optional single-field corpus is present.
        probes = [train[0]]
        other = next((row for row in train if row["field_count"] != train[0]["field_count"]), None)
        if other is not None:
            probes.append(other)
        else:
            probes.append(dev[0])
        batch = collate(probes)
        stats["batch_check"] = {
            "input_shape": list(batch["input_ids"].shape), "image_shape": list(batch["pixel_values"].shape),
            "supervised_tokens": batch["labels"].ne(-100).sum(dim=1).tolist(),
        }
        (args.output / "training_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(stats["batch_check"]), flush=True)
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with BF16 support")
    from peft import LoraConfig, get_peft_model

    set_seed(args.seed)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", **model_kwargs
    )
    expected_layers = model.config.text_config.num_hidden_layers
    selected = [name for name, module in model.named_modules() if re.fullmatch(TARGETS, name) and isinstance(module, torch.nn.Linear)]
    if model.config.model_type != "internvl" or expected_layers != 28 or len(selected) != 4 * expected_layers:
        raise ValueError("Unexpected InternVL model architecture or LoRA target modules")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable({"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=TARGETS,
    ))
    adapter_config = model.peft_config["default"]
    adapter_config.base_model_name_or_path = MODEL_ID
    adapter_config.revision = MODEL_REVISION
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any(".language_model." not in name or ".lora_" not in name for name in trainable):
        raise ValueError("Only language-attention LoRA parameters may be trained")
    model.print_trainable_parameters()
    stats["trainable_parameters"], stats["total_parameters"] = model.get_nb_trainable_parameters()
    stats["gpu"] = torch.cuda.get_device_name()
    stats["torch"] = torch.__version__
    stats["target_modules"] = selected

    trainer = Trainer(
        model=model, train_dataset=train, eval_dataset=dev, data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"), num_train_epochs=1,
            max_steps=args.max_steps or -1, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
            gradient_checkpointing=False, learning_rate=args.learning_rate, optim="adamw_torch",
            lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, tf32=True,
            logging_steps=1 if args.max_steps else 10, logging_first_step=True, logging_nan_inf_filter=False,
            save_strategy="no" if args.max_steps else "steps", save_steps=args.save_steps, save_total_limit=2,
            remove_unused_columns=False, label_names=["labels"], prediction_loss_only=True, report_to=[], seed=args.seed,
        ),
    )
    stats["dev_before"] = trainer.evaluate(metric_key_prefix="dev_before")
    if not math.isfinite(stats["dev_before"]["dev_before_loss"]):
        raise RuntimeError("Base model produced non-finite development loss")
    result = trainer.train()
    if not math.isfinite(result.training_loss) or result.global_step < 1:
        raise RuntimeError("Training did not produce finite optimizer updates")
    stats["training"] = result.metrics
    stats["optimizer_steps"] = result.global_step
    stats["dev_after"] = trainer.evaluate(metric_key_prefix="dev_after")
    if not math.isfinite(stats["dev_after"]["dev_after_loss"]):
        raise RuntimeError("Adapter produced non-finite development loss")
    trainer.save_model(str(args.output / "adapter"))
    stats["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    stats["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    stats["wall_seconds"] = time.monotonic() - started
    stats["history"] = trainer.state.log_history
    (args.output / "training_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
