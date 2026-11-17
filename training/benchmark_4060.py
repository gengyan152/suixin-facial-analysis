"""GPU-resident synthetic full-architecture training benchmark, isolated per case."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import ROOT, autocast, seed_everything, write_json
import torch
from models import TwoDFAP, age_model, fap_loss, make_optimizer


def measure(args):
    seed_everything(args.seed, deterministic=False)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    model = (TwoDFAP() if args.model == "2dfap" else age_model()).to(device).train()
    optimizer = make_optimizer(model, args.model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16", init_scale=128.)
    x = torch.randn(args.batch, 3, 224, 224, device=device) if args.model == "2dfap" else torch.rand(args.batch, 3, 224, 224, device=device)*255
    fine = torch.full((args.batch, 40), 1/40, device=device)
    coarse = torch.softmax(torch.randn(args.batch, 5, device=device), 1)
    means = torch.linspace(2., 4., args.batch, device=device, dtype=torch.float64)
    ages = torch.arange(args.batch, device=device) % 101
    row = {"model": args.model, "precision": args.precision, "batch": args.batch,
           "warmup": args.warmup, "steps": args.steps, "gpu": torch.cuda.get_device_name(),
           "torch": str(torch.__version__), "cuda": torch.version.cuda,
           "parameters": sum(p.numel() for p in model.parameters()),
           "free_before_mib": free/2**20, "total_mib": total/2**20,
           "timestamp": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
           "tf32": False, "cudnn_benchmark": False, "input": "GPU-resident synthetic NCHW"}
    row['nvidia_smi_start'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,power.draw,power.limit,temperature.gpu', '--format=csv'], capture_output=True, text=True).stdout.strip()
    row['pid'] = os.getpid()
    row['gpu_processes_start'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'], capture_output=True, text=True).stdout.strip()
    times = []
    skipped = 0
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.precision):
            output = model(x)
            loss = fap_loss(output, fine, means, coarse) if args.model == "2dfap" else torch.nn.functional.cross_entropy(output.float(), ages)
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end.record()
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append(begin.elapsed_time(end))
            skipped += int(scaler.get_scale() < scale_before)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite loss: benchmark cannot be accepted")
        if (step+1) % 10 == 0:
            print(f'completed {step+1}/{args.warmup+args.steps} steps', flush=True)
    elapsed = time.perf_counter()-start
    row.update(status="ok", seconds=elapsed, images_per_second=args.steps*args.batch/elapsed,
               step_ms_mean=sum(times)/len(times), step_ms_min=min(times), step_ms_max=max(times),
               step_times_ms=times, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
               peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
               skipped_optimizer_steps=skipped, final_loss=loss.item())
    if skipped:
        raise RuntimeError(f"{skipped} measured optimizer updates skipped: invalid timing")
    model.eval()
    with torch.no_grad(), autocast(args.precision):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
    row["eval_images_per_second"] = 20*args.batch/(time.perf_counter()-start)
    row['nvidia_smi_end'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,power.draw,power.limit,temperature.gpu', '--format=csv'], capture_output=True, text=True).stdout.strip()
    row['gpu_processes_end'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'], capture_output=True, text=True).stdout.strip()
    epochs, n = (90, 4400) if args.model == "2dfap" else (80, 4113)
    row["train_hours_sample_scaled"] = n*epochs/row["images_per_second"]/3600
    row["train_hours_full_batch_steps"] = math.ceil(n/args.batch)*epochs*elapsed/args.steps/3600
    write_json(args.output, row)
    print(json.dumps(row, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--model", choices=["2dfap", "appa"], default="2dfap")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--models", nargs="+", default=["2dfap", "appa"])
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300, help="Per-case wall-clock cap; paging stalls are reported as timeout")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/benchmark_4060.json")
    args = parser.parse_args()
    if args.case:
        try:
            measure(args)
        except torch.OutOfMemoryError as error:
            write_json(args.output, {"model": args.model, "precision": args.precision, "batch": args.batch,
                                    "status": "oom", "error": str(error)})
            print(f"OOM: {args.model} {args.precision} batch={args.batch}", flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        for precision in args.precisions:
            for batch in args.batches:
                case = args.output.parent/f"{args.output.stem}_{name}_{precision}_b{batch}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--case", "--model", name,
                           "--precision", precision, "--batch", str(batch), "--warmup", str(args.warmup),
                           "--steps", str(args.steps), "--seed", str(args.seed), "--output", str(case)]
                print(f"RUN {name} {precision} batch={batch}", flush=True)
                with case.with_suffix('.log').open('w', encoding='utf-8') as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        row = {"model": name, "precision": precision, "batch": batch, "status": "timeout",
                               "timeout_seconds": args.timeout, "log": str(case.with_suffix('.log'))}
                    else:
                        if result.returncode:
                            row = {"model": name, "precision": precision, "batch": batch, "status": "error",
                                   "returncode": result.returncode, "log": str(case.with_suffix('.log'))}
                        else:
                            row = json.loads(case.read_text(encoding="utf-8"))
                rows.append(row)
                write_json(args.output, rows)
                print({k: v for k, v in row.items() if k in ("status", "images_per_second", "peak_allocated_mib", "log")}, flush=True)


if __name__ == "__main__":
    main()
