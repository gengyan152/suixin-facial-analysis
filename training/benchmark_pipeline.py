"""Synthetic JPEG pipeline throughput including workers, augmentation and H2D copy."""
import argparse
import json
import os
from pathlib import Path
import time
import subprocess
from runtime import ROOT, autocast, loader, seed_everything, write_json
import numpy as np
from PIL import Image
import torch
from datasets import APPADataset, SCUTDataset
from models import TwoDFAP, age_model, fap_loss, make_optimizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['2dfap', 'appa'], required=True)
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    seed_everything(20)
    folder = ROOT/'artifacts/pipeline_fixtures'
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20)
    rows = []
    for index in range(256):
        path = folder/f'{index:04d}.jpg'
        if not path.is_file():
            Image.fromarray(rng.integers(0, 256, (256, 280, 3), dtype=np.uint8)).save(path, quality=90)
        rows.append({'name': path.name, 'path': str(path), 'mean': 2.+index/256*2 if args.model == '2dfap' else 18.+index/256*50,
                     'std': .7 if args.model == '2dfap' else 2., 'levels': np.asarray([.05, .15, .4, .3, .1], dtype=np.float32)})
    dataset = SCUTDataset(rows, True) if args.model == '2dfap' else APPADataset(rows, True)
    data = loader(dataset, args.batch, args.workers, True, 20, drop_last=True)
    iterator = iter(data)
    model = (TwoDFAP() if args.model == '2dfap' else age_model()).cuda().train()
    processes_start = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'], capture_output=True, text=True).stdout.strip()
    optimizer = make_optimizer(model, args.model)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16', init_scale=128.)
    samples, skipped = 0, 0
    for step in range(args.warmup+args.steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.precision):
            output = model(batch[0].cuda(non_blocking=True))
            if args.model == '2dfap':
                loss = fap_loss(output, batch[1].cuda(non_blocking=True), batch[2].cuda(non_blocking=True), batch[3].cuda(non_blocking=True))
            else:
                loss = torch.nn.functional.cross_entropy(output.float(), batch[1].cuda(non_blocking=True))
        previous = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite loss')
        if step >= args.warmup:
            samples += len(batch[0])
            skipped += int(scaler.get_scale() < previous)
        if (step+1) % 10 == 0:
            print(f'{args.model}: {step+1}/{args.warmup+args.steps}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    result = {'model': args.model, 'precision': args.precision, 'batch': args.batch,
              'workers': args.workers, 'warmup': args.warmup, 'steps': args.steps,
              'fixture_images': 256, 'synthetic_only': True, 'input': 'JPEG + original augmentation + H2D',
              'deterministic': True, 'seconds': elapsed, 'images_per_second': samples/elapsed,
              'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'skipped_optimizer_steps': skipped}
    result['pid'] = os.getpid()
    result['gpu_processes_start'] = processes_start
    result['gpu_processes_end'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'], capture_output=True, text=True).stdout.strip()
    if skipped:
        raise RuntimeError('Measured optimizer steps were skipped')
    write_json(args.output or ROOT/f'artifacts/pipeline_{args.model}.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
