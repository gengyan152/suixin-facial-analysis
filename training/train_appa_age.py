"""SE-ResNeXt-50 APPA-REAL training and rounded/float apparent-age evaluation."""
import argparse
import csv
from pathlib import Path
import time
from runtime import ROOT, autocast, loader, load_weights, save_weights, seed_everything, write_json, verify_roundtrip
import numpy as np
import torch
from datasets import APPADataset, read_appa
from models import age_model, make_optimizer


def evaluate(model, data, device, output=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, rounded, mean, names in data:
            probabilities = model(x.to(device)).float().softmax(1)
            ages = (probabilities*torch.arange(101, device=device)).sum(1)
            for name, gt, label, age, probs in zip(names, mean.tolist(), rounded.tolist(), ages.cpu().tolist(), probabilities.cpu().tolist()):
                rows.append({'filename': name, 'apparent_age_avg': gt, 'rounded_target': label, 'prediction': age,
                             **{f'age_{i}': value for i, value in enumerate(probs)}})
    if output:
        with Path(output).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {'n': len(rows), 'rounded_MAE': float(np.mean([abs(r['prediction']-r['rounded_target']) for r in rows])),
               'float_MAE': float(np.mean([abs(r['prediction']-r['apparent_age_avg']) for r in rows])), 'age_bands': {}}
    for lo, hi in [(0, 13), (13, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 101)]:
        group = [r for r in rows if lo <= r['apparent_age_avg'] < hi]
        summary['age_bands'][f'[{lo},{hi})'] = {'n': len(group),
            'float_MAE': float(np.mean([abs(r['prediction']-r['apparent_age_avg']) for r in group])) if group else None,
            'rounded_MAE': float(np.mean([abs(r['prediction']-r['rounded_target']) for r in group])) if group else None}
    return summary


def run(args):
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, audits = {}, {}
    for split, expected in [('train', 4113), ('valid', 1500), ('test', 1978)]:
        rows[split], audits[split] = read_appa(args.data_dir, split, args.ignore_list)
        if not args.allow_small and audits[split]['raw_count'] != expected:
            raise ValueError(f'{split}: expected {expected} raw annotations, got {audits[split]["raw_count"]}')
    write_json(args.output/'data_audit.json', audits)
    write_json(args.output/'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    model = age_model(pretrained=not args.no_pretrained and not args.evaluate).to(args.device)
    if args.init_weights:
        load_weights(model, args.init_weights, args.device)
    if args.evaluate:
        load_weights(model, args.evaluate, args.device)
    else:
        train = loader(APPADataset(rows['train'], True, args.age_stddev, not args.no_augmentation),
                       args.batch_size, args.workers, True, args.seed, drop_last=not args.keep_last)
        if not len(train):
            raise ValueError('No training batches: reduce --batch-size or use --keep-last')
        valid = loader(APPADataset(rows['valid']), args.batch_size, args.workers)
        optimizer = make_optimizer(model, 'appa')
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=.2)
        scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16' and args.device == 'cuda')
        best, history, start = float('inf'), [], time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            total = 0.
            for x, label, _, _ in train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(args.precision, args.device):
                    loss = torch.nn.functional.cross_entropy(model(x.to(args.device)).float(), label.to(args.device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite APPA loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
            metrics = evaluate(model, valid, args.device)
            if metrics['rounded_MAE'] < best:
                best = metrics['rounded_MAE']
                save_weights(model, args.output/'best.pt')
                write_json(args.output/'selection.json', {'epoch': epoch+1, 'valid': metrics, 'criterion': 'rounded_MAE'})
            history.append({'epoch': epoch+1, 'loss': total/len(train), 'valid': metrics,
                            'lr': optimizer.param_groups[0]['lr']})
            scheduler.step()
            write_json(args.output/'history.json', history)
            print(f'appa epoch={epoch+1}/{args.epochs} loss={total/len(train):.6f} valid_MAE={metrics["rounded_MAE"]:.6f}', flush=True)
        save_weights(model, args.output/'last.pt')
        write_json(args.output/'timing.json', {'train_and_valid_wall_seconds': time.perf_counter()-start})
        load_weights(model, args.output/'best.pt', args.device)
    inputs = torch.stack([APPADataset(rows['test'])[i][0] for i in range(min(2, len(rows['test'])))]).to(args.device)
    verify_roundtrip(model, age_model, args.evaluate or args.output/'best.pt', inputs, args.output/'roundtrip.json')
    metrics = evaluate(model, loader(APPADataset(rows['test']), args.batch_size, args.workers), args.device, args.output/'predictions.csv')
    write_json(args.output/'metrics.json', metrics)
    print(metrics, flush=True)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/appa')
    p.add_argument('--ignore-list', type=Path)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--age-stddev', type=float, default=1.)
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--no-augmentation', action='store_true')
    p.add_argument('--keep-last', action='store_true')
    p.add_argument('--allow-small', action='store_true')
    p.add_argument('--init-weights', type=Path)
    p.add_argument('--evaluate', type=Path)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.age_stddev < 0:
        p.error('Invalid epochs, batch size or age stddev')
    run(args)


if __name__ == '__main__':
    main()
