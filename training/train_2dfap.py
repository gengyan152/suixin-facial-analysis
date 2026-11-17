"""SCUT-FBP5500 2D-FAP; held-out fold is evaluated only after model selection."""
import argparse
import csv
from pathlib import Path
import time
from runtime import ROOT, autocast, loader, load_weights, regression_metrics, save_weights, seed_everything, write_json, verify_roundtrip
import numpy as np
import torch
from datasets import SCUTDataset, read_scut
from models import TwoDFAP, fap_loss, score_outputs, make_optimizer


def evaluate(model, data, device, output=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, _, mean, _, names in data:
            probabilities = model(x.to(device, non_blocking=True))
            score, levels = score_outputs(probabilities)
            for name, gt, pred, fine, coarse in zip(names, mean.tolist(), score.cpu().tolist(), probabilities.cpu().tolist(), levels.cpu().tolist()):
                rows.append({'filename': name, 'target': gt, 'prediction': pred,
                             **{f'p{i+1}': p for i, p in enumerate(coarse)},
                             **{f'fine_{i:02d}': p for i, p in enumerate(fine)}})
    if output:
        with Path(output).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return regression_metrics([r['target'] for r in rows], [r['prediction'] for r in rows])


def run(args):
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train = read_scut(args.train_list, args.images)
    test = read_scut(args.test_list, args.images)
    if {r['name'] for r in train} & {r['name'] for r in test}:
        raise ValueError('Training and test overlap')
    if not args.allow_small and (len(train), len(test)) != (4400, 1100):
        raise ValueError(f'Expected official 4400/1100, got {len(train)}/{len(test)}')
    dev = []
    if args.dev_fraction:
        indices = np.random.default_rng(args.seed).permutation(len(train))
        n = max(1, int(len(train)*args.dev_fraction))
        dev = [train[i] for i in indices[:n]]
        train = [train[i] for i in indices[n:]]
    write_json(args.output/'split_manifest.json', {'train': [r['name'] for r in train],
               'dev': [r['name'] for r in dev], 'test': [r['name'] for r in test]})
    write_json(args.output/'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    train_loader = loader(SCUTDataset(train, True), args.batch_size, args.workers, True, args.seed)
    dev_loader = loader(SCUTDataset(dev), args.batch_size, args.workers) if dev else None
    model = TwoDFAP(pretrained=not args.no_pretrained).to(args.device)
    if args.init_weights:
        load_weights(model, args.init_weights, args.device)
    optimizer = make_optimizer(model, '2dfap')
    for group in optimizer.param_groups:
        group['lr'] = args.lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16' and args.device == 'cuda')
    history, best, selected = [], float('inf'), 0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for x, fine, mean, levels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.precision, args.device):
                loss = fap_loss(model(x.to(args.device)), fine.to(args.device), mean.to(args.device), levels.to(args.device),
                                args.alpha, args.beta, args.gamma, args.score_loss)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite 2D-FAP loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        metrics = evaluate(model, dev_loader, args.device) if dev else None
        if not dev or metrics['MAE'] < best:
            best = metrics['MAE'] if dev else best
            selected = epoch+1
            save_weights(model, args.output/'best.pt')
        history.append({'epoch': epoch+1, 'loss_batch_mean': total/len(train_loader),
                        'lr': optimizer.param_groups[0]['lr'], 'dev': metrics})
        scheduler.step()
        write_json(args.output/'history.json', history)
        print(f'2dfap fold={args.fold} epoch={epoch+1}/{args.epochs} loss={total/len(train_loader):.6f}', flush=True)
    save_weights(model, args.output/'last.pt')
    load_weights(model, args.output/'best.pt', args.device)
    inputs = torch.stack([SCUTDataset(test)[i][0] for i in range(min(2, len(test)))]).to(args.device)
    verify_roundtrip(model, TwoDFAP, args.output/'best.pt', inputs, args.output/'roundtrip.json')
    metrics = evaluate(model, loader(SCUTDataset(test), args.batch_size, args.workers), args.device, args.output/'predictions.csv')
    result = {'fold': args.fold, 'seed': args.seed, 'selected_epoch': selected,
              'selection': 'fold-internal dev MAE' if dev else 'fixed final epoch',
              'metrics': metrics, 'wall_seconds': time.perf_counter()-start}
    write_json(args.output/'metrics.json', result)
    print(result, flush=True)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--train-list', type=Path, required=True)
    p.add_argument('--test-list', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/2dfap/fold1')
    p.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--dev-fraction', type=float, default=0.)
    p.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--gamma', type=float, default=1.)
    p.add_argument('--score-loss', choices=['exp', 'l1'], default='exp')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--init-weights', type=Path)
    p.add_argument('--allow-small', action='store_true', help='Synthetic smoke fixtures only')
    args = p.parse_args()
    if not 0 <= args.dev_fraction < .5 or args.epochs < 1 or args.batch_size < 1:
        p.error('Invalid dev fraction, epochs or batch size')
    run(args)


if __name__ == '__main__':
    main()
