"""Two real epochs per architecture, spawned Windows loaders, exact checkpoint checks."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from runtime import ROOT, write_json
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/dry_run')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output/'scut/images'
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20)
    names = []
    for i in range(12):
        name = f'{"AF" if i%2 == 0 else "AM"}{i:04d}.jpg'
        Image.fromarray(rng.integers(0, 256, (256, 280, 3), dtype=np.uint8)).save(images/name)
        names.append(name)
    for split, indices in [('train', range(8)), ('test', range(8, 12))]:
        (args.output/f'scut/{split}.txt').write_text('\n'.join(
            f'{names[i]} {2+i*.15:.4f} 0.7 3 10 24 18 5' for i in indices)+'\n', encoding='utf-8')
    appa = args.output/'appa_data'
    for split, n in [('train', 8), ('valid', 4), ('test', 4)]:
        folder = appa/split
        folder.mkdir(parents=True, exist_ok=True)
        with (appa/f'gt_avg_{split}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['file_name', 'apparent_age_avg', 'apparent_age_std'])
            writer.writeheader()
            for i in range(n):
                name = f'synthetic_{split}_{i}.jpg'
                image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
                image.save(folder/name)
                image.save(folder/(name+'_face.jpg'))
                writer.writerow({'file_name': name, 'apparent_age_avg': 18.3+i*3.1, 'apparent_age_std': 2.2})
    commands = [
        [str(ROOT/'train_2dfap.py'), '--images', str(images), '--train-list', str(args.output/'scut/train.txt'),
         '--test-list', str(args.output/'scut/test.txt'), '--output', str(args.output/'2dfap'), '--dev-fraction', '.25'],
        [str(ROOT/'train_appa_age.py'), '--data-dir', str(appa), '--output', str(args.output/'appa')],
    ]
    results = {}
    for name, command in zip(['2dfap', 'appa'], commands):
        command = [sys.executable, *command, '--epochs', '2', '--batch-size', '2', '--workers', str(args.workers),
                   '--no-pretrained', '--allow-small', '--device', args.device]
        print(f'RUN dry-run {name}', flush=True)
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
        (args.output/f'{name}.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        print(result.stdout, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError(f'{name} dry-run failed, see {args.output/name}.log')
        results[name] = json.loads((args.output/name/'roundtrip.json').read_text(encoding='utf-8'))
        results[name]['epochs'] = 2
        results[name]['workers'] = args.workers
    write_json(args.output/'verification.json', results)
    print(results, flush=True)


if __name__ == '__main__':
    main()
