"""Validate all official lists, image decodability, extended labels and exclusions."""
import argparse
from pathlib import Path
from collections import Counter
from runtime import ROOT, write_json
import cv2
import numpy as np
from datasets import read_scut, read_appa


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='dataset', required=True)
    scut = sub.add_parser('scut')
    scut.add_argument('--images', type=Path, required=True)
    scut.add_argument('--splits', type=Path, required=True, help='Directory containing cross_validation_1..5')
    appa = sub.add_parser('appa')
    appa.add_argument('--data-dir', type=Path, required=True)
    for child in [scut, appa]:
        child.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report, image_paths = {}, set()
    if args.dataset == 'scut':
        all_tests, universe = [], None
        for fold in range(1, 6):
            folder = args.splits/f'cross_validation_{fold}'
            train = read_scut(folder/f'train_{fold}_aligned_ext.txt', args.images)
            test = read_scut(folder/f'test_{fold}_aligned_ext.txt', args.images)
            a, b = {r['name'] for r in train}, {r['name'] for r in test}
            if (len(a), len(b)) != (4400, 1100) or a & b:
                raise ValueError(f'Invalid official fold {fold}')
            if universe is not None and universe != a | b:
                raise ValueError('Fold universes differ')
            universe = a | b
            all_tests.extend(b)
            report[str(fold)] = {'train': len(a), 'test': len(b),
                                 'train_groups': dict(Counter(Path(r['name']).name[:2] for r in train))}
            image_paths.update(r['path'] for r in train+test)
        if len(set(all_tests)) != 5500:
            raise ValueError('Every image must be tested exactly once across five folds')
        counts = Counter(Path(name).name[:2] for name in universe)
        if dict(counts) != {'AF': 2000, 'AM': 2000, 'CF': 750, 'CM': 750}:
            raise ValueError(f'Unexpected full SCUT groups: {counts}')
        report['full_groups'] = dict(counts)
    else:
        for split, expected in [('train', 4113), ('valid', 1500), ('test', 1978)]:
            rows, audit = read_appa(args.data_dir, split)
            if audit['raw_count'] != expected:
                raise ValueError(f'{split}: expected {expected} original annotations')
            report[split] = audit
            image_paths.update(r['path'] for r in rows)
    unreadable = [path for path in sorted(image_paths)
                  if cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) is None]
    report['unreadable'] = unreadable
    report['decoded_unique_files'] = len(image_paths)
    write_json(args.output, report)
    if unreadable:
        raise ValueError(f'{len(unreadable)} unreadable images; see {args.output}')
    print(f'PASS: {args.output}')


if __name__ == '__main__':
    main()
