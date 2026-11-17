"""Aggregate five official fold predictions; reject duplicate test membership."""
import argparse
import csv
from pathlib import Path
from runtime import write_json, regression_metrics
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folds', type=Path, nargs=5, help='Five fold output directories, in order 1..5')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-small', action='store_true')
    args = p.parse_args()
    metrics, combined, seen = [], [], set()
    for index, folder in enumerate(args.folds, 1):
        with (folder/'predictions.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        names = [r['filename'] for r in rows]
        if len(set(names)) != len(names) or seen.intersection(names):
            raise ValueError('Repeated filename within/across test folds')
        if not args.allow_small and len(rows) != 1100:
            raise ValueError(f'Fold {index}: expected 1100 predictions')
        seen.update(names)
        metrics.append({'fold': index, **regression_metrics([float(r['target']) for r in rows], [float(r['prediction']) for r in rows])})
        combined.extend(rows)
    summary = {'folds': metrics, 'fold_mean_std': {}, 'std_definition': 'sample standard deviation, ddof=1',
               'pooled': regression_metrics([float(r['target']) for r in combined], [float(r['prediction']) for r in combined])}
    for key in ['PC', 'MAE', 'RMSE']:
        values = [r[key] for r in metrics]
        summary['fold_mean_std'][key] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))} if all(v is not None for v in values) else None
    write_json(args.output, summary)
    print(summary)


if __name__ == '__main__':
    main()
