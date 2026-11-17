"""Arithmetic check of five-fold aggregation using explicit synthetic predictions."""
import csv
import json
import math
import subprocess
import sys
from runtime import ROOT, write_json


def main():
    folder = ROOT/'artifacts/summary_fixture'
    folders = []
    for fold in range(1, 6):
        path = folder/f'fold{fold}'
        path.mkdir(parents=True, exist_ok=True)
        folders.append(str(path))
        with (path/'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['filename', 'target', 'prediction'])
            writer.writeheader()
            writer.writerows({'filename': f'synthetic_f{fold}_{i}', 'target': score, 'prediction': score+fold*.1}
                             for i, score in enumerate([1., 3., 5.]))
    output = folder/'summary.json'
    subprocess.run([sys.executable, str(ROOT/'summarize_folds.py'), *folders, '--output', str(output), '--allow-small'], check=True)
    result = json.loads(output.read_text(encoding='utf-8'))
    assert abs(result['fold_mean_std']['MAE']['mean']-.3) < 1e-12
    assert abs(result['fold_mean_std']['MAE']['std']-math.sqrt(.025)) < 1e-12
    assert abs(result['pooled']['RMSE']-math.sqrt(.11)) < 1e-12
    assert result['pooled']['n'] == 15
    write_json(folder/'verification.json', {'synthetic_only': True, 'five_folds': True, 'mean_std_and_pooled_exact_formula': True})
    print('Five-fold summary arithmetic passed')


if __name__ == '__main__':
    main()
