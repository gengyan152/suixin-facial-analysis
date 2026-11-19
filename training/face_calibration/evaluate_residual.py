"""Cross-fit a score calibration and structural residual within each official SCUT test fold."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler


TRAINING = Path(__file__).resolve().parents[1]
SCUT = TRAINING / 'data/SCUT-FBP5500/SCUT-FBP5500_v2'
ARTIFACTS = TRAINING / 'artifacts/face_calibration'
VARIANTS = ('raw', 'score_only', 'score_plus_geometry')
FOLDS = range(1, 6)
SEED = 20
KNOTS = 5
RIDGE_ALPHA = 40


def metrics(target, predictions):
    return {'n': len(target), 'MAE': float(np.abs(target - predictions).mean()),
            'Pearson': float(np.corrcoef(target, predictions)[0, 1])}


def read_fold(fold, predictions_dir, scut):
    official = (scut / 'train_test_files/5_folders_cross_validations_files'
                / f'cross_validation_{fold}/test_{fold}_aligned_ext.txt')
    expected = {}
    for line in official.read_text(encoding='utf-8-sig').splitlines():
        name, target, *_ = line.split()
        if name in expected:
            raise ValueError(f'Duplicate official entry: {name}')
        expected[name] = float(target)
    with (predictions_dir / f'fold{fold}/predictions.csv').open(encoding='utf-8', newline='') as handle:
        saved = list(csv.DictReader(handle))
    names = sorted(row['filename'] for row in saved)
    if len(names) != 1100 or names != sorted(expected):
        raise ValueError(f'Fold {fold} predictions do not match the official 1100-image test split')
    rows = sorted(saved, key=lambda row: row['filename'])
    target = np.array([float(row['target']) for row in rows])
    raw = np.array([float(row['prediction']) for row in rows])
    if not np.allclose(target, [expected[name] for name in names], rtol=0, atol=1e-6):
        raise ValueError(f'Fold {fold} targets differ from the official split')
    if not np.isfinite(raw).all() or not ((raw >= 1) & (raw <= 5)).all():
        raise ValueError(f'Fold {fold} predictions are outside the score range')
    return names, target, raw


def crossfit(fold, target, raw, geometry):
    valid = np.isfinite(geometry).all(axis=1)
    prediction = np.empty((len(raw), len(VARIANTS)))
    prediction[:, 0] = raw
    subfold = np.empty(len(raw), dtype=int)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED + fold)
    for index, (train, test) in enumerate(splitter.split(raw), start=1):
        subfold[test] = index
        basis = SplineTransformer(n_knots=KNOTS, degree=3, knots='quantile', extrapolation='constant')
        score_train = basis.fit_transform(raw[train, None])
        score_test = basis.transform(raw[test, None])
        score_model = np.linalg.lstsq(score_train, target[train], rcond=None)[0]
        prediction[test, 1] = score_test @ score_model
        prediction[test, 2] = prediction[test, 1]

        train_ok = train[valid[train]]
        test_ok = test[valid[test]]
        if not len(train_ok) or not len(test_ok):
            continue
        # Winsorization, scaling and both fits only see the calibration subset.
        # Remove score-explainable variation in the ratios before fitting their residual effect.
        low, high = np.quantile(geometry[train_ok], [0.01, 0.99], axis=0)
        scaler = StandardScaler()
        geometry_train = scaler.fit_transform(np.clip(geometry[train_ok], low, high))
        geometry_test = scaler.transform(np.clip(geometry[test_ok], low, high))
        score_train_ok = score_train[valid[train]]
        score_test_ok = score_test[valid[test]]
        geometry_by_score = np.linalg.lstsq(score_train_ok, geometry_train, rcond=None)[0]
        residual_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        residual_model.fit(geometry_train - score_train_ok @ geometry_by_score,
                           target[train_ok] - score_train_ok @ score_model)
        prediction[test_ok, 2] += residual_model.predict(geometry_test - score_test_ok @ geometry_by_score)
    return np.clip(prediction, 1, 5), subfold


def paired_bootstrap(target, predictions, fold_indices):
    # Paired uncertainty over evaluation cases for the fixed cross-fitted predictions.
    rng = np.random.default_rng(SEED)
    count = 2000
    mae, pearson = np.empty(count), np.empty(count)
    errors = np.abs(target[:, None] - predictions)
    for draw in range(count):
        rows = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices])
        mae[draw] = np.mean(errors[rows, 2] - errors[rows, 1])
        corr = np.corrcoef(target[rows], predictions[rows, 1:].T)[0, 1:]
        pearson[draw] = corr[1] - corr[0]
    return {'draws': count, 'unit': 'evaluation image, resampled within official fold',
            'geometry_minus_score_MAE_95pct': np.quantile(mae, [0.025, 0.975]).tolist(),
            'geometry_minus_score_Pearson_95pct': np.quantile(pearson, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, default=ARTIFACTS / 'scut_geometry.jsonl')
    parser.add_argument('--predictions', type=Path, default=TRAINING / 'artifacts/2dfap')
    parser.add_argument('--scut', type=Path, default=SCUT)
    parser.add_argument('--output', type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    statuses = Counter()
    measurements = {}
    with args.geometry.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            name = row['filename']
            if name in measurements:
                raise ValueError(f'Duplicate geometry entry: {name}')
            statuses[row['status']] += 1
            measurements[name] = row['measurements'] if row['status'] == 'ok' else None
    feature_names = tuple(next(features for features in measurements.values() if features is not None))
    if len(feature_names) != 15 or any(features.keys() != set(feature_names) for features in measurements.values() if features is not None):
        raise ValueError('Inconsistent geometry columns')

    args.output.mkdir(parents=True, exist_ok=True)
    all_names = set()
    all_targets, all_predictions, folds, csv_rows = [], [], [], []
    for fold in FOLDS:
        names, target, raw = read_fold(fold, args.predictions, args.scut)
        if all_names.intersection(names):
            raise ValueError(f'Official fold {fold} overlaps earlier folds')
        all_names.update(names)
        geometry = np.array([[measurements[name][feature] for feature in feature_names]
                             if measurements[name] is not None else [np.nan] * len(feature_names) for name in names])
        predictions, subfold = crossfit(fold, target, raw, geometry)
        all_targets.append(target)
        all_predictions.append(predictions)
        results = {variant: metrics(target, predictions[:, i]) for i, variant in enumerate(VARIANTS)}
        folds.append({'fold': fold, 'geometry_n': int(np.isfinite(geometry).all(axis=1).sum()), **results})
        csv_rows.extend((name, fold, member, gt, *values) for name, member, gt, values in
                        zip(names, subfold, target, predictions, strict=True))
    if all_names != set(measurements):
        raise ValueError('Geometry filenames do not match the full official fold union')

    target = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    pooled = {variant: metrics(target, predictions[:, i]) for i, variant in enumerate(VARIANTS)}
    fold_indices = np.split(np.arange(len(target)), np.cumsum([len(array) for array in all_targets])[:-1])
    report = {
        'protocol': '5-way cross-fitting inside each official 2D-FAP holdout, one prediction per image',
        'prediction_input': str(args.predictions), 'geometry_input': str(args.geometry),
        'dataset': 'SCUT-FBP5500_v2', 'seed': SEED, 'score_spline_quantile_knots': KNOTS,
        'geometry_ridge_alpha': RIDGE_ALPHA, 'geometry_status_counts': dict(statuses),
        'geometry_features': feature_names, 'folds': folds, 'pooled': pooled,
        'geometry_minus_score': {metric: pooled['score_plus_geometry'][metric] - pooled['score_only'][metric]
                                 for metric in ('MAE', 'Pearson')},
        'paired_bootstrap': paired_bootstrap(target, predictions, fold_indices),
    }
    with (args.output / 'crossfit_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('filename', 'official_fold', 'calibration_subfold', 'target', *VARIANTS))
        writer.writerows(csv_rows)
    summary = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output / 'crossfit_summary.json').write_text(summary + '\n', encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
