"""Extract roll-aligned geometry, bootstrap sex-stratified correlations, verify holdout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from runtime import ROOT, write_json
import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr
from datasets import read_scut
from geometry import LANDMARKS, VERSION, measure_geometry


def correlate(rows, bootstraps, seed):
    rng = np.random.default_rng(seed)
    output = {}
    for gender in ['female', 'male']:
        group = [r for r in rows if r['gender'] == gender]
        output[gender] = {'n': len(group), 'metrics': {}}
        names = list(group[0]['geometry']) if group else []
        for name in names:
            x = np.asarray([r['geometry'][name] for r in group])
            y = np.asarray([r['score'] for r in group])
            record = {'n': len(group), 'pearson': None, 'spearman': None, 'supported': False}
            if len(group) >= 4 and np.ptp(x) > 0 and np.ptp(y) > 0:
                for label, function in [('pearson', pearsonr), ('spearman', spearmanr)]:
                    result = function(x, y)
                    estimates = []
                    for _ in range(bootstraps):
                        indices = rng.integers(len(group), size=len(group))
                        if np.ptp(x[indices]) > 0 and np.ptp(y[indices]) > 0:
                            estimates.append(float(function(x[indices], y[indices]).statistic))
                    record[label] = {'r': float(result.statistic), 'p': float(result.pvalue),
                                     'ci95': np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
                                     'valid_bootstraps': len(estimates)}
            output[gender]['metrics'][name] = record
    # BH over both genders, all metrics and both correlation measures as one declared family.
    tests = [m[label] for g in output.values() for m in g['metrics'].values()
             for label in ['pearson', 'spearman'] if m[label] is not None]
    order = sorted(range(len(tests)), key=lambda i: tests[i]['p'])
    running = 1.
    for rank in range(len(order), 0, -1):
        i = order[rank-1]
        running = min(running, tests[i]['p']*len(order)/rank)
        tests[i]['q_bh'] = running
    for group in output.values():
        for item in group['metrics'].values():
            item['supported'] = all(item[label] is not None and item[label]['ci95'] is not None
                and item[label]['q_bh'] < .05 and np.prod(item[label]['ci95']) > 0
                for label in ['pearson', 'spearman'])
            if item['supported']:
                item['supported'] = item['pearson']['r']*item['spearman']['r'] > 0
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--train-list', type=Path, required=True)
    p.add_argument('--holdout-list', type=Path)
    p.add_argument('--landmarker', type=Path, help='Official face_landmarker.task; no automatic download')
    p.add_argument('--landmarks-jsonl', type=Path, help='Cached normalized 478 points keyed by filename')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/gender_stats.json')
    p.add_argument('--bootstraps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--max-yaw', type=float, default=25.)
    p.add_argument('--max-pitch', type=float, default=25.)
    args = p.parse_args()
    if args.bootstraps < 100 or bool(args.landmarker) == bool(args.landmarks_jsonl):
        p.error('Use exactly one of --landmarker / --landmarks-jsonl, and >=100 bootstraps')
    splits = {'train': read_scut(args.train_list, args.images)}
    if args.holdout_list:
        splits['holdout'] = read_scut(args.holdout_list, args.images)
        if {r['name'] for r in splits['train']} & {r['name'] for r in splits['holdout']}:
            raise ValueError('Training and holdout overlap')
    cache, detector = {}, None
    if args.landmarks_jsonl:
        entries = [json.loads(line) for line in args.landmarks_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
        cache = {r['filename']: r for r in entries}
        if len(cache) != len(entries):
            raise ValueError('Duplicate cached landmark filenames')
    else:
        import mediapipe as mp
        detector = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=args.landmarker.read_bytes()), num_faces=2,
            output_facial_transformation_matrixes=True))
    collected, rejected = {}, []
    for split, records in splits.items():
        collected[split] = []
        for row in records:
            prefix = Path(row['name']).name[:2].upper()
            if prefix not in ('AF', 'AM', 'CF', 'CM'):
                raise ValueError(f"Cannot derive official dataset sex from filename: {row['name']}")
            reason, entry = None, None
            if detector:
                bgr = cv2.imdecode(np.fromfile(row['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    reason = 'decode_failed'
                else:
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                    if len(result.face_landmarks) != 1:
                        reason = f'face_count_{len(result.face_landmarks)}'
                    else:
                        entry = {'width': bgr.shape[1], 'height': bgr.shape[0],
                                 'landmarks': [[v.x, v.y, v.z] for v in result.face_landmarks[0]],
                                 'matrix': result.facial_transformation_matrixes[0].tolist()}
            else:
                entry = cache.get(row['name'])
                if entry is None:
                    reason = 'missing_cached_landmarks'
            if entry is not None:
                if 'matrix' in entry:
                    rotation = np.asarray(entry['matrix'], dtype=np.float64)[:3, :3]
                    rotation = rotation/np.linalg.norm(rotation, axis=0)
                    pitch, yaw = cv2.RQDecomp3x3(rotation)[0][:2]
                    if abs(yaw) > args.max_yaw or abs(pitch) > args.max_pitch:
                        reason = 'pose_out_of_range'
                # Bad geometry is an expected per-image measurement failure, recorded explicitly.
                if reason is None:
                    try:
                        geometry = measure_geometry(entry['landmarks'], entry['width'], entry['height'])
                    except ValueError as error:
                        reason = str(error)
            if reason:
                rejected.append({'split': split, 'filename': row['name'], 'reason': reason})
            else:
                collected[split].append({'filename': row['name'], 'score': row['mean'],
                                        'gender': 'female' if prefix[1] == 'F' else 'male', 'geometry': geometry})
        print(f'{split}: {len(collected[split])}/{len(records)} geometry records', flush=True)
    if detector:
        detector.close()
    statistics = {split: correlate(rows, args.bootstraps, args.seed) for split, rows in collected.items()}
    for gender, group in statistics['train'].items():
        for name, item in group['metrics'].items():
            held = statistics.get('holdout', {}).get(gender, {}).get('metrics', {}).get(name)
            item['holdout_confirmed'] = bool(item['supported'] and held and held['supported'] and
                                            item['pearson']['r']*held['pearson']['r'] > 0)
            item['display_trend'] = item['holdout_confirmed']
    output = {'schema_version': VERSION, 'source': 'SCUT-FBP5500 human mean scores', 'seed': args.seed,
              'bootstrap': {'method': 'paired percentile', 'resamples': args.bootstraps, 'confidence': .95},
              'multiplicity': 'BH across both sexes, all features and both measures, separately per split',
              'landmarks': LANDMARKS, 'splits': statistics, 'excluded': rejected,
              'input_counts': {split: len(rows) for split, rows in splits.items()},
              'train_list_sha256': hashlib.sha256(args.train_list.read_bytes()).hexdigest(),
              'landmarker_sha256': hashlib.sha256(args.landmarker.read_bytes()).hexdigest() if args.landmarker else None,
              'pose_checked': bool(args.landmarker) or all('matrix' in e for e in cache.values()),
              'interpretation': 'Dataset association, not causation. mirror_asymmetry is lower for more symmetric geometry.',
              'ethics': '仅反映特定公开数据集标注群体的主观统计偏好，不代表客观美丑标准或个体价值。'}
    write_json(args.output, output)
    args.output.with_suffix('.measurements.jsonl').write_text('\n'.join(json.dumps({'split': s, **r}, ensure_ascii=False)
        for s, rows in collected.items() for r in rows)+'\n', encoding='utf-8')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
