"""Extract MediaPipe landmarks and structural ratios from the original SCUT photos."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps


TRAINING = Path(__file__).resolve().parents[1]
SCUT = TRAINING / 'data/SCUT-FBP5500/SCUT-FBP5500_v2'
ARTIFACTS = TRAINING / 'artifacts/face_calibration'
POINTS = 478


def structural_ratios(landmarks, width, height):
    # Use pixel distances and an eye-centred basis to remove in-plane rotation.
    p = landmarks[:, :2] * [width, height]

    def distance(first, second):
        return np.linalg.norm(p[first] - p[second])

    eye_left, eye_right = (p[33] + p[133]) / 2, (p[362] + p[263]) / 2
    eye_axis = eye_right - eye_left
    interocular = np.linalg.norm(eye_axis)
    face_width = distance(234, 454)
    if min(interocular, face_width) < 1:
        return None
    down = np.array([-eye_axis[1], eye_axis[0]]) / interocular
    face_height = (p[152] - p[10]) @ down
    left_eye_width, right_eye_width = distance(33, 133), distance(362, 263)
    mouth_width = distance(61, 291)
    if min(face_height, left_eye_width, right_eye_width, mouth_width) < 1:
        return None
    return {
        'face_height_to_width': face_height / face_width,
        'jaw_width_to_face_width': distance(172, 397) / face_width,
        'forehead_width_to_face_width': distance(54, 284) / face_width,
        'eye_spacing_to_face_width': interocular / face_width,
        'eye_width_to_face_width': (left_eye_width + right_eye_width) / (2 * face_width),
        'eye_opening_to_width': (distance(159, 145) / left_eye_width + distance(386, 374) / right_eye_width) / 2,
        'brow_to_eye_distance': ((p[159] - p[105]) + (p[386] - p[334])) @ down / (2 * face_height),
        'nose_width_to_face_width': distance(98, 327) / face_width,
        'nose_length_to_face_height': (p[2] - p[168]) @ down / face_height,
        'mouth_width_to_face_width': mouth_width / face_width,
        'lip_height_to_mouth_width': distance(0, 17) / mouth_width,
        'mouth_opening_to_width': distance(13, 14) / mouth_width,
        'philtrum_to_face_height': (p[0] - p[2]) @ down / face_height,
        'lower_face_to_face_height': (p[152] - p[2]) @ down / face_height,
        'chin_to_face_height': (p[152] - p[17]) @ down / face_height,
    }


def initialize(model_path):
    global landmarker
    vision = mp.tasks.vision
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
        num_faces=2,
        output_face_blendshapes=True,
    ))


def extract(path):
    row = {'filename': path.name}
    try:
        with Image.open(path) as image:
            rgb = np.array(ImageOps.exif_transpose(image).convert('RGB'))
    except OSError:
        return {**row, 'status': 'invalid_image'}
    height, width = rgb.shape[:2]
    row.update(width=width, height=height)
    detected = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    n_faces = len(detected.face_landmarks)
    if n_faces != 1:
        return {**row, 'status': 'no_face' if n_faces == 0 else 'multiple_faces'}
    landmarks = np.array([[p.x, p.y, p.z] for p in detected.face_landmarks[0]], dtype=np.float64)
    if landmarks.shape != (POINTS, 3) or not np.isfinite(landmarks).all():
        return {**row, 'status': 'invalid_geometry'}
    ratios = structural_ratios(landmarks, width, height)
    if ratios is None:
        return {**row, 'status': 'invalid_geometry'}
    blendshapes = {c.category_name: c.score for c in detected.face_blendshapes[0]}
    return {**row, 'status': 'ok', 'landmarks': landmarks.round(7).tolist(),
            'blendshapes': blendshapes, 'measurements': ratios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=SCUT / 'Images')
    parser.add_argument('--model', type=Path, default=TRAINING.parent / 'backend/weights/face_landmarker.task')
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'scut_geometry.jsonl')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    photos = sorted(args.images.glob('*.jpg'))
    if len(photos) != 5500 or args.workers < 1:
        parser.error(f'Expected 5500 original SCUT JPEGs and positive workers, found {len(photos)} JPEGs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.partial')
    statuses = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.model,)) as pool:
        with tmp.open('w', encoding='utf-8') as out:
            for index, row in enumerate(pool.map(extract, photos, chunksize=8), start=1):
                out.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                statuses[row['status']] += 1
                if index % 500 == 0:
                    print(index, dict(statuses), flush=True)
    tmp.replace(args.output)
    metadata = {
        'dataset_root': str(args.images), 'mediapipe_version': mp.__version__,
        'model_sha256': sha256(args.model.read_bytes()).hexdigest(),
        'status_counts': dict(statuses),
        'coordinate_source': 'MediaPipe normalized x,y,z after EXIF transpose of original image',
    }
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(metadata, flush=True)


if __name__ == '__main__':
    main()
