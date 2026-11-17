"""Validated SCUT extended annotations and official APPA-REAL split loading."""
import csv
from pathlib import Path
from runtime import ROOT
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def read_scut(path, images):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 8:
            raise ValueError(f"{path}:{line_number}: expected filename mean std count1..count5 (8 fields)")
        name, mean, std, *counts = values
        mean, std = float(mean), float(std)
        counts = np.asarray(counts, dtype=np.float64)
        if not (np.isfinite([mean, std, *counts]).all() and 1 <= mean <= 5 and std > 0 and
                (counts >= 0).all() and counts.sum() > 0):
            raise ValueError(f"{path}:{line_number}: invalid extended annotation")
        image = Path(images)/name
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append({"name": name, "path": str(image), "mean": mean, "std": std,
                     "levels": (counts/counts.sum()).astype(np.float32)})
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError(f"Empty split or duplicate filenames: {path}")
    return rows


def fine_target(mean, std):
    x = np.arange(1, 5.1, .1)
    b = std/np.sqrt(2)
    cdf = .5*(1+np.sign(x-mean)*(1-np.exp(-np.abs(x-mean)/b)))
    target = torch.tensor(np.maximum(np.diff(cdf), 1e-15), dtype=torch.float32)
    return torch.nn.functional.normalize(torch.sigmoid(target), p=1, dim=0)


class SCUTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        self.targets = [fine_target(r['mean'], r['std']) for r in rows]
        steps = [transforms.Resize(256), transforms.CenterCrop(224)]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(
            [.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            x = self.transform(image.convert('RGB'))
        return x, self.targets[index], np.float64(row['mean']), row['levels'], row['name']


def read_appa(root, split, ignore_path=None, require_original=True):
    root = Path(root)
    ignore_path = ignore_path or ROOT/'vendor/age-estimation-pytorch/ignore_list.csv'
    with Path(ignore_path).open(encoding='utf-8-sig', newline='') as handle:
        ignored = {r['img_name'] for r in csv.DictReader(handle)}
    with (root/f'gt_avg_{split}.csv').open(encoding='utf-8-sig', newline='') as handle:
        raw = list(csv.DictReader(handle))
    rows, excluded = [], []
    for row in raw:
        name = row['file_name']
        if name in ignored:
            excluded.append({'name': name, 'reason': 'official_ignore_list'})
            continue
        mean, std = float(row['apparent_age_avg']), float(row['apparent_age_std'])
        if not (np.isfinite([mean, std]).all() and 0 <= mean <= 100 and std >= 0):
            raise ValueError(f"Invalid APPA annotation: {split}/{name}")
        face = root/split/(name+'_face.jpg')
        if not face.is_file():
            raise FileNotFoundError(face)
        if require_original and not (root/split/name).is_file():
            raise FileNotFoundError(root/split/name)
        rows.append({'name': name, 'path': str(face), 'mean': mean, 'std': std})
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate APPA split: {split}")
    return rows, {'raw_count': len(raw), 'used_count': len(rows), 'excluded': excluded}


def appa_augmentation():
    # imgaug 0.4's dtype registry uses np.sctypes, removed in NumPy 2.
    # Supply only this compatibility registry before importing the unchanged operators.
    if not hasattr(np, 'sctypes'):
        np.sctypes = {'float': [np.float16, np.float32, np.float64],
                     'int': [np.int8, np.int16, np.int32, np.int64],
                     'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                     'complex': [np.complex64, np.complex128], 'others': [bool, object, str, bytes]}
    from imgaug import augmenters as iaa
    return iaa.Sequential([
        iaa.OneOf([iaa.Sometimes(.25, iaa.AdditiveGaussianNoise(scale=.1*255)),
                   iaa.Sometimes(.25, iaa.GaussianBlur(sigma=(0, 3.)))]),
        iaa.Affine(rotate=(-20, 20), mode='edge', scale={'x': (.95, 1.05), 'y': (.95, 1.05)},
                   translate_percent={'x': (-.05, .05), 'y': (-.05, .05)}),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
        iaa.GammaContrast((.3, 2)), iaa.Fliplr(.5)])


class APPADataset(torch.utils.data.Dataset):
    def __init__(self, rows, train=False, age_stddev=1., augmentation=True):
        self.rows, self.train, self.age_stddev = rows, train, age_stddev
        # Lazy construction in each spawned Windows worker avoids pickling imgaug RNG/module state.
        self.use_augmentation = train and augmentation
        self.augment = None
        self.worker_seed = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # imdecode/fromfile retains OpenCV BGR semantics and supports Windows Unicode paths.
        image = cv2.imdecode(np.fromfile(row['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {row['path']}")
        image = cv2.resize(image, (224, 224))
        if self.use_augmentation:
            if self.augment is None:
                self.augment = appa_augmentation()
            seed = torch.initial_seed() % 2**32
            if self.worker_seed != seed:
                self.augment.seed_(seed)
                self.worker_seed = seed
            image = self.augment.augment_image(image)
        age = row['mean'] + np.random.randn()*row['std']*self.age_stddev if self.train else row['mean']
        label = int(np.clip(round(age), 0, 100))
        return torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1).copy()), label, np.float64(row['mean']), row['name']
