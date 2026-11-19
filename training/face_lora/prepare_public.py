"""Build single-region Chinese supervision from public facial captions and matched photos."""

import argparse
from collections import Counter, defaultdict, deque
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
from zipfile import ZipFile

from PIL import Image, ImageOps

from common import DuplicateIndex, FIELD_NAMES, fingerprint, sample_prompt_and_target, write_rows


FACE2TEXT = 'Face2Text v2.1 (human)'
UNIFACE = 'UniF2aceD-1M MM-CelebA-HQ (generated)'
CELEBA_IMAGE_COUNT = 202599
FORM = {
    'brows': ('eyebrows|brows', {
        'thin|slim': '眉毛较细。', 'thick|bushy|full': '眉毛较浓密。',
        'arched|curved': '眉毛带有弧度。', 'straight': '眉毛走向较平直。',
        'sparse': '眉毛较稀疏。',
    }),
    'eyes': ('eyes', {
        'round|rounded': '眼形偏圆。', 'almond.shaped|almond': '眼形呈杏仁状。',
        'large|big': '眼睛较大。', 'small': '眼睛较小。',
        'narrow|narrowed': '眼睛开合较窄。', 'wide.open': '双眼睁得较开。',
        'closed': '双眼闭合。', 'deep.set': '眼窝较深。',
        'wide.set': '双眼间距较宽。', 'close.set': '双眼间距较窄。',
    }),
    'nose': (r'nose(?!\s+(?:bridge|tip|ring|piercing|stud))', {
        'broad|wide': '鼻部较宽。', 'narrow': '鼻部较窄。', 'long': '鼻部较长。',
        'short': '鼻部较短。', 'straight': '鼻梁较直。', 'small|little|petite': '鼻子较小。',
        'large|big': '鼻子较大。', 'pointy|pointed': '鼻尖偏尖。',
        'round|rounded|bulbous': '鼻尖偏圆。', 'upturned|turned.up': '鼻尖微翘。',
    }),
    'mouth': ('lips', {
        'thin|slim': '嘴唇较薄。', 'full|plump|thick': '嘴唇较饱满。',
        'parted|open': '双唇分开。', 'closed': '双唇合拢。',
        'pursed': '双唇抿起。',
    }),
    'contour': (r'face(?!\s+(?:mask|paint|makeup))|facial shape|face shape', {
        'round|rounded': '脸部轮廓偏圆。', 'oval|egg.shaped': '脸部轮廓呈椭圆形。',
        'square': '脸部轮廓偏方。', 'heart.shaped': '脸部轮廓呈心形。',
        'oblong|elongated|long': '脸部轮廓偏长。', 'angular': '脸部轮廓棱角较明显。',
    }),
}
DEGREE = r'(?:(?:very|slightly|rather|quite|somewhat|fairly|relatively|noticeably|gently|subtly|moderately)\s+)?'
HEDGE = re.compile(r"\b(?:no|not|never|without|neither|nor|or|barely|hardly|possibly|perhaps|maybe|might|may|unclear|rather than|instead of)\b|n['’]t\b")
CONTEXT_BREAK = re.compile(r'\b(?:but|whereas|however)\b|[.!?;\n]+')
MENTIONS = {
    'brows': re.compile(r'\b(?:eyebrows|brows)\b'),
    'eyes': re.compile(r'\beyes\b'),
    'nose': re.compile(r'\b(?:nose|nasal)\b'),
    'mouth': re.compile(r'\blips\b'),
    'contour': re.compile(r'\b(?:face|facial|jaw|jawline|chin|cheekbones)\b'),
    'expression': re.compile(r'\b(?:smil(?:e|es|ing)|furrowed)\b'),
}
REFERENCE = re.compile(r'\b(?:seems? to|appears? to|as if|like (?:a|an|the)|looks? like)\b[^,]*$|["“]\s*$')
REFERENCE_AFTER = re.compile(r'\s*(?:would\b|if\b|["”])')


def make_rules():
    compiled = defaultdict(list)
    for field, (noun, variants) in FORM.items():
        for descriptors, chinese in variants.items():
            modifier = f'(?:{descriptors})'
            token = rf'\b{DEGREE}{modifier}\s+(?:{noun})\b'
            predicate = rf'\b(?:{noun})\s+(?:is|are)\s+{DEGREE}{modifier}\b(?!\s+(?:or|ish)\b|-)'
            compiled[field].append((re.compile(f'{token}|{predicate}'), chinese))
    specific = {
        'nose': {
            r'(?:straight (?:nose bridge|bridge of (?:the|her|his|their) nose)|(?:nose bridge|nasal bridge) is straight)': '鼻梁较直。',
            r'(?:rounded nose tip|(?:nose tip|tip of (?:the|her|his|their) nose) is rounded)': '鼻尖偏圆。',
        },
        'contour': {
            r'(?:defined|prominent|chiselled|chiseled|sharp) jawline|jawline is (?:defined|prominent|chiselled|chiseled|sharp)': '下颌轮廓较清晰。',
            r'(?:square jaw|jawline is square)': '下颌轮廓偏方。',
            r'(?:pointed|pointy) chin|chin is (?:pointed|pointy)': '下巴偏尖。',
            r'(?:rounded|round) chin|chin is (?:rounded|round)': '下巴偏圆。',
            r'(?:high|prominent) cheekbones|cheekbones are (?:high|prominent)': '颧骨较突出。',
        },
        'expression': {
            r'(?:she|he|they|the (?:person|subject)) (?:is|are) (?:[a-z]+ly )?smiling|(?:she|he|they) (?:smiles|smile)(?!\s+(?:lines|wrinkles))': '嘴角上扬。',
            r'(?:has|wears|shows|displays|with|forming|forms) (?:a |an )?(?:(?:slight|subtle|gentle|broad|wide|warm|small|faint|soft|bright|big) )?smile(?!\s+(?:lines|wrinkles))': '嘴角上扬。',
            r'(?:toothy smile|smile(?:s)?[, ]+showing (?:his |her |their )?teeth|smile(?:s)?[, ]+revealing (?:his |her |their )?teeth)': '嘴角上扬，露出牙齿。',
            r'(?:broad smile|wide smile|(?:smiling|smiles) broadly)': '嘴角明显上扬。',
            r'(?:furrowed (?:brows|eyebrows|brow)|(?:brows|eyebrows) are furrowed)': '眉间收紧。',
        },
    }
    for field, variants in specific.items():
        for phrase, chinese in variants.items():
            compiled[field].append((re.compile(rf'\b(?:{phrase})\b'), chinese))
    return compiled


RULES = make_rules()


def extract_observations(caption):
    matches = defaultdict(dict)
    normalized = caption.casefold().replace('–', '-').replace('—', ' ')
    for clause in CONTEXT_BREAK.split(normalized):
        if HEDGE.search(clause):
            continue
        for field, rules in RULES.items():
            if not MENTIONS[field].search(clause):
                continue
            for pattern, answer in rules:
                for match in pattern.finditer(clause):
                    # Skip quoted, hypothetical and reference language in this match's local context.
                    before = clause[max(0, match.start() - 55):match.start()]
                    after = clause[match.end():match.end() + 28]
                    if REFERENCE.search(before):
                        continue
                    if REFERENCE_AFTER.match(after):
                        continue
                    # Adjectives and nouns match directly. Missing fields have no row or null target.
                    matches[field].setdefault(answer, match.group(0))
    return matches


def collect(records, source):
    labels = defaultdict(lambda: defaultdict(dict))
    for row in records:
        for field, observations in extract_observations(row['caption']).items():
            for answer, phrase in observations.items():
                labels[row['name']][field].setdefault(answer, (phrase, row['reference']))
    pools = {field: [] for field in FIELD_NAMES}
    for name, fields in labels.items():
        for field, matches in fields.items():
            # Free-form annotators can disagree. Keep only fields with a single unambiguous target.
            if field == 'expression' and len(matches) > 1:
                smile_targets = ('嘴角上扬，露出牙齿。', '嘴角明显上扬。', '嘴角上扬。')
                if matches.keys() <= set(smile_targets):
                    best = next(answer for answer in smile_targets if answer in matches)
                    matches = {best: matches[best]}
            if len(matches) != 1:
                continue
            (answer, (phrase, reference)), = matches.items()
            pools[field].append({'id': f'{source.split()[0]}:{name}', 'source': source, 'name': name,
                                 'fields': {field: answer}, 'evidence': phrase, 'annotation_id': reference})
    return pools


def read_face2text(path):
    with ZipFile(path) as archive:
        partitions = {}
        for split in ('train', 'dev', 'test'):
            with archive.open(f'clean_{split}_2.1.json') as file:
                partitions[split] = [
                    {'name': row['filename'], 'caption': row['description'], 'reference': row['response_id']}
                    for row in json.load(file)
                ]
    split_names = [{row['name'] for row in rows} for rows in partitions.values()]
    if any(split_names[i].intersection(split_names[j]) for i in range(3) for j in range(i)):
        raise ValueError('Overlapping official Face2Text partitions')
    return partitions


def read_uniface(path):
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    result = []
    names = set()
    for row in rows:
        image = PurePosixPath(row['image'])
        if image.parent.name != 'MM-CelebA-HQ':
            continue
        if row['id'] != f'MM-CelebA-{image.stem}' or not re.fullmatch(r'\d+\.jpg', image.name):
            raise ValueError(f'Inconsistent MM image id: {row["id"]}')
        if image.name in names:
            continue
        names.add(image.name)
        result.append({'name': image.name, 'caption': row['caption'], 'reference': row['id']})
    return result


def read_mapping(path):
    lines = [line.split() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if lines[0][0] == 'idx':
        lines = lines[1:]
    mapping = {f'{int(idx)}.jpg': name for idx, original_index, name, *_ in lines
               if re.fullmatch(r'\d{6}\.jpg', name) and int(original_index) + 1 == int(Path(name).stem)}
    if len(mapping) != 30000 or len(set(mapping.values())) != 30000:
        raise ValueError(f'Expected complete one-to-one 30000-image HQ mapping in {path}')
    return mapping


def select(pools, quotas, accept, seed):
    rng = random.Random(seed)
    choices = {}
    for field, rows in pools.items():
        per_answer = defaultdict(list)
        for row in rows:
            per_answer[row['fields'][field]].append(row)
        for group in per_answer.values():
            rng.shuffle(group)
        choices[field] = deque(deque(group) for group in sorted(per_answer.values(), key=len))
    result = []
    counts = Counter()
    fields = sorted(pools, key=lambda field: len(pools[field]))
    while any(choices[field] and counts[field] < quotas[field] for field in fields):
        for field in fields:
            while choices[field] and counts[field] < quotas[field]:
                group = choices[field].popleft()
                row = group.popleft()
                if group:
                    choices[field].append(group)
                accepted = accept(row)
                if accepted is not None:
                    result.append(accepted)
                    counts[field] += 1
                    break
    return result


def build(args, pools, face, uni_test_names, mapping):
    identities = dict(line.split() for line in args.celeba_identity.read_text(encoding='utf-8-sig').splitlines() if line.strip())
    official_excluded = {row['name'] for split in ('dev', 'test') for row in face[split]}
    official_excluded.update(mapping[name] for name in uni_test_names)
    heldout_identities = set()
    originals = set()
    duplicates = DuplicateIndex()
    missing = Counter()
    mm_images = {path.name: path.resolve() for path in sorted(args.mm_images.rglob('*.jpg'))
                 if re.fullmatch(r'\d+\.jpg', path.name)} if args.mm_images else {}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    with ZipFile(args.celeba_zip) as archive:
        zipped = {PurePosixPath(item.filename).name: item for item in archive.infolist()
                  if not item.is_dir() and re.fullmatch(r'\d{6}\.jpg', PurePosixPath(item.filename).name)}
        if len(zipped) != CELEBA_IMAGE_COUNT:
            raise ValueError(f'Expected {CELEBA_IMAGE_COUNT} images in CelebA aligned archive, found {len(zipped)}')

        def accept(row, split):
            name = row['name']
            is_mm = row['source'] == UNIFACE
            original = mapping[name] if is_mm else name
            identity = identities[original]
            if original in originals or (split == 'train' and (original in official_excluded or identity in heldout_identities)):
                missing[split, row['source'], 'overlap'] += 1
                return None
            path = mm_images.get(name) if is_mm else None
            if path is None and is_mm and not args.allow_mapped_celeba:
                missing[split, row['source'], 'missing_hq'] += 1
                return None
            contents = path.read_bytes() if path is not None else archive.read(zipped[original])
            try:
                with Image.open(BytesIO(contents)) as image:
                    hashes = fingerprint(ImageOps.exif_transpose(image).convert('RGB'), contents)
            except OSError:
                missing[split, row['source'], 'unreadable'] += 1
                return None
            stable = {**row, **hashes}
            if duplicates.find(stable) is not None:
                missing[split, row['source'], 'image_near_duplicate'] += 1
                return None
            if path is None:
                path = output / 'celeba' / original
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(contents)
                image_variant = 'celeba_aligned_same_photo' if is_mm else 'celeba_aligned'
            else:
                image_variant = 'mm_celeba_hq'
            stable.update(image=Path(os.path.relpath(path, output)).as_posix(), image_variant=image_variant,
                          original_celeba=original)
            sample_prompt_and_target(stable)
            duplicates.add(stable)
            originals.add(original)
            if split == 'heldout':
                heldout_identities.add(identity)
            return stable

        selections = {}
        for index, ((split, source), candidates) in enumerate(pools.items()):
            quota = args.train_per_field if split == 'train' else (
                args.face_heldout_per_field if source == FACE2TEXT else args.uni_heldout_per_field)
            selections[split, source] = select(candidates, dict.fromkeys(FIELD_NAMES, quota),
                                               lambda row: accept(row, split), args.seed + index)
    return selections, missing, len(heldout_identities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face2text', type=Path, required=True)
    parser.add_argument('--uniface-train', type=Path, required=True)
    parser.add_argument('--uniface-test', type=Path, required=True)
    parser.add_argument('--celeba-zip', type=Path)
    parser.add_argument('--celeba-identity', type=Path)
    parser.add_argument('--hq-mapping', type=Path)
    parser.add_argument('--mm-images', type=Path)
    parser.add_argument('--allow-mapped-celeba', action='store_true',
                        help='Use the official HQ-to-CelebA mapping when an HQ image is missing')
    parser.add_argument('--plan', action='store_true', help='Report text candidates without image preparation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-per-field', type=int, default=250)
    parser.add_argument('--face-heldout-per-field', type=int, default=24)
    parser.add_argument('--uni-heldout-per-field', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()
    if not args.plan and not (args.output and args.celeba_zip and args.celeba_identity and args.hq_mapping
                             and (args.mm_images or args.allow_mapped_celeba)):
        parser.error('Image preparation needs output, CelebA archive/identities, HQ mapping and an image source')
    if min(args.train_per_field, args.face_heldout_per_field, args.uni_heldout_per_field) < 1:
        parser.error('Per-field targets must be positive')

    face = read_face2text(args.face2text)
    uni_train = read_uniface(args.uniface_train)
    uni_test = read_uniface(args.uniface_test)
    test_names = {row['name'] for row in uni_test}
    uni_train_disjoint = [row for row in uni_train if row['name'] not in test_names]
    pools = {
        ('heldout', FACE2TEXT): collect(face['dev'], FACE2TEXT),
        ('heldout', UNIFACE): collect(uni_test, UNIFACE),
        ('train', FACE2TEXT): collect(face['train'], FACE2TEXT),
        ('train', UNIFACE): collect(uni_train_disjoint, UNIFACE),
    }
    report = {
        'source_notes': {FACE2TEXT: 'Face2Text clean_train_2.1 and clean_dev_2.1; human-written',
                         UNIFACE: 'caption_train and caption_test; restrict to MM-CelebA-HQ; generated captions'},
        'source_links': {FACE2TEXT: 'https://zenodo.org/records/10973388',
                         UNIFACE: 'https://huggingface.co/datasets/tulvgengenr/UniF2aceD-1M',
                         'CelebA aligned': 'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'},
        'official_counts': {'face2text': {split: {'annotations': len(rows), 'images': len({r['name'] for r in rows})}
                                        for split, rows in face.items()},
                            'uniface_mm': {'train': len(uni_train), 'heldout': len(uni_test),
                                           'train_ids_removed_as_test': len(uni_train) - len(uni_train_disjoint)}},
        'text_candidates': {f'{split}/{source}': {field: len(rows) for field, rows in candidates.items()}
                            for (split, source), candidates in pools.items()},
    }
    if not args.plan:
        mapping = read_mapping(args.hq_mapping)
        selections, missing, n_identities = build(args, pools, face, test_names, mapping)
        for split in ('train', 'heldout'):
            rows = [row for (partition, _), group in selections.items() if partition == split for row in group]
            write_rows(args.output / f'public_{split}.jsonl', rows)
        report['selected'] = {
            f'{split}/{source}': {'total': len(rows), 'fields': dict(Counter(next(iter(row['fields'])) for row in rows)),
                                  'image_variants': dict(Counter(row['image_variant'] for row in rows))}
            for (split, source), rows in selections.items()
        }
        report['heldout_identities_excluded_from_training'] = n_identities
        report['rejections'] = {'/'.join(key): value for key, value in missing.items()}
        (args.output / 'public_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
