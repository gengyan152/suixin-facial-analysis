"""Compare the base and LoRA on held-out public photos with single-region prompts."""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import random
import re
import time

from common import DuplicateIndex, MODEL_ID, MODEL_REVISION, fingerprint, open_rgb, read_rows, sample_prompt_and_target, sha256
from prepare_public import RULES


# Each key is a reference phrase from prepare_public.py. Only clear lexical equivalents are credited.
WORDS = {
    '眉毛较细。': r'(?:眉[毛形]?(?:较|比[较較]|偏|很|略|稍|纤)*细|细眉|眉毛[^，。]{0,4}纤细)',
    '眉毛较浓密。': r'(?:眉毛?[^，。]{0,6}(?:浓密|浓厚|粗浓|茂密|浓黑)|浓眉|眉毛?浓)',
    '眉毛带有弧度。': r'(?:眉毛?[^，。]{0,10}(?:弧形|有弧度|带弧度|上弯|弯曲|较弯)|弯眉|弧形眉)',
    '眉毛走向较平直。': r'(?:眉[毛形]?[^，。]{0,7}(?:平直|较直|平缓|平坦)|一字眉|平眉)',
    '眉毛较稀疏。': r'眉毛?[^，。]{0,6}(?:稀疏|稀少|较疏)',
    '眼形偏圆。': r'(?:眼睛?|眼形|眼型)[^，。]{0,8}(?:偏圆|较圆|圆形|圆润)|圆眼',
    '眼形呈杏仁状。': r'杏仁[形状]|杏[眼仁]',
    '眼睛较大。': r'(?:眼睛?|双眼|两眼)(?:较|比较|比[較]|偏|很|略|稍)*大|大眼睛',
    '眼睛较小。': r'(?:眼睛?|双眼|两眼)(?:较|比较|比[較]|偏|很|略|稍)*小|小眼睛',
    '眼睛开合较窄。': r'(?:眼睛?|双眼|两眼)[^，。]{0,6}(?:眯起|眯着|微眯|开合较窄|睁得较窄|偏窄)|眯眼',
    '双眼睁得较开。': r'(?:眼睛?|双眼|两眼)[^，。]{0,8}(?:睁得(?:较|很)?[开大]|睁开较多|睁大)|睁大[双两]?眼',
    '双眼闭合。': r'(?:眼睛?|双眼|两眼)[^，。]{0,5}(?:闭合|闭着|紧闭|闭上)|闭着[双两]?眼|闭[双两]?眼',
    '眼窝较深。': r'眼窝[^，。]{0,6}(?:深|凹陷)|深眼窝',
    '双眼间距较宽。': r'(?:眼距|眼睛?间距|双眼间距)[^，。]{0,5}(?:宽|远|大)',
    '双眼间距较窄。': r'(?:眼距|眼睛?间距|双眼间距)[^，。]{0,5}(?:窄|近|小)',
    '鼻部较宽。': r'鼻[部翼子]?[^，。]{0,6}(?:宽阔|较宽|偏宽|稍宽)|宽鼻',
    '鼻部较窄。': r'鼻[部翼子]?[^，。]{0,6}(?:较窄|偏窄|略窄|狭窄)|窄鼻',
    '鼻部较长。': r'鼻[部子]?[^，。]{0,6}(?:较长|偏长|稍长|修长)|长鼻',
    '鼻部较短。': r'鼻[部子]?[^，。]{0,6}(?:较短|偏短|稍短)|短鼻',
    '鼻梁较直。': r'鼻[梁子]?[^，。]{0,6}(?:较直|偏直|笔直|直挺|平直|直的)|笔直的鼻梁',
    '鼻子较小。': r'鼻[子部]?[^，。]{0,6}(?:小巧|较小|偏小)|小鼻子',
    '鼻子较大。': r'鼻[子部]?[^，。]{0,6}(?:较大|偏大|略大)|大鼻子',
    '鼻尖偏尖。': r'鼻[尖头子]?[^，。]{0,6}(?:偏尖|较尖|尖尖|尖细)|尖鼻[尖头子]',
    '鼻尖偏圆。': r'鼻[尖头子]?[^，。]{0,6}(?:偏圆|较圆|圆润|圆钝)|圆鼻[头尖]',
    '鼻尖微翘。': r'鼻[尖头子]?[^，。]{0,6}(?:微翘|上翘|翘起)|翘鼻',
    '嘴唇较薄。': r'(?:嘴唇|双唇|唇形|唇部)[^，。]{0,6}(?:较薄|偏薄|略薄|薄)|薄唇',
    '嘴唇较饱满。': r'(?:嘴唇|双唇|唇形|唇部)[^，。]{0,7}(?:饱满|丰盈|丰厚|丰润|较厚|偏厚)|厚唇',
    '双唇分开。': r'(?:嘴唇|双唇|唇部|嘴巴)[^，。]{0,7}(?:分开|张开|微张|张着)|张着嘴|微张的双唇',
    '双唇合拢。': r'(?:嘴唇|双唇|嘴巴)[^，。]{0,7}(?:合拢|合上|闭合|闭着|合在)|闭着嘴',
    '双唇抿起。': r'(?:嘴唇|双唇|嘴巴)[^，。]{0,6}(?:抿起|抿紧|抿着)|抿嘴',
    '脸部轮廓偏圆。': r'脸(?:部轮廓|型|形)?[^，。]{0,6}(?:偏圆|较圆|圆形|圆润)|圆脸',
    '脸部轮廓呈椭圆形。': r'椭圆(?:形|脸)|鹅蛋脸|卵圆',
    '脸部轮廓偏方。': r'脸(?:部轮廓|型|形)?[^，。]{0,6}(?:偏方|较方|方形)|方脸',
    '脸部轮廓呈心形。': r'心形(?:脸|的脸)|脸(?:部轮廓|型|形)?[^，。]{0,6}心形',
    '脸部轮廓偏长。': r'脸(?:部轮廓|型|形)?[^，。]{0,6}(?:偏长|较长|略长|狭长)|长脸',
    '脸部轮廓棱角较明显。': r'棱角(?:分明|较明显|明显)|轮廓[^，。]{0,7}有棱角',
    '下颌轮廓较清晰。': r'(?:下颌|下巴线条)[^，。]{0,9}(?:清晰|分明|明显|突出)',
    '下颌轮廓偏方。': r'下颌[^，。]{0,8}(?:偏方|较方|方正|方形)|方下颌',
    '下巴偏尖。': r'下巴[^，。]{0,6}(?:偏尖|较尖|尖尖|尖细)|尖下巴',
    '下巴偏圆。': r'下巴[^，。]{0,6}(?:偏圆|较圆|圆钝|圆润)|圆下巴',
    '颧骨较突出。': r'颧骨[^，。]{0,6}(?:突出|偏高|较高|明显)|高颧骨',
    '嘴角上扬。': r'嘴角[^，。]{0,7}(?:上扬|上翘|上提|翘起|向上)|微笑|笑容|带笑|咧嘴笑',
    '嘴角明显上扬。': r'嘴角[^，。]{0,6}(?:明显|显著|大幅)[^，。]{0,4}(?:上扬|上翘|上提)|咧嘴笑|大笑|灿烂的笑容',
    '眉间收紧。': r'(?:眉间|眉毛)[^，。]{0,8}(?:收紧|皱起|向内靠拢)|皱眉|蹙眉',
    '嘴角上扬，露出牙齿。': r'露出(?:一排|一口|一些|部分|几颗)?[牙齿齿]{1,2}|露齿|露牙',
}
PATTERNS = {label: re.compile(words) for label, words in WORDS.items()}
FIELD_LABELS = {field: {label for _, label in rules} for field, rules in RULES.items()}
OPPOSITIONS = (
    ('眉毛带有弧度。', '眉毛走向较平直。'), ('眉毛较浓密。', '眉毛较稀疏。'),
    ('眼睛较大。', '眼睛较小。'), ('双眼睁得较开。', '双眼闭合。'),
    ('眼睛开合较窄。', '双眼睁得较开。'), ('双眼间距较宽。', '双眼间距较窄。'),
    ('鼻部较宽。', '鼻部较窄。'), ('鼻部较长。', '鼻部较短。'), ('鼻子较小。', '鼻子较大。'),
    ('鼻尖偏尖。', '鼻尖偏圆。'), ('嘴唇较薄。', '嘴唇较饱满。'), ('双唇分开。', '双唇合拢。'),
    ('双唇分开。', '双唇抿起。'), ('脸部轮廓偏圆。', '脸部轮廓偏方。'),
    ('脸部轮廓偏圆。', '脸部轮廓偏长。'), ('下巴偏尖。', '下巴偏圆。'),
)
OPPOSITE = defaultdict(set)
for left, right in OPPOSITIONS:
    OPPOSITE[left].add(right)
    OPPOSITE[right].add(left)


def parse_answer(raw, field):
    fenced = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', raw, re.IGNORECASE)
    try:
        value = json.loads(fenced[1] if fenced else raw, object_pairs_hook=list)
    except (TypeError, ValueError):
        return False, False, None
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], tuple) and value[0][0] == field):
        return False, False, None
    answer = value[0][1]
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        return False, False, None
    return True, not fenced, answer


def semantic_labels(answer):
    if not isinstance(answer, str):
        return set()
    clauses = [clause for clause in re.split(r'[，,。.!！；;、？?]|但是|不过|而是', answer) if not re.search(
        r'不(?:算|是|太|够|很|见|清楚|明显|像|大|小|宽|窄|直|厚|薄|高|长|短|尖|圆|浓|疏)|没有|并非|未见|难以|看不(?:清|到|出)|可能|似乎', clause
    )]
    labels = {label for label, pattern in PATTERNS.items() if any(pattern.search(clause) for clause in clauses)}
    if '嘴角上扬。' not in labels:
        labels.discard('嘴角上扬，露出牙齿。')
    return labels


def relation(expected, labels):
    if expected not in PATTERNS:
        return 'no_reference_rule'
    if OPPOSITE[expected].intersection(labels):
        return 'conflict'
    if expected in labels:
        return 'match'
    return 'unrecognized_or_unrelated'


def select(rows, limit, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda value: value['id']):
        field, = row['fields']
        groups[row['source'], field].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    strata = sorted(groups)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for key in strata:
            if groups[key] and len(selected) < limit:
                previous = [row['fields'][key[1]] for row in selected if row['source'] == key[0] and key[1] in row['fields']]
                index = min(range(len(groups[key])), key=lambda i: previous.count(groups[key][i]['fields'][key[1]]))
                selected.append(groups[key].pop(index))
    return selected


def swaps(rows, cap):
    pairs = []
    # Pair only contrasts in the same source, crop format and requested field.
    remaining = list(rows)
    while remaining and len(pairs) < cap:
        left = remaining.pop(0)
        field, = left['fields']
        target = left['fields'][field]
        match = next((i for i, right in enumerate(remaining) if right['source'] == left['source']
                      and right['image_variant'] == left['image_variant']
                      and right['fields'].get(field) in OPPOSITE[target]), None)
        if match is not None:
            pairs.append((left, remaining.pop(match)))
    return pairs


def summarize(records):
    def split_rows(rows):
        metrics = {}
        for variant in ('base', 'adapter'):
            subset = [row[variant] for row in rows]
            available = len(subset)
            count = lambda key: sum(bool(value[key]) for value in subset)
            counts = Counter(value['annotated_relation'] for value in subset)
            metrics[variant] = {
                'n': available, 'strict_json_rate': count('strict_json') / available,
                'parseability_rate': count('parseable') / available,
                'lexicon_output_coverage_rate': count('recognized_labels') / available,
                'reference_rule_coverage': (available - counts['no_reference_rule']) / available,
                'annotated_positive_match_rate': counts['match'] / available, 'relation_counts': dict(counts),
                'generation_errors': count('error'),
            }
        return metrics

    ordinary = [row for row in records if row['condition'] == 'same_image']
    negative = [row for row in records if row['condition'] == 'swapped_image']
    results = {'same_image': split_rows(ordinary) if ordinary else {}, 'by_source': {}, 'by_field': {}}
    for name in ('source', 'field'):
        for value in sorted({row[name] for row in ordinary}):
            results[f'by_{name}'][value] = split_rows([row for row in ordinary if row[name] == value])
    ordinary_by_id = {row['id']: row for row in ordinary}
    results['swapped_image'] = {}
    for variant in ('base', 'adapter'):
        paired = [row for row in negative if row['id'] in ordinary_by_id]
        results['swapped_image'][variant] = {
            'n_directions': len(paired),
            'retains_old_attribute': sum(row[variant]['original_relation'] == 'match' for row in paired),
            'agrees_with_replacement': sum(row[variant]['annotated_relation'] == 'match' for row in paired),
            'both_correct_and_changes_attribute': sum(row[variant]['annotated_relation'] == 'match'
                and row[variant]['original_relation'] != 'match'
                and ordinary_by_id[row['id']][variant]['annotated_relation'] == 'match' for row in paired),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-manifest', action='append', type=Path, default=[])
    parser.add_argument('--training-stats', type=Path)
    parser.add_argument('--limit', type=int, default=48)
    parser.add_argument('--swap-pairs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--max-seconds', type=int, default=780)
    parser.add_argument('--per-answer-seconds', type=int, default=15)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.limit <= 48 and 0 <= args.swap_pairs <= 24 and args.max_seconds > 0 and args.per_answer_seconds > 0):
        parser.error('Use a positive time limit, 1–48 images and at most 24 swap pairs')
    if not args.plan and not (args.base_model and args.adapter):
        parser.error('A local base snapshot and adapter are required')

    rows = list(read_rows(args.manifest))
    if not rows or len(rows) != len({row['id'] for row in rows}):
        parser.error('The evaluation manifest must contain unique sample identifiers')
    excluded = DuplicateIndex()
    for path in args.exclude_manifest:
        for row in read_rows(path):
            excluded.add(row)
    if args.training_stats:
        stats = json.loads(args.training_stats.read_text(encoding='utf-8'))
        excluded.ids.update(stats['train_ids'])
        excluded.ids.update(stats['dev_ids'])
    if any(duplicate := excluded.find(row) for row in rows):
        parser.error(f'Held-out public manifest overlaps with training or dev: {duplicate}')

    selected = select(rows, args.limit, args.seed)
    pairs = swaps(selected, args.swap_pairs)
    protocol = {
        'image_count': len(selected), 'manifest_sha256': sha256(args.manifest.read_bytes()), 'seed': args.seed,
        'same_image_ids': [row['id'] for row in selected],
        'strata': dict(Counter(f'{row["source"]}/{next(iter(row["fields"]))}' for row in selected)),
        'swap_pairs': [(left['id'], right['id']) for left, right in pairs],
        'swap_interpretation': 'Bilateral swaps of an incompatible captioned attribute, within source, field, and crop format.',
        'scoring': 'Only clear positive lexical matches count. Unrecognized phrases and caption incompleteness can lower scores.',
        'lexicon': WORDS, 'base': MODEL_ID, 'revision': MODEL_REVISION, 'max_new_tokens': 64,
        'decoder': 'greedy', 'image_tiles': 1, 'max_seconds': args.max_seconds,
        'per_answer_seconds': args.per_answer_seconds,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.plan:
        print(json.dumps({key: protocol[key] for key in ('image_count', 'strata', 'swap_pairs')}, ensure_ascii=False, indent=2))
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList

    if not torch.cuda.is_available():
        parser.error('CUDA GPU required for base and adapter inference')
    started = time.monotonic()
    deadline = started + args.max_seconds

    class Deadline(StoppingCriteria):
        def __init__(self, until):
            self.until = until

        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() >= self.until

    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.image_processor.max_patches = 1
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=dtype, local_files_only=True,
                                                       attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True, is_trainable=False).to('cuda').eval()
    if model.peft_config['default'].base_model_name_or_path != MODEL_ID:
        raise ValueError('Adapter belongs to another base model')

    images = {}
    for row in selected:
        path = args.manifest.parent / row['image']
        image = open_rgb(path)
        hashes = fingerprint(image, path.read_bytes())
        if any(digest != row[key] for key, digest in hashes.items()):
            raise ValueError(f'Image fingerprint changed: {path}')
        images[row['id']] = image

    tasks = [(row, row) for row in selected]
    tasks += [(origin, replacement) for pair in pairs for origin, replacement in (pair, pair[::-1])]
    records = []
    with (args.output / 'public_eval.jsonl').open('w', encoding='utf-8', buffering=1) as output:
        for index, (row, picture) in enumerate(tasks, start=1):
            if deadline <= time.monotonic():
                break
            field, = row['fields']
            expected = picture['fields'][field]
            original = row['fields'][field]
            prompt, _ = sample_prompt_and_target(row)
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': images[picture['id']]},
                                                       {'type': 'text', 'text': prompt}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors='pt').to('cuda', dtype=dtype)
            record = {'id': row['id'], 'condition': 'same_image' if row is picture else 'swapped_image',
                      'image_id': picture['id'], 'crop_sha256': picture['sha256'], 'source': row['source'],
                      'field': field, 'expected': expected, 'original_expected': original, 'prompt': prompt}
            for variant in ('base', 'adapter'):
                start_answer = time.monotonic()
                until = min(deadline, start_answer + args.per_answer_seconds)
                raw, error = '', None
                try:
                    if until <= start_answer:
                        raise TimeoutError('evaluation time budget exhausted')
                    with torch.inference_mode(), (model.disable_adapter() if variant == 'base' else nullcontext()):
                        sequence = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True,
                                                  pad_token_id=processor.tokenizer.eos_token_id,
                                                  stopping_criteria=StoppingCriteriaList([Deadline(until)]))
                    raw = processor.decode(sequence[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    if time.monotonic() >= until:
                        raise TimeoutError('answer exceeded its time limit')
                except (TimeoutError, ValueError, RuntimeError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                parseable, strict, answer = (False, False, None) if error else parse_answer(raw, field)
                labels = semantic_labels(answer) & FIELD_LABELS[field]
                record[variant] = {'raw': raw, 'error': error, 'parseable': parseable, 'strict_json': strict,
                                   'answer': answer, 'recognized_labels': sorted(labels),
                                   'annotated_relation': relation(expected, labels),
                                   'original_relation': relation(original, labels),
                                   'elapsed_ms': round((time.monotonic() - start_answer) * 1000, 1)}
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            if index % 8 == 0 or index == len(tasks):
                print(f'{index}/{len(tasks)} paired queries', flush=True)
    results = {'completed': len(records) == len(tasks), 'expected_queries': len(tasks), 'complete_queries': len(records),
               'wall_seconds': round(time.monotonic() - started, 2), 'gpu': torch.cuda.get_device_name(),
               'dtype': str(dtype), 'adapter_sha256': sha256((args.adapter / 'adapter_model.safetensors').read_bytes()),
               **summarize(records)}
    (args.output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
