"""Validate and freeze private, authored text-expansion supervision.

Answers are editorial examples, not verified pixel captions. This module never
creates answers, downloads data, loads a model, or approves creative quality.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .gemma4_tuning_data import validate_splits
from .tipo_evaluation import (assert_training_disjoint, read_rows, score,
                              source_keys, validate)
from tipo_prompt import VERSION

SCHEMA = "tipo-authored-supervision/1"


def validate_supervision(train, evaluation, reserved, *, allow_images=False):
    if not train or not evaluation:
        raise ValueError("Both training and validation examples are required")
    validate(reserved)
    if {r['split'] for r in reserved} != {'dev', 'final'}:
        raise ValueError("Reserve both development and final evaluation")
    rows = [*train, *evaluation]
    ids = set()
    for split, subset in (('train', train), ('eval', evaluation)):
        for row in subset:
            if row.get('schema') != SCHEMA or row.get('prompt_contract') != VERSION:
                raise ValueError("Unknown supervision schema or prompt contract")
            if row.get('purpose') != 'training' or row.get('split') != split:
                raise ValueError("Wrong training purpose/split")
            if not row.get('id') or row['id'] in ids:
                raise ValueError("Missing or duplicate supervision ID")
            ids.add(row['id'])
            if not row.get('family') or not row.get('group_id'):
                raise ValueError("Missing scene/source family")
            if row.get('input', {}).get('image'):
                if not allow_images:
                    raise ValueError("Image supervision requires the multimodal pipeline")
                from .tipo_multimodal import validate_image_row
                validate_image_row(row)
            source = row.get('source', {})
            if not all(re.fullmatch(r'[a-f0-9]{64}', source.get(k, ''))
                       for k in ('image_sha256', 'intent_sha256')):
                raise ValueError("Invalid source digest")
            if not re.fullmatch(r'[a-f0-9]{16}', source.get('dhash64', '')):
                raise ValueError("Invalid source dHash")
            if not row.get('answer_origin') or not row.get('review', {}).get('author'):
                raise ValueError("Answer authorship and review status required")
            if not row.get('required') or not set(row.get('protected_names', {})) <= set(row['required']):
                raise ValueError("Missing required tags/protected names")
            if not score(row['answer'], row)['constraint_pass']:
                raise ValueError(f"Answer violates constraints: {row['id']}")
    if allow_images:
        from .gemma4_tuning_data import validate_row
        from .tipo_evaluation import digest
        for row in rows:
            if not row['input'].get('image'):
                validate_row(row)
        if {r['family'] for r in train} & {r['family'] for r in evaluation}:
            raise ValueError('Scene-family leakage')
        if {digest(r['input']) for r in train} & {digest(r['input']) for r in evaluation}:
            raise ValueError('Prompt leakage')
    else:
        validate_splits(train, evaluation)
    assert_training_disjoint(rows, reserved)
    # Split before deriving multiple instruction forms. Compare provenance and
    # near-duplicates across train/validation as well as against final evaluation.
    train_keys = set().union(*(source_keys(r) for r in train))
    if any(source_keys(r) & train_keys for r in evaluation):
        raise ValueError("Training/validation source leakage")
    a = {r['source']['dhash64'] for r in train}
    b = {r['source']['dhash64'] for r in evaluation}
    if any((int(x, 16) ^ int(y, 16)).bit_count() <= 6 for x in a for y in b):
        raise ValueError("Possible training/validation near-duplicate")
    return {'examples': len(rows), 'splits': {'train': len(train), 'eval': len(evaluation)},
            'sources': len({r['source']['image_sha256'] for r in rows}),
            'scene_families': len({r['family'] for r in rows}),
            'unique_answers': len({r['answer'] for r in rows}),
            'modes': dict(Counter(r['mode'] for r in rows)),
            'reserved_cases': len(reserved), 'automatic_constraint_checks': 'pass',
            'quality_scope': 'Structural and annotated constraints only; no human preference or pixel-ground-truth claim'}


def freeze_supervision(rows, reserved, output, *, allow_images=False):
    train = [r for r in rows if r.get('split') == 'train']
    evaluation = [r for r in rows if r.get('split') == 'eval']
    if len(train) + len(evaluation) != len(rows):
        raise ValueError("Unknown supervision split")
    summary = validate_supervision(train, evaluation, reserved, allow_images=allow_images)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, subset in (('train', train), ('eval', evaluation)):
        data = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in subset).encode()
        name = split + '.jsonl'
        (output / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    report = {'schema': SCHEMA, 'prompt_contract': VERSION, **summary, 'files': files,
              'image_supervision': any(r['input'].get('image') for r in rows),
              'image_rows': sum(bool(r['input'].get('image')) for r in rows),
              'human_preference_review': 'pending'}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--reserved-evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--multimodal', action='store_true', help='Require reviewed, hash-bound image inputs')
    args = parser.parse_args()
    reserved = [r for path in args.reserved_evaluation for r in read_rows(path)]
    print(json.dumps(freeze_supervision(read_rows(args.draft), reserved, args.output, allow_images=args.multimodal),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
