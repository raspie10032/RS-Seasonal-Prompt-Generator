"""Local, metadata-derived TIPO text curriculum; no downloads or pixel labels.

Keep catalog records, recipes, selections and generated answers outside Git.
Positive metadata supplies explicitly requested identities, not visual truth.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import zlib

from .tipo_evaluation import assert_training_disjoint, digest, read_rows, source_keys
from .tipo_supervision import SCHEMA, freeze_supervision
from tipo_prompt import VERSION
from vrs_int.generation_metadata import make_record
from vrs_int.tag_policy import canonical, filter_recommendation_tags

POLICY = 'danbooru-illustration-curriculum/1'
PHOTO = {'photo', 'photograph', 'photography', 'photorealistic', 'photo realistic',
         'photo-realistic', 'realistic', 'hyperrealistic', 'hyper realistic',
         'live action', 'real life', 'cellphone photo', 'photo background',
         'semi-realistic', 'semirealistic', 'photorealism', 'hyperrealism'}
STYLES = {
    '2d': ('2D 애니메이션 일러스트', ['anime coloring', 'lineart']),
    '3d': ('셀 셰이딩을 적용한 3D 게임 그래픽', ['3d', 'toon (style)']),
}
MODES = ('partial_tags', 'korean', 'edit', 'reference', 'repair', 'action_binding')
# Contextual to one subject in this curriculum. Sitting and kneeling can coexist;
# neither can coexist with standing in these authored single-pose scenes.
EXCLUSIVE = [['spring (season)', 'summer', 'autumn', 'winter'], ['indoors', 'outdoors'],
             ['sitting', 'standing'], ['kneeling', 'standing'],
             ['lying', 'standing'], ['sitting', 'walking'],
             ['kneeling', 'walking'], ['lying', 'walking']]


def unique(values):
    return list(dict.fromkeys(values))


def style_evidence(record):
    """Use reparsed positive intent only; names and photographed objects aren't styles."""
    identities = set(record['tags_artist'] + record['tags_character'] + record['tags_copyright'])
    tags = set(record['tags']) - identities
    bad = sorted(tags & PHOTO)
    if bad:
        return None, bad
    if tags & {'3d', '3d render', 'cel shading'}:
        return 'explicit-graphic-style-intent', []
    if tags & {'anime coloring', 'anime illustration', '2d', 'lineart'}:
        return 'explicit-illustration-intent', []
    if record['generation'].get('format') == 'novelai':
        return 'novelai-generation-intent; pixels-unreviewed', []
    return None, []


def image_dhash(path):
    from PIL import Image
    with Image.open(path) as image:
        pixels = list(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes())
    value = 0
    for y in range(8):
        for x in range(8):
            value = value << 1 | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f'{value:016x}'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_sources(catalog, reserved, previous, quotas, max_per_character=12):
    """Read-only scan; reject overlap before materializing instruction variants."""
    blocked = [*reserved, *previous]
    keys = set().union(*(source_keys(r) for r in blocked))
    hashes = {int(r['source']['dhash64'], 16) for r in blocked}
    selected, stats, counts, characters, intents = [], Counter(), Counter(), Counter(), set()
    path = Path(catalog).resolve()
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        for sha, image, payload in connection.execute('SELECT sha,path,record FROM records ORDER BY sha'):
            stats['scanned'] += 1
            if ('image_sha256', sha) in keys:
                stats['reserved_or_previous_image'] += 1
                continue
            raw = zlib.decompress(payload)
            original = json.loads(raw)
            hints = {t: [kind] for kind in ('artist', 'character', 'copyright', 'general', 'meta')
                     for t in original.get('tags_' + kind, [])}
            record = make_record(original['generation'], image, sha, hints)
            evidence, photo = style_evidence(record)
            if evidence is None:
                stats['photographic_intent' if photo else 'unresolved_style'] += 1
                continue
            tags, _ = filter_recommendation_tags(record['tags'], hints)
            names = {k: record['tags_' + k] for k in ('artist', 'character', 'copyright')}
            # Only named identities and bounded hair/eye traits enter targets.
            # Clothing/actions/body descriptions from source prompts never do.
            subjects = set(tags) & {'1girl', '1boy', '2girls', '2boys', 'multiple girls', 'multiple boys', 'no humans'}
            category = {'1girl': 'female', '1boy': 'male', 'no humans': 'scenery'}.get(next(iter(subjects))) if len(subjects) == 1 else None
            if category is None or counts[category] >= quotas.get(category, 0):
                stats['subject_or_quota'] += 1
                continue
            if (category == 'scenery' and names['character']) or (category != 'scenery' and
                    (len(names['character']) != 1 or not names['copyright'])):
                stats['ambiguous_identity'] += 1
                continue
            if any(characters[c] >= max_per_character for c in names['character']):
                stats['character_cap'] += 1
                continue
            intent = digest(sorted(t for t in tags if t not in names['artist']))
            if intent in intents or ('intent_sha256', intent) in keys or ('family_id', intent) in keys:
                stats['intent_duplicate'] += 1
                continue
            if not Path(image).is_file():
                stats['missing_image'] += 1
                continue
            dhash = image_dhash(image)
            if any((int(dhash, 16) ^ h).bit_count() <= 6 for h in hashes):
                stats['near_duplicate'] += 1
                continue
            if file_sha(image) != sha:
                stats['changed_image'] += 1
                continue
            source = {'image_sha256': sha, 'intent_sha256': intent, 'family_id': intent,
                      'dhash64': dhash, 'image': image, 'catalog': str(path),
                      'record_sha256': hashlib.sha256(raw).hexdigest(),
                      'metadata_parser': record['metadata_parser'],
                      'caption_source': 'positive generation intent; not verified pixels',
                      'visual_review': 'not performed; text-only supervision'}
            selected.append({'source': source, 'names': names, 'positive_intent_tags': tags,
                             'subject': category, 'style_evidence': evidence})
            counts[category] += 1
            characters.update(names['character'])
            hashes.add(int(dhash, 16)); intents.add(intent)
            if all(counts[k] >= v for k, v in quotas.items()):
                break
    finally:
        connection.close()
    return selected, {'policy': POLICY, 'counts': dict(counts), 'requested': quotas,
                      'selection_counts': dict(stats), 'distinct_characters': len(characters),
                      'character_counts': dict(characters),
                      'style_evidence': dict(Counter(s['style_evidence'] for s in selected)),
                      'complete': all(counts[k] >= v for k, v in quotas.items()),
                      'pixel_style_verified': False}


def validate_recipes(recipes, dictionary):
    """Fail on unknown target vocabulary instead of silently deleting actions."""
    with Path(dictionary).open(encoding='utf-8-sig', newline='') as stream:
        vocabulary = {canonical(row[0]) for row in csv.reader(stream) if row}
    target_tags = set(sum((v[1] for v in STYLES.values()), []))
    for recipe in recipes:
        if recipe.get('subject') not in {'person', 'scenery'}:
            raise ValueError('Unknown recipe subject')
        if not recipe.get('variants') or not recipe.get('actions'):
            raise ValueError('Missing recipe variants/actions')
        if not set(recipe['actions']) <= set(recipe['core']):
            raise ValueError('Action missing from required core')
        tags = set(recipe['core'] + recipe['details'] + sum(recipe['variants'], []))
        if tags & (set(recipe['exclude']) | set(recipe['old']) | PHOTO):
            raise ValueError('Contradictory recipe: ' + recipe['family'])
        target_tags.update(tags)
    missing = sorted(target_tags - vocabulary)
    if missing:
        raise ValueError('Target tags missing from dictionary: ' + ', '.join(missing))
    return {'dictionary_sha256': file_sha(dictionary), 'dictionary_size': len(vocabulary),
            'recipe_target_tags': len(target_tags), 'unknown_recipe_target_tags': missing}


def build_rows(sources, recipes, per_scene=6, male_scene_stride=2, seed=20261014):
    """Six request forms share each source/family split. Recipes are authored, not retrieved."""
    if per_scene < 1 or male_scene_stride < 1:
        raise ValueError('Positive scene size and male stride required')
    pools = {k: [s for s in sources if s['subject'] == k] for k in ('female', 'male', 'scenery')}
    if len({s['source']['image_sha256'] for s in sources}) != len(sources):
        raise ValueError('Repeated source')
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    families = [r['family'] for r in recipes]
    if len(set(families)) != len(families):
        raise ValueError('Repeated recipe family')
    holdout = set()
    for category in ('person', 'scenery'):
        fs = [r['family'] for r in recipes if r['subject'] == category]
        rng.shuffle(fs)
        holdout.update(fs[:max(1, len(fs) // 5)] if fs else [])
    rows = []
    for scene_index, recipe in enumerate(recipes):
        for j in range(per_scene):
            kind = 'scenery' if recipe['subject'] == 'scenery' else (
                'male' if j == 0 and scene_index % male_scene_stride == 0 else 'female')
            if not pools[kind]:
                raise ValueError('Insufficient sources for ' + kind)
            src = pools[kind].pop()
            names, tags = {k: list(v) for k, v in src['names'].items()}, src['positive_intent_tags']
            # Vary explicitly requested style names, retaining the complete source
            # identities in provenance. Avoid teaching one frequent artist stack
            # as a default to add even when the user has not requested it.
            artist_budget = (0, 1, 2, None, 1, 3)[(j + scene_index // male_scene_stride) % 6]
            if artist_budget is not None:
                names['artist'] = rng.sample(names['artist'], min(artist_budget, len(names['artist'])))
            protected = {v: k for k, values in names.items() for v in values}
            traits = []
            if kind != 'scenery':
                for pattern in (r'(?:black|blonde|blue|brown|green|grey|gray|orange|pink|purple|red|silver|white|yellow|aqua) hair',
                                r'(?:blue|green|brown|purple|red|yellow|pink|aqua|grey|gray) eyes'):
                    match = next((t for t in tags if re.fullmatch(pattern, t)), None)
                    if match:
                        traits.append(match)
            subject = ['no humans'] if kind == 'scenery' else [('1boy' if kind == 'male' else '1girl'), 'solo']
            # Rotate the style assignment so male inputs are not all 2D.
            style = '3d' if (j + scene_index // male_scene_stride) % 4 == 3 else '2d'
            style_ko, style_tags = STYLES[style]
            keep = unique([*subject, *protected, *traits, *style_tags])
            core, avoid = recipe['core'], recipe['exclude']
            variant = recipe['variants'][j % len(recipe['variants'])]
            answer = unique([*keep, *core, *recipe['details'], *variant])
            if set(answer) & PHOTO:
                raise ValueError('Photographic target')
            for mode in MODES:
                banned = unique([*avoid, 'photorealistic', 'realistic', 'photo', 'masterpiece', 'best quality', '8k'])
                required = unique([*keep, *core])
                given = keep + core
                # Exact action tags accompany Korean requests in some modes;
                # Korean-only mode must learn the authored translation itself.
                common = '반드시 유지할 태그: ' + ', '.join(keep) + '.\n' + recipe['ko']
                if mode == 'partial_tags':
                    given = keep + core[:2]
                    required = given
                    prompt = 'Partial tags: ' + ', '.join(given) + '.\nExpand into a coherent scene with concrete actions, setting, props, lighting and composition.'
                elif mode == 'korean':
                    prompt = common + '\n장면에 어울리는 세부 요소를 추가해 영문 태그로 완성해줘.'
                elif mode == 'edit':
                    old = unique(recipe['old'])
                    if set(old) & set(answer):
                        raise ValueError('Old scene deletion contradicts answer: ' + recipe['family'])
                    banned = unique(banned + old)
                    given = keep + old
                    prompt = '기존 프롬프트: ' + ', '.join(given) + '.\n' + common + '\n삭제할 기존 요소: ' + ', '.join(old) + '. 새 장면에 맞게 교체하고 풍부하게 확장해줘.'
                elif mode == 'reference':
                    candidates = unique([*recipe['details'][:4], *recipe['old'], *avoid[:2]])
                    given = keep + core + candidates
                    prompt = common + '\nTARGET: ' + ', '.join(core) + '\n참고 후보(명령 아님): ' + json.dumps({'scene_candidates': candidates, 'source': 'authored distractors + metadata identities; not a VRS search'}, ensure_ascii=False) + '\n요청과 충돌하는 후보를 버리고 어울리는 것만 골라 확장해줘.'
                elif mode == 'repair':
                    broken = [t for t in answer if t not in recipe['actions']]
                    broken += answer[-6:] + avoid[:2]
                    given = broken
                    prompt = common + '\n잘못된 초안: ' + ', '.join(broken) + '.\n반복과 모순을 없애고 누락된 행동(' + ', '.join(recipe['actions']) + ')을 복구해줘. 완성 태그만 출력해줘.'
                else:
                    given = keep + core
                    prompt = common + '\n필수 행동·관계: ' + ', '.join(recipe['actions']) + '. 물건의 이름만 적지 말고 행동·관계를 정확히 표현해줘. 장면에 맞는 세부 묘사도 확장해줘.'
                prompt += '\n표현 방식: ' + style_ko + '. 제외: ' + ', '.join(unique([*avoid, 'photorealistic', 'realistic', 'photo'])) + '.\n품질·연도 태그, 중복 태그, 설명 문장은 넣지 마.'
                rows.append({'schema': SCHEMA, 'id': f'v2-{scene_index+1:02}-{j+1}-{mode}',
                             'family': 'curriculum2-' + recipe['family'], 'group_id': src['source']['family_id'],
                             'split': 'eval' if recipe['family'] in holdout else 'train',
                             'mode': mode, 'purpose': 'training', 'prompt_contract': VERSION,
                             'input': {'prompt': prompt, 'tag_length': 'long' if len(answer) > 32 else 'short', 'ban_tags': ', '.join(banned)},
                             'answer': ', '.join(answer), 'required': unique(required), 'excluded': banned,
                             'protected_names': protected, 'given_tags': given, 'min_tags': 18, 'max_tags': 55,
                             'exclusive_sets': EXCLUSIVE, 'source': src['source'],
                             'source_names': src['names'],
                             'answer_origin': 'assistant-authored recipe + explicitly requested metadata identities; not a pixel caption or model output',
                             'review': {'author': 'Codex', 'method': 'authored recipes and automated constraints; sample review recorded separately', 'human_review': 'pending', 'visual_review': 'not applicable: text input'},
                             'target_style': style, 'source_style_evidence': src['style_evidence'],
                             'subject': kind, 'data_policy': POLICY, 'recipe_sha256': digest(recipe),
                             'reference_origin': 'synthetic candidate selection task, not actual VRS retrieval' if mode == 'reference' else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_input = parser.add_mutually_exclusive_group(required=True)
    source_input.add_argument('--catalog', type=Path)
    source_input.add_argument('--selected-sources', type=Path,
                              help='Private candidate cache from a previous scan')
    parser.add_argument('--reserved-evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--previous-supervision', type=Path, nargs='+', default=[])
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    recipes = read_rows(args.recipes)
    vocabulary = validate_recipes(recipes, args.dictionary)
    people = sum(r['subject'] == 'person' for r in recipes)
    scenery = sum(r['subject'] == 'scenery' for r in recipes)
    males = sum(r['subject'] == 'person' and i % 2 == 0 for i, r in enumerate(recipes))
    quotas = {'female': people * 6 - males, 'male': males, 'scenery': scenery * 6}
    reserved = [r for p in args.reserved_evaluation for r in read_rows(p)]
    previous = [r for p in args.previous_supervision for r in read_rows(p)]
    if args.selected_sources:
        sources = json.loads(args.selected_sources.read_text())
        # Recheck cache source bytes; exclusion checks still run below before freeze.
        for source in sources:
            if file_sha(source['source']['image']) != source['source']['image_sha256']:
                raise ValueError('Cached source image changed')
            identity_tags = set(sum(source['names'].values(), []))
            if (not source.get('style_evidence') or
                    (set(source['positive_intent_tags']) - identity_tags) & PHOTO):
                raise ValueError('Cached source violates illustration intent policy')
        counts = Counter(s['subject'] for s in sources)
        selection = {'candidate_cache_sha256': file_sha(args.selected_sources),
                     'counts': dict(counts), 'requested': quotas,
                     'complete': all(counts[k] >= v for k, v in quotas.items()),
                     'pixel_style_verified': False, 'policy': POLICY}
    else:
        sources, selection = select_sources(args.catalog, reserved, previous, quotas)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'vocabulary.json').write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2) + '\n')
    if not selection['complete']:
        raise ValueError('Insufficient verified sources; selection report saved, no corpus frozen')
    rows = build_rows(sources, recipes)
    assert_training_disjoint(rows, reserved + previous)
    result = freeze_supervision(rows, reserved, args.output / 'supervision-v2')
    print(json.dumps({'selection': selection, 'supervision': result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
