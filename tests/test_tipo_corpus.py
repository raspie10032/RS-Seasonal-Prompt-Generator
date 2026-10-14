"""Data selection, action supervision, and recipe splitting regressions."""
from copy import deepcopy
import csv
import json
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from experiments.tipo_corpus import (STYLES, build_rows, select_sources, style_evidence,
                                     validate_recipes)
from vrs_int.generation_metadata import make_record


def recipe(family):
    return dict(family=family, subject='person', ko='컵을 들고 서 있게.',
                core=['standing', 'holding cup', 'cup'], actions=['holding cup'],
                exclude=['kneeling', 'sitting'], old=['beach'],
                details=['coat', 'pants', 'boots', 'balcony', 'railing', 'tree', 'cloud', 'sky'],
                variants=[['sunlight', 'from side', 'full body']])


def sources(count=40):
    rows = []
    for i in range(count):
        rows.append(dict(subject='male' if i % 4 == 0 else 'female',
                         source={'image_sha256': f'{i:064x}', 'family_id': f'family-{i}'},
                         names={'artist': ['artist name'], 'character': ['character name'],
                                'copyright': ['series name']},
                         positive_intent_tags=['blue hair', 'blue eyes', 'swimsuit'],
                         style_evidence='novelai-generation-intent; pixels-unreviewed'))
    return rows


class CorpusTests(unittest.TestCase):
    def test_catalog_selection_is_read_only_and_reserves_evaluation_images(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            image = p/'source.png'
            Image.new('RGB', (9, 8), 'white').save(image)
            sha = hashlib.sha256(image.read_bytes()).hexdigest()
            gen = {'format': 'novelai', 'characters': [], 'negative': [],
                   'positive': ['1girl, artist:a, character:c, copyright:s, -1::photo::']}
            record = make_record(gen, str(image), sha)
            db = p/'catalog.sqlite3'
            with sqlite3.connect(db) as con:
                con.execute('CREATE TABLE records (sha TEXT, path TEXT, record BLOB)')
                con.execute('INSERT INTO records VALUES (?, ?, ?)',
                            (sha, str(image), zlib.compress(json.dumps(record).encode())))
            before = db.read_bytes()
            selected, report = select_sources(db, [], [], {'female': 1})
            self.assertEqual(len(selected), 1)
            self.assertTrue(report['complete'])
            self.assertNotIn('photo', selected[0]['positive_intent_tags'])
            self.assertEqual(db.read_bytes(), before)
            blocked = {'source': {'image_sha256': sha, 'dhash64': 'f'*16}}
            selected, report = select_sources(db, [blocked], [], {'female': 1})
            self.assertFalse(selected)
            self.assertFalse(report['complete'])
            self.assertEqual(report['selection_counts']['reserved_or_previous_image'], 1)

    def test_positive_style_screen_keeps_camera_action_and_negative_style_separate(self):
        generation = {'format': 'novelai', 'characters': [], 'negative': ['photo'],
                      'positive': ['1girl, holding camera, taking picture, -1::realistic::']}
        r = make_record(generation, '/unused.png', 'a'*64)
        self.assertIsNotNone(style_evidence(r)[0])
        generation['positive'] = ['1girl, realistic']
        r = make_record(generation, '/unused.png', 'a'*64)
        self.assertEqual(style_evidence(r), (None, ['realistic']))
        generation['positive'] = ['1girl, artist:photo']
        self.assertIsNotNone(style_evidence(make_record(generation, '/unused.png', 'a'*64))[0])
        generation['format'] = 'unknown'
        self.assertIsNone(style_evidence(make_record(generation, '/unused.png', 'a'*64))[0])

    def test_source_and_scene_split_before_variants_with_bound_actions(self):
        rows = build_rows(sources(60), [recipe(str(i)) for i in range(5)], per_scene=8, male_scene_stride=1)
        self.assertEqual(len(rows), 240)
        for field in ('family', 'group_id'):
            a = {r[field] for r in rows if r['split'] == 'train'}
            b = {r[field] for r in rows if r['split'] == 'eval'}
            self.assertFalse(a & b)
        self.assertTrue(any(r['subject'] == 'male' and r['target_style'] == '3d' for r in rows))
        self.assertTrue(any('artist' not in r['protected_names'].values() for r in rows))
        self.assertTrue(any('artist' in r['protected_names'].values() for r in rows))
        self.assertTrue(all(r['source_names']['artist'] == ['artist name'] for r in rows))
        self.assertTrue(any(r['subject'] == 'female' and r['target_style'] == '3d' for r in rows))
        for r in rows:
            self.assertIn('holding cup', r['answer'].split(', '))
            self.assertNotIn('swimsuit', r['answer'])
            self.assertTrue(set(r['protected_names']) <= set(r['required']))
            self.assertFalse(r['input'].get('image'))
            self.assertNotIn('beach', r['answer'])
            if r['mode'] == 'edit':
                self.assertIn('beach', r['excluded'])
            if r['mode'] == 'repair':
                draft = r['input']['prompt'].split('잘못된 초안: ')[1].split('.\n')[0]
                self.assertNotIn('holding cup', draft.split(', '))

    def test_vocabulary_audit_refuses_unknown_action_or_contradiction(self):
        r = recipe('one')
        tags = r['core'] + r['details'] + r['variants'][0] + sum((s[1] for s in STYLES.values()), [])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'tags.csv'
            with p.open('w', newline='') as stream:
                csv.writer(stream).writerows((t, 0) for t in tags)
            report = validate_recipes([r], p)
            self.assertFalse(report['unknown_recipe_target_tags'])
            bad = deepcopy(r)
            bad['core'].append('invented action')
            with self.assertRaisesRegex(ValueError, 'missing from dictionary'):
                validate_recipes([bad], p)
            bad = deepcopy(r)
            bad['actions'] = ['invented action']
            with self.assertRaisesRegex(ValueError, 'Action missing'):
                validate_recipes([bad], p)
            bad = deepcopy(r)
            bad['variants'][0].append('sitting')
            with self.assertRaisesRegex(ValueError, 'Contradictory'):
                validate_recipes([bad], p)

    def test_duplicate_sources_and_insufficient_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, 'Repeated source'):
            build_rows(sources()+sources()[:1], [recipe('one')])
        with self.assertRaisesRegex(ValueError, 'Insufficient sources'):
            build_rows([], [recipe('one')])
        with self.assertRaisesRegex(ValueError, 'Positive scene size'):
            build_rows([], [recipe('one')], male_scene_stride=0)


if __name__ == '__main__':
    unittest.main()
