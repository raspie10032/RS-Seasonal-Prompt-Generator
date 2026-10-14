"""Prevent source/scene contamination and invalid authored targets."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from experiments.tipo_supervision import SCHEMA, freeze_supervision, validate_supervision
from tipo_prompt import VERSION


def source(number, dhash):
    return {'image_sha256': str(number)*64, 'intent_sha256': str(number+1)*64,
            'family_id': 'source-'+str(number), 'dhash64': dhash}


def reserved():
    return [dict(id=split, group_id=split, task='korean', split=split,
        purpose='evaluation', prompt_contract=VERSION,
        input={'prompt': 'winter'}, source=source(number, dhash),
        required=['winter'], excluded=['summer'])
        for split, number, dhash in [('dev', 1, '0'*16), ('final', 3, 'f'*16)]]


def training(split, number, dhash):
    return {'schema': SCHEMA, 'id': split, 'group_id': split, 'family': split,
            'mode': 'korean', 'split': split, 'purpose': 'training',
            'prompt_contract': VERSION, 'input': {'prompt': split, 'tag_length': 'short'},
            'source': source(number, dhash), 'required': ['winter', 'test artist'],
            'excluded': ['summer'], 'protected_names': {'test artist': 'artist'},
            'answer': 'winter, test artist, coat', 'min_tags': 3, 'max_tags': 30,
            'answer_origin': 'test-authored', 'review': {'author': 'test'}}


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.train = training('train', 5, 'a'*16)
        self.evaluation = training('eval', 7, 'c'*16)

    def check(self):
        return validate_supervision([self.train], [self.evaluation], reserved())

    def test_freeze_never_overwrites_and_checks_reserved_first(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'supervision'
            result = freeze_supervision([self.train, self.evaluation], reserved(), output)
            self.assertEqual(result['examples'], 2)
            self.assertEqual(len(result['files']['train.jsonl']), 64)
            with self.assertRaises(FileExistsError):
                freeze_supervision([self.train, self.evaluation], reserved(), output)
            self.train['source'] = deepcopy(reserved()[0]['source'])
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                freeze_supervision([self.train, self.evaluation], reserved(), Path(directory)/'bad')
            self.assertFalse((Path(directory)/'bad').exists())

    def test_source_split_and_near_duplicate_guards(self):
        self.evaluation['source']['image_sha256'] = self.train['source']['image_sha256']
        with self.assertRaisesRegex(ValueError, 'source leakage'):
            self.check()
        self.evaluation['source']['image_sha256'] = '7'*64
        self.evaluation['source']['dhash64'] = 'a'*15+'b'
        with self.assertRaisesRegex(ValueError, 'near-duplicate'):
            self.check()

    def test_scene_split_and_target_errors(self):
        self.evaluation['family'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Scene-family'):
            self.check()
        self.evaluation['family'] = 'eval'
        self.train['answer'] += ', summer'
        with self.assertRaisesRegex(ValueError, 'constraints'):
            self.check()

    def test_image_rows_not_silently_used_as_text(self):
        self.train['input']['image'] = '/private/image.png'
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            self.check()

    def test_repeated_targets_and_unknown_split_fail(self):
        self.train['answer'] += ', winter'
        with self.assertRaisesRegex(ValueError, 'constraints'):
            self.check()
        with tempfile.TemporaryDirectory() as directory:
            self.train['split'] = 'final'
            with self.assertRaisesRegex(ValueError, 'Unknown supervision split'):
                freeze_supervision([self.train, self.evaluation], reserved(), Path(directory)/'data')

    def test_reserved_final_is_required(self):
        with self.assertRaisesRegex(ValueError, 'both development and final'):
            validate_supervision([self.train], [self.evaluation], reserved()[:1])


if __name__ == '__main__':
    unittest.main()
