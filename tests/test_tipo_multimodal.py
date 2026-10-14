"""Protect reviewed pixels, source identity and explicit image training opt-in."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments.tipo_multimodal import validate_image_row
from tipo_prompt import VERSION, text_messages


class ImageSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'reviewed.png'
        self.path.write_bytes(b'fixed source bytes')
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.row = {'prompt_contract': VERSION, 'purpose': 'training',
                    'input': {'image': str(self.path), 'prompt': '배경만 바꿔줘'},
                    'source': {'image': str(self.path), 'image_sha256': self.sha},
                    'visual_required': ['blue hair'], 'required': ['blue hair', 'winter'],
                    'visual_review': {'image_sha256': self.sha, 'reviewer': 'test',
                                      'style': '2d', 'observed_tags': ['blue hair']}}

    def test_reviewed_pixels_cannot_be_replaced(self):
        validate_image_row(self.row)
        self.path.write_bytes(b'changed pixels')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            validate_image_row(self.row)

    def test_unreviewed_visual_labels_are_not_ground_truth(self):
        self.row['visual_required'].append('red eyes')
        with self.assertRaisesRegex(ValueError, 'reviewed observations'):
            validate_image_row(self.row)

    def test_visual_labels_must_be_scored(self):
        self.row['required'] = ['winter']
        with self.assertRaisesRegex(ValueError, 'must be scored'):
            validate_image_row(self.row)

    def test_review_bound_to_exact_source(self):
        for key, value in [('image_sha256', '0'*64), ('reviewer', ''), ('style', 'photo')]:
            row = deepcopy(self.row)
            row['visual_review'][key] = value
            with self.assertRaisesRegex(ValueError, 'visual review'):
                validate_image_row(row)

    def test_no_text_only_downgrade_or_cached_messages(self):
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            text_messages(self.row['input'], 'blue hair, winter')
        self.row['messages'] = [{'role': 'user', 'content': 'old caption'}]
        with self.assertRaisesRegex(ValueError, 'no stored text substitute'):
            validate_image_row(self.row)

    def test_no_different_input_or_evaluation_target(self):
        self.row['input']['image'] = str(self.path.with_name('other.png'))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_image_row(self.row)
        self.row['input']['image'] = str(self.path)
        self.row['purpose'] = 'evaluation'
        with self.assertRaisesRegex(ValueError, 'training contract'):
            validate_image_row(self.row)


if __name__ == '__main__':
    unittest.main()
