"""Regression coverage for supervision masking and evaluation contamination."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.tipo_evaluation import (assert_training_disjoint, freeze, score,
                                         score_results, validate)
from experiments.train_gemma4_tipo import encode_example
from tipo_prompt import VERSION, build_user_prompt, row_messages, text_messages
from gemma_tipo_native_node import build_prompt


def case():
    return {"id": "one", "group_id": "g1", "task": "korean", "split": "dev",
            "purpose": "evaluation", "prompt_contract": VERSION,
            "input": {"prompt": "겨울 밤, 목도리 제외", "tag_length": "short"},
            "source": {"image_sha256": "1"*64, "intent_sha256": "2"*64,
                       "family_id": "f1", "dhash64": "0000000000000000"},
            "required": ["winter", "test artist"], "excluded": ["scarf", "summer"],
            "protected_names": {"test artist": "artist"}, "given_tags": ["winter"],
            "min_tags": 2, "max_tags": 10,
            "exclusive_sets": [["sitting", "standing"]]}


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return ''.join('<'+m['role']+'>'+m['content']+'</turn>' for m in messages) + ('<assistant>' if add_generation_prompt else '')

    def encode(self, text, **kw):
        return list(text.encode())


class EvaluationTests(unittest.TestCase):
    def test_native_training_same_instruction_and_no_target_leak(self):
        inputs = {"prompt": "겨울", "tag_length": "short", "ban_tags": "scarf"}
        messages = text_messages(inputs, "winter, coat")
        self.assertEqual(messages[0]["content"], build_prompt("겨울", "short", "scarf", False))
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        row = {"prompt_contract": VERSION, "input": inputs, "answer": "winter, coat"}
        encoded = encode_example(Tokenizer(), row, 10000)
        p = encoded['prompt_length']
        self.assertEqual(encoded['labels'][:p], [-100]*p)
        self.assertTrue(bytes(encoded['labels'][p:]).startswith(b'winter, coat'))
        self.assertNotIn('winter, coat', messages[0]['content'])

    def test_image_supervision_cannot_be_silently_text(self):
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            text_messages({"prompt": "image", "image": "/private/image.png"}, "answer")

    def test_stale_templates_and_evaluation_training_rejected(self):
        row = {"prompt_contract": VERSION, "input": {"prompt": "hello"}, "answer": "world",
               "messages": [{"role": "user", "content": "old"}]}
        with self.assertRaisesRegex(ValueError, 'differ'):
            row_messages(row)
        with self.assertRaisesRegex(ValueError, 'Evaluation'):
            row_messages(case())

    def test_family_and_source_split_leaks(self):
        first = case(); second = deepcopy(first)
        second.update(id='two', split='final')
        with self.assertRaisesRegex(ValueError, 'family'):
            validate([first, second])
        second['group_id'] = 'g2'
        with self.assertRaisesRegex(ValueError, 'crosses'):
            validate([first, second])

    def test_training_source_and_near_duplicate_guard(self):
        first = case(); train = deepcopy(first)
        train.update(purpose='training', group_id='g2')
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            assert_training_disjoint([train], [first])
        train['source'].update(image_sha256='3'*64, intent_sha256='4'*64, family_id='f2', dhash64='0000000000000001')
        with self.assertRaisesRegex(ValueError, 'near-duplicate'):
            assert_training_disjoint([train], [first])
        train['source']['dhash64'] = 'f'*16
        assert_training_disjoint([train], [first])
        del train['source']['dhash64']
        with self.assertRaisesRegex(ValueError, 'provenance'):
            assert_training_disjoint([train], [first])

    def test_raw_failure_is_not_repaired_by_postprocess(self):
        row = case()
        report = score_results([row], [{"id": "one", "raw": "winter, test artist, scarf",
                                        "postprocessed": "winter, test artist"}])
        self.assertEqual(report['raw_constraint_passes'], 0)
        self.assertTrue(report['results'][0]['postprocessed']['constraint_pass'])
        self.assertIsNone(report['results'][0]['raw']['creative_quality'])

    def test_repetition_conflicts_quality_and_truncation(self):
        row = case()
        result = score('winter, test artist, winter, sitting, standing, best quality, 2025', row)
        self.assertFalse(result['constraint_pass'])
        self.assertTrue(result['duplicate_tags'])
        self.assertTrue(result['conflicts'])
        self.assertTrue(result['quality_year'])
        report = score_results([row], [{"id": "one", "raw": "winter, test artist", "truncated": True}])
        self.assertEqual(report['raw_constraint_passes'], 0)

    def test_aliases_and_tag_boundaries(self):
        row = case(); row['required'].append('bookstore'); row['aliases'] = {'bookstore': ['bookshop']}
        self.assertTrue(score('winter, test artist, bookshop, scarf print', row)['constraint_pass'])
        self.assertFalse(score('winter, test artist, bookshop, scarf', row)['constraint_pass'])

    def test_incomplete_or_duplicate_results_not_silent(self):
        row = case()
        self.assertFalse(score_results([row], [])['complete'])
        output = {'id': 'one', 'raw': 'winter, test artist'}
        with self.assertRaisesRegex(ValueError, 'repeated'):
            score_results([row], [output, output])

    def test_freeze_is_non_overwriting_and_detects_near_duplicates(self):
        first = case(); second = deepcopy(first)
        second.update(id='two', split='final', group_id='g2')
        second['source'].update(image_sha256='3'*64, intent_sha256='4'*64, family_id='f2')
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'evaluation'
            with self.assertRaisesRegex(ValueError, 'near-duplicate'):
                freeze([first, second], out)
            self.assertFalse(out.exists())
            second['source']['dhash64'] = 'f'*16
            manifest = freeze([first, second], out)
            self.assertEqual(manifest['cases'], 2)
            with self.assertRaises(FileExistsError):
                freeze([first, second], out)


if __name__ == '__main__':
    unittest.main()
