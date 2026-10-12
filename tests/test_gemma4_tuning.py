"""Instruction supervision, held-out evaluation and exact-score boundaries."""
from copy import deepcopy
import unittest

from experiments.gemma4_tuning_data import make_rows, score_output, validate_row, validate_splits
from experiments.train_gemma4_tipo import encode_example, eval_subset


class ToyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        value = ''.join(f"<{m['role']}>" + m['content'] + '</turn>' for m in messages)
        return value + ('<assistant>' if add_generation_prompt else '')

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class TuningTests(unittest.TestCase):
    def test_scene_families_and_prompt_inputs_are_held_out(self):
        train, evaluation = make_rows()
        validate_splits(train, evaluation)
        self.assertEqual(len(train), 240)
        self.assertEqual(len(evaluation), 116)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_splits(train, [train[0]])

    def test_invalid_supervision_is_rejected(self):
        train, _ = make_rows()
        row = deepcopy(train[0])
        row['excluded'].append(row['required'][0])
        with self.assertRaisesRegex(ValueError, 'Contradictory'):
            validate_row(row)
        row = deepcopy(train[0])
        row['messages'][-1]['content'] += ', ' + row['excluded'][0]
        with self.assertRaisesRegex(ValueError, 'violates'):
            validate_row(row)

    def test_korean_negation_pairs_share_vocabulary_but_change_target(self):
        train, _ = make_rows()
        rows = [r for r in train if r['family'] == 'korean_scarf' and 'blonde hair' in r['required']]
        self.assertEqual(len(rows), 2)
        self.assertTrue(any('scarf' in r['required'] for r in rows))
        self.assertTrue(any('scarf' in r['excluded'] for r in rows))
        for row in rows:
            self.assertIn('"목도리": "scarf"', row['messages'][1]['content'])

    def test_prompt_labels_masked_no_silent_truncation(self):
        row = {'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                            {'role': 'assistant', 'content': 'answer'}]}
        encoded = encode_example(ToyTokenizer(), row, 1000)
        p = encoded['prompt_length']
        self.assertEqual(encoded['labels'][:p], [-100] * p)
        self.assertEqual(encoded['labels'][p:], encoded['input_ids'][p:])
        self.assertEqual(chr(encoded['labels'][p]), 'a')
        with self.assertRaisesRegex(ValueError, 'truncation'):
            encode_example(ToyTokenizer(), row, 5)

    def test_template_prefix_mismatch_fails(self):
        class BadTokenizer(ToyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return 'different' if len(messages) == 3 else 'prefix'
        row = {'messages': [{'role': role, 'content': 'x'} for role in ('system','user','assistant')]}
        with self.assertRaisesRegex(ValueError, 'prefix'):
            encode_example(BadTokenizer(), row, 1000)

    def test_eval_subset_represents_families_and_modes(self):
        _, rows = make_rows()
        subset = eval_subset(rows, 12)
        self.assertEqual(len(subset), 12)
        self.assertEqual({r['family'] for r in subset}, {r['family'] for r in rows})
        self.assertEqual({r['mode'] for r in subset}, {'request','flat','state','korean_only'})

    def test_scoring_detects_reintroduction_and_conflict(self):
        result = score_output('blue eyes, green eyes, scarf, best quality', ['blue eyes'], ['scarf'])
        self.assertFalse(result['pass'])
        self.assertEqual(result['forbidden'], ['scarf'])
        self.assertTrue(result['conflicts'])
        self.assertTrue(result['quality_year'])
        self.assertTrue(score_output('blue eyes, coat', ['blue eyes'], ['scarf'])['pass'])


if __name__ == '__main__':
    unittest.main()
