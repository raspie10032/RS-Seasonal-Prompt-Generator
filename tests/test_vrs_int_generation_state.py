"""Fictional references only: source preservation, identity and matched-arm tests."""
from copy import deepcopy
import json
import unittest

from vrs_int.generation_state import build_state, messages_for, reference_tags


def fixture():
    def row(tag, category, keys):
        return {'tag': tag, 'categories': [category], 'score': .5,
                'support': len(keys), 'observation_keys': keys}
    return {'snapshot': 'a' * 64, 'contract_sha256': 'b' * 64,
            'candidates': [row('best quality', 'general', ['a']),
                           row('year 2024', 'meta', ['a']),
                           row('blue hair', 'general', ['a']),
                           row('red hair', 'general', ['b']),
                           row('greenhouse', 'general', ['a']),
                           row('unknown creator', 'unknown', ['a']),
                           row('snow', 'unknown', ['a'])],
            'identities': {'artist': [row('newest', 'artist', ['a'])],
                           'character': [row('example character', 'character', ['a'])],
                           'copyright': [row('example series', 'copyright', ['a'])]},
            'neighbors': [{'key': 'a', 'similarity': .8}, {'key': 'b', 'similarity': .7}]}


class StateTests(unittest.TestCase):
    def test_source_immutable_quality_filtered_identity_preserved(self):
        source = fixture()
        before = deepcopy(source)
        state = build_state(source, 'Move to a library')
        self.assertEqual(source, before)
        tags = reference_tags(state)
        self.assertNotIn('best quality', tags)
        self.assertNotIn('year 2024', tags)
        self.assertIn('newest', tags)
        self.assertEqual(state['visual_verdict'], 'U')
        self.assertEqual(state['omitted_counts']['quality_year'], 2)

    def test_unknown_never_promoted_to_names_or_visual_evidence(self):
        state = build_state(fixture(), 'A park')
        self.assertNotIn('unknown creator', reference_tags(state))
        self.assertNotIn('snow', reference_tags(state))
        self.assertEqual(state['omitted_counts']['unclassified'], 2)

    def test_cooccurrence_is_shared_observation_not_union(self):
        state = build_state(fixture(), 'A park')
        first, second = state['cooccurrence']
        self.assertIn('blue hair', first['tags'])
        self.assertNotIn('red hair', first['tags'])
        self.assertEqual(second['tags'], ['red hair'])

    def test_explicit_constraints_and_source_provenance(self):
        state = build_state(fixture(), 'A park', keep=['Example_Character'],
                            target=['green_coat'], exclude=['red hair'])
        self.assertEqual(state['controls']['keep'], ['example character'])
        self.assertNotIn('red hair', reference_tags(state))
        self.assertEqual(len(state['recommendation_sha256']), 64)
        self.assertEqual(state['snapshot'], 'a' * 64)
        with self.assertRaises(ValueError):
            build_state(fixture(), 'A park', keep=['blue_hair'], exclude=['blue hair'])

    def test_flat_and_state_have_identical_reference_tag_set(self):
        state = build_state(fixture(), 'A park', keep=['example character'])
        flat = messages_for(state, 'flat')
        structured = messages_for(state, 'state')
        self.assertEqual(flat[0], structured[0])
        self.assertEqual(flat[1]['content'].split('\nReference')[0], structured[1]['content'].split('\nReference')[0])
        payload = json.loads(structured[1]['content'].split('Reference state: ', 1)[1])
        tags = {r['tag'] for rows in payload['identity_candidates'].values() for r in rows}
        tags.update(t for rows in payload['visual_reference_candidates'].values() for t in rows)
        self.assertEqual(tags, set(reference_tags(state)))
        for bundle in payload['cooccurring_examples']:
            self.assertTrue(set(bundle) <= tags)
        self.assertNotIn('Reference', messages_for(state, 'request')[1]['content'])

    def test_snapshot_request_and_quota_validation(self):
        source = fixture()
        source['snapshot'] = 'CURRENT'
        with self.assertRaises(ValueError):
            build_state(source, 'A park')
        for request, quota in [('', 5), ('A park', 0), ('A park', 13)]:
            with self.assertRaises(ValueError):
                build_state(fixture(), request, per_group=quota)

    def test_tipo_native_framing_keeps_same_explicit_controls(self):
        from tipo_engine import SYSTEM_TIPO
        state = build_state(fixture(), 'A winter park', keep=['example character'],
                            target=['winter'], exclude=['greenhouse'])
        for mode in ('request', 'flat', 'state'):
            messages = messages_for(state, mode, framing='tipo-partial')
            self.assertEqual(messages[0]['content'], SYSTEM_TIPO)
            self.assertTrue(messages[1]['content'].startswith('Partial tags: example character, winter\n'))
            self.assertIn('Avoid these tags: greenhouse.', messages[1]['content'])
        with self.assertRaises(ValueError):
            messages_for(state, 'state', framing='unknown')


if __name__ == '__main__':
    unittest.main()
