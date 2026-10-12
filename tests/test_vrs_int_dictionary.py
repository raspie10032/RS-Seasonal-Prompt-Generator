"""Fictional dictionary data only; source identity and selection invariants."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from vrs_int.generation_state import build_state, messages_for, reference_tags
from vrs_int.tag_dictionary import TagDictionary
from vrs_int.text_knowledge import ground_text, writer_vocabulary


class DictionaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fictional.csv'
        rows = [
            ['example_artist', 1, 5, '[ 작가 > 예시 ] 예시 작가. 키워드: <작가>, 예시화가'],
            ['example_character', 4, 9, '[캐릭터 > 예시] 예시 인물. 키워드: <인물>, 예시인물'],
            ['other_character', 4, 10, '[캐릭터 > 예시] 다른 인물.'],
            ['blue_eyes', 0, 10, '[ 인물 > 눈 ] 파란 눈. 키워드: <눈색>, 벽안, 파란 눈'],
            ['green_eyes', 0, 9, '[신체 > 눈] 초록 눈.'],
            ['red_hair', 0, 8, '[패션 > 헤어컬러] 붉은 머리.'],
            ['greenhouse', 0, 3, '[건물] 온실.'],
            ['winter', 0, 3, '[배경 > 계절] 겨울. 키워드: <계절>, [겨울]'],
            ['scarf', 0, 1, '[패션 > 액세서리] 목도리. 키워드: <의상>, 목도리'],
            ['example_wrap', 0, 2, '[의상 > 액세서리] 다른 목도리. 키워드: <의상>, 목도리'],
            ['example_book', 0, 5, '[사물 > 도구] 예시 책.'],
            ['umbrella', 0, 3, '[사물 > 소품] 비를 막는 도구. 키워드: <우산>'],
            ['rain', 0, 3, '[배경 > 자연] 비. 키워드: <강우>, 비'],
            ['example_artist_alias', 1, 2, '[작가 > 예시] 예시. 키워드: 우산, 유지'],
        ]
        with self.path.open('w', encoding='utf-8-sig', newline='') as stream:
            csv.writer(stream).writerows(rows)
        self.dictionary = TagDictionary(self.path)

    def recommendation(self):
        def row(tag, category='general', keys=('a',)):
            return {'tag': tag, 'categories': [category], 'score': .5,
                    'support': len(keys), 'observation_keys': list(keys)}
        return {'snapshot': 'a' * 64, 'candidates': [
            row('example artist', 'unknown'), row('example character', 'character'),
            row('other character', 'character', ('b',)), row('blue eyes'), row('green eyes'),
            row('red hair'), row('greenhouse'), row('scarf'), row('example book'),
            row('unlisted name', 'unknown'), row('best quality'), row('year 2025')],
            'identities': {}, 'neighbors': [{'key': 'a', 'similarity': .9}, {'key': 'b', 'similarity': .8}]}

    def test_csv_hash_taxonomy_and_korean_aliases(self):
        d = self.dictionary
        self.assertEqual(d.provenance['sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(d.lookup('BLUE_eyes').semantic_path, ('인물', '눈'))
        self.assertEqual(d.lookup('red hair').group, 'appearance')
        self.assertEqual(d.lookup('example book').group, 'objects')
        self.assertEqual(d.search('벽안')[0]['tag'], 'blue eyes')
        self.assertEqual(d.search('겨울')[0]['match'], 'exact')
        self.assertEqual({r['tag'] for r in d.search('목도리')}, {'scarf', 'example wrap'})

    def test_bad_csv_rejected_without_partial_dictionary(self):
        for text in ('', 'x,0,1\n', 'x,2,1,desc\n', 'x,0,-1,desc\n',
                     'some_tag,0,1,desc\nsome tag,0,2,desc\n'):
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                TagDictionary(self.path)

    def test_enrichment_is_immutable_and_does_not_change_votes(self):
        source = self.recommendation()
        before = deepcopy(source)
        enriched = self.dictionary.enrich(source)
        self.assertEqual(source, before)
        self.assertEqual(enriched['neighbors'], before['neighbors'])
        for original, row in zip(source['candidates'], enriched['candidates']):
            for key in ('tag', 'score', 'support', 'observation_keys'):
                self.assertEqual(original[key], row[key])
        self.assertEqual(enriched['candidates'][0]['categories'], ['artist'])
        self.assertEqual(enriched['candidates'][0]['source_categories'], ['unknown'])
        self.assertEqual(self.dictionary.enrich(enriched), enriched)

    def test_sentence_terms_keep_negation_and_ambiguous_alternatives(self):
        text = '예시인물은 파란눈으로, 겨울에 목도리는 빼줘. 미지의장소.'
        packet = ground_text(self.dictionary, text)
        by_text = {row['text']: row for row in packet['mentions']}
        self.assertEqual(by_text['파란눈']['candidates'][0]['tag'], 'blue eyes')
        self.assertEqual(by_text['예시인물']['candidates'][0]['category'], 'character')
        self.assertEqual({r['tag'] for r in by_text['목도리']['candidates']}, {'scarf', 'example wrap'})
        self.assertTrue(by_text['목도리']['ambiguous'])
        self.assertTrue(all(r['role'] == 'unresolved' for r in packet['mentions']))
        self.assertEqual(packet['input'], text)
        self.assertIn('미지의장소', ''.join(r['text'] for r in packet['unmatched_spans']))
        for row in packet['mentions']:
            self.assertEqual(text[row['start']:row['end']], row['text'])

    def test_class_label_is_weak_evidence_and_embedded_word_is_not_a_hit(self):
        packet = ground_text(self.dictionary, '우산, 비단, 비가 내려요')
        by_text = {row['text']: row for row in packet['mentions']}
        self.assertEqual(len([r for r in packet['mentions'] if r['text'] == '비']), 1)
        self.assertTrue(by_text['우산']['ambiguous'])
        self.assertEqual(by_text['우산']['candidates'][0]['match_source'], 'class_label')
        self.assertEqual(by_text['우산']['candidate_count'], 2)
        self.assertIn('example artist alias', {c['tag'] for c in by_text['우산']['candidates']})
        forwarded = writer_vocabulary(packet)['mentions'][0]
        self.assertNotIn('example artist alias', {c['tag'] for c in forwarded['meanings']})
        self.assertEqual(forwarded['alternatives_omitted'], 1)
        self.assertEqual(ground_text(self.dictionary, '벽안 유지하고')['mentions'][0]['text'], '벽안')
        self.assertEqual(len(ground_text(self.dictionary, '벽안 유지하고')['mentions']), 1)

    def test_translation_candidates_are_separate_from_visual_evidence_and_controls(self):
        state = build_state(self.recommendation(), '목도리는 빼고 벽안 유지',
                            dictionary=self.dictionary, keep=['blue eyes'], exclude=['scarf'])
        self.assertEqual(state['controls']['exclude'], ['scarf'])
        self.assertEqual(state['controls']['keep'], ['blue eyes'])
        self.assertEqual(state['controls']['target'], [])
        self.assertEqual(state['request_knowledge']['input'], state['request'])
        vocabulary = []
        for mode in ('request', 'flat', 'state'):
            user = messages_for(state, mode)[1]['content']
            vocabulary.append(user.split('Request vocabulary: ')[1].split('\nReference')[0])
        self.assertEqual(len(set(vocabulary)), 1)
        self.assertEqual(json.loads(vocabulary[0]), writer_vocabulary(state['request_knowledge']))

    def test_text_limit_truncation_and_english_reverse_lookup(self):
        packet = ground_text(self.dictionary, 'blue_eyes, 목도리', limit=1)
        self.assertEqual(packet['mentions'][0]['candidates'][0]['tag'], 'blue eyes')
        self.assertIn('파란 눈', packet['mentions'][0]['candidates'][0]['description_ko'])
        self.assertTrue(packet['mentions'][1]['candidates_truncated'])
        self.assertTrue(packet['mentions'][1]['ambiguous'])
        self.assertEqual(packet['mentions'][1]['candidate_count'], 2)
        for text in ('', ' ' * 3, '가' * 4097):
            with self.assertRaises(ValueError):
                ground_text(self.dictionary, text)

    def test_explicit_source_identity_wins_over_dictionary_conflict(self):
        row = self.dictionary.annotate({'tag': 'blue eyes', 'categories': ['artist']})
        self.assertEqual(row['categories'], ['artist'])
        self.assertEqual(row['dictionary']['status'], 'category_conflict')
        self.assertIsNone(row['dictionary']['group'])
        source = self.recommendation()
        source['candidates'] = [row]
        state = build_state(source, 'A park', dictionary=self.dictionary)
        self.assertEqual(state['identity_candidates']['artist'][0]['tag'], 'blue eyes')

    def test_new_dictionary_cannot_silently_replace_prior_semantics(self):
        enriched = self.dictionary.enrich(self.recommendation())
        enriched['tag_dictionary']['sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            self.dictionary.enrich(enriched)

    def test_dictionary_expands_groups_without_creating_visual_evidence(self):
        state = build_state(self.recommendation(), 'A park', dictionary=self.dictionary)
        self.assertIn('example book', reference_tags(state))
        self.assertIn('example artist', reference_tags(state))
        self.assertNotIn('unlisted name', reference_tags(state))
        self.assertEqual(state['visual_verdict'], 'U')
        self.assertEqual(state['dictionary_coverage']['category_additions'], ['example artist'])

    def test_resolved_references_keep_all_evidence_but_choose_controls(self):
        state = build_state(self.recommendation(), 'Winter market',
                            dictionary=self.dictionary, resolve=True,
                            keep=['example character', 'example artist', 'blue eyes'],
                            target=['winter', 'scarf', 'night market'])
        tags = reference_tags(state)
        self.assertIn('example artist', tags)
        self.assertIn('example character', tags)
        self.assertNotIn('other character', tags)
        self.assertNotIn('green eyes', tags)
        self.assertNotIn('greenhouse', tags)
        self.assertNotIn('best quality', tags)
        self.assertEqual(len(state['identity_candidates']['character']), 2)
        self.assertIn('night market', state['controls']['target'])
        self.assertEqual(state['writer_reference']['anchor_observation_key'], 'a')
        reasons = {r['tag']: r['reason'] for r in state['writer_reference']['rejected']}
        self.assertEqual(reasons['green eyes'], 'conflicting_attribute')

    def test_resolved_flat_and_state_share_selected_reference_set(self):
        state = build_state(self.recommendation(), 'A park', dictionary=self.dictionary, resolve=True)
        flat = messages_for(state, 'flat')[1]['content'].split('Reference tags: ')[1]
        structured = json.loads(messages_for(state, 'state')[1]['content'].split('Reference state: ')[1])
        tags = {t for rows in structured['visual_reference_candidates'].values() for t in rows}
        tags.update(row['tag'] for rows in structured['identity_candidates'].values() for row in rows)
        self.assertEqual(set(flat.split(', ')), tags)
        self.assertEqual(tags, set(reference_tags(state)))
        self.assertFalse(any(structured['identity_candidates'].values()))

    def test_contradictory_explicit_colors_need_a_declared_exception(self):
        for keep in (['blue eyes', 'green eyes'], ['2girls']):
            with self.assertRaises(ValueError):
                build_state(self.recommendation(), 'A park', dictionary=self.dictionary, resolve=True, keep=keep)
        state = build_state(self.recommendation(), 'A park', dictionary=self.dictionary, resolve=True,
                            keep=['blue eyes', 'green eyes', 'heterochromia'])
        self.assertIn('heterochromia', state['controls']['keep'])
        with self.assertRaises(ValueError):
            build_state(self.recommendation(), 'A park', resolve=True)

    def test_original_recommendation_hash_is_used_even_after_enrichment(self):
        source = self.recommendation()
        state = build_state(source, 'A park', dictionary=self.dictionary)
        encoded = json.dumps(source, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(state['recommendation_sha256'], hashlib.sha256(encoded).hexdigest())

    def test_explicit_fourth_identity_is_not_lost_to_candidate_quota(self):
        source = self.recommendation()
        source['candidates'] = [
            {'tag': f'fictional character {i}', 'categories': ['character'],
             'observation_keys': [str(i)]} for i in range(4)]
        source['neighbors'] = [{'key': str(i), 'similarity': .9} for i in range(4)]
        state = build_state(source, 'A park', dictionary=self.dictionary,
                            resolve=True, keep=['fictional character 3'])
        self.assertEqual(len(state['identity_candidates']['character']), 4)
        self.assertEqual(reference_tags(state), ['fictional character 3'])
        self.assertEqual(state['writer_reference']['anchor_observation_key'], '3')


if __name__ == '__main__':
    unittest.main()
