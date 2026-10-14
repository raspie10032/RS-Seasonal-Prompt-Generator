"""Two-format native contracts; no model, private corpus, or downloads."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from gemma_prompt_expand_node import GemmaPromptExpandNative
from prompt_expansion import build_expansion_prompt, clean_natural_language
from tag_knowledge import SCHEMA, TagKnowledge, load_knowledge


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.clip = Mock()
        self.clip.tokenize.return_value = {"gemma4": []}
        self.args = dict(clip=self.clip, prompt="파란 머리는 유지하고 겨울에 서서 차를 들게, 목도리 제외",
                         tag_length="short", max_tokens=512, temperature=0, ban_tags="scarf", seed=42)

    def test_both_reuses_the_expanded_scene_without_second_image_encoding(self):
        scene = "Create a winter scene. A woman, dressed in blue, holds a cup."
        self.clip.decode.side_effect = [scene, "1girl, standing, holding cup, scarf, standing"]
        image = SimpleNamespace(shape=(1, 32, 32, 3))
        tags, prose, raw = GemmaPromptExpandNative().run(**self.args, image=image)
        self.assertEqual(prose, scene)
        self.assertEqual(tags, "1girl, standing, holding cup")
        calls = self.clip.tokenize.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0].kwargs["image"], image)
        self.assertIsNone(calls[1].kwargs["image"])
        self.assertIn(scene, calls[1].args[0])
        self.assertIn(self.args["prompt"], calls[1].args[0])
        self.assertIn("scarf", json.loads(raw)["tags"])

    def test_single_modes_do_not_pay_for_unused_output(self):
        self.clip.decode.return_value = "Create a scene with a woman, a cup, and a window."
        tags, prose, raw = GemmaPromptExpandNative().run(**self.args, output_format="natural_language")
        self.assertEqual(tags, "")
        self.assertEqual(prose, self.clip.decode.return_value)
        self.assertIsNone(json.loads(raw)["tags"])
        self.assertEqual(self.clip.generate.call_count, 1)
        self.clip.generate.reset_mock()
        self.clip.decode.return_value = "1girl, winter"
        tags, prose, raw = GemmaPromptExpandNative().run(**self.args, output_format="tags")
        self.assertEqual(tags, "1girl, winter")
        self.assertEqual(prose, "")
        self.assertEqual(self.clip.generate.call_count, 1)

    def test_empty_wrong_mode_and_runtime_failure(self):
        self.assertEqual(GemmaPromptExpandNative().run(**dict(self.args, prompt="")), ("", "", ""))
        self.clip.generate.assert_not_called()
        with self.assertRaises(ValueError):
            GemmaPromptExpandNative().run(**self.args, output_format="bad")
        self.clip.decode.return_value = "Create a scene."
        self.clip.generate.side_effect = [[], RuntimeError("interrupted")]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            GemmaPromptExpandNative().run(**self.args)
        self.assertEqual(self.clip.generate.call_count, 2)

    def test_prose_never_uses_comma_filter_and_legacy_contract_stays_separate(self):
        self.assertEqual(clean_natural_language("<think>x</think>\nCreate a scene, with a red coat."),
                         "Create a scene, with a red coat.")
        with self.assertRaises(RuntimeError):
            clean_natural_language("<think>x</think>")
        from tipo_prompt import VERSION
        self.assertEqual(VERSION, "gemma4-tipo-native/1")
        self.assertIn("e621", build_expansion_prompt("fox", "short", "", False, "tags", tag_set="e621"))


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fictional.jsonl"
        self.rows = [self.row("danbooru", "blue_eyes", "general", {"ko": ["벽안", "파란 눈"], "ja": ["青い目"]}),
                     self.row("e621", "blue_eyes", "general", {"ko": ["푸른 눈"]}),
                     self.row("e621", "example_species", "species", {"ko": ["예시 종"]}),
                     self.row("danbooru", "example_artist", "artist", {"ko": ["벽안"]})]
        self.save()

    @staticmethod
    def row(source, tag, category, terms):
        return {"schema": SCHEMA, "source": source, "tag": tag, "category": category,
                "terms": terms, "definitions": {}, "status": "unreviewed",
                "provenance": {"test": "fictional data"}}

    def save(self):
        self.path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in self.rows), encoding="utf-8")

    def test_source_boundaries_language_terms_and_no_positive_role_inference(self):
        knowledge = TagKnowledge(self.path)
        self.assertEqual(knowledge.lookup("blue eyes")["terms"]["ko"], ["벽안", "파란 눈"])
        self.assertEqual(knowledge.lookup("blue eyes", "e621")["terms"]["ko"], ["푸른 눈"])
        self.assertIsNone(knowledge.lookup("example species"))
        packet = knowledge.vocabulary("青い目 대신 벽안은 빼줘. eyeblue_eyesx")
        self.assertEqual([r["term"] for r in packet["mentions"]], ["青い目", "벽안"])
        self.assertTrue(all(r["role"] == "unresolved" for r in packet["mentions"]))
        self.assertEqual(packet["mentions"][1]["meanings"][0]["tag"], "blue_eyes")
        self.assertEqual(len(packet["mentions"][1]["meanings"]), 1)
        self.assertEqual(knowledge.check_tags("blue_eyes, invented tag")["unverified"], ["invented tag"])
        self.assertEqual(knowledge.provenance["reviewed_rows"], 0)

    def test_duplicate_invalid_category_and_missing_provenance_rejected(self):
        for extra in (self.rows[0], self.row("danbooru", "x", "species", {}),
                      dict(self.row("e621", "x", "general", {}), provenance={})):
            with self.subTest(extra=extra):
                self.rows.append(extra)
                self.save()
                with self.assertRaises(ValueError):
                    TagKnowledge(self.path)
                self.rows.pop()

    def test_ordinary_sentence_fragments_do_not_become_names(self):
        self.rows += [self.row("danbooru", "a", "artist", {"ko": ["이"]}),
                      self.row("danbooru", "night market", "general", {"ko": ["야시장"]}),
                      self.row("danbooru", "some_artist", "artist", {"ko": ["야시"]})]
        self.save()
        knowledge = TagKnowledge(self.path)
        packet = knowledge.vocabulary("a woman 야시장으로 파란 눈은 유지해줘. 미지의 장면")
        self.assertEqual([r["term"] for r in packet["mentions"]], ["야시장", "파란 눈"])
        self.assertIsNotNone(knowledge.lookup("a"))

    def test_generated_scene_does_not_discover_artists_from_descriptive_words(self):
        self.rows.append(dict(self.row("danbooru", "fictional_artist", "artist", {"en": ["illustration"]}),
                              name_aliases={"ko": ["가상작가"]}))
        self.save()
        knowledge = TagKnowledge(self.path)
        self.assertTrue(knowledge.vocabulary("가상작가")["mentions"])
        self.assertTrue(knowledge.vocabulary("fictional_artist")["mentions"])
        self.assertEqual(knowledge.vocabulary("illustration")["mentions"], [])
        self.assertEqual(knowledge.vocabulary("가상작가", include_identities=False)["mentions"], [])
        self.assertEqual(knowledge.vocabulary("illustration", include_identities=False)["mentions"], [])

    def test_legacy_csv_stays_danbooru_and_cache_refreshes(self):
        path = self.path.with_suffix(".csv")
        with path.open("w", newline="", encoding="utf-8") as out:
            csv.writer(out).writerow(["example_meta", 5, 2, "설명. 키워드: 예시"])
        first = load_knowledge(str(path))
        self.assertEqual(first.lookup("example_meta")["category"], "meta")
        self.assertIsNone(first.lookup("example_meta", "e621"))
        path.write_text("other_tag,0,3,다른 설명\n", encoding="utf-8")
        second = load_knowledge(str(path))
        self.assertIsNone(second.lookup("example_meta"))
        self.assertNotEqual(first.provenance["sha256"], second.provenance["sha256"])

    def test_node_delivers_vocabulary_and_reports_unverified_tags_without_dropping(self):
        clip = Mock()
        clip.tokenize.return_value = {"gemma4": []}
        clip.decode.return_value = "blue_eyes, invented tag"
        args = dict(clip=clip, prompt="벽안", tag_length="short", max_tokens=128, temperature=0,
                    ban_tags="", seed=0, output_format="tags", tag_knowledge_path=str(self.path))
        tags, _, raw = GemmaPromptExpandNative().run(**args)
        self.assertIn("blue_eyes", clip.tokenize.call_args.args[0])
        self.assertIn("invented tag", tags)
        self.assertEqual(json.loads(raw)["tag_membership"]["unverified"], ["invented tag"])
        self.rows = [self.rows[0]]
        self.save()
        clip.generate.reset_mock()
        with self.assertRaisesRegex(ValueError, "no entries"):
            GemmaPromptExpandNative().run(**args, tag_set="e621")
        clip.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
