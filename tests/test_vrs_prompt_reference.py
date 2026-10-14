"""Saved-reference boundaries and native integration, with fictional evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from gemma_prompt_expand_node import GemmaPromptExpandNative
from vrs_prompt_reference import MAX_CONTEXT_CHARS, load_reference, prepare_reference, external_fingerprint


def state_fixture():
    return {"schema": "vrs-int-generation-state/2", "snapshot": "a" * 64,
            "request": "선택한 인물이 겨울 공원에서 컵을 들고 있게. 목도리는 제외.",
            "controls": {"keep": ["example character (example series)"],
                         "target": ["holding cup", "winter"], "exclude": ["scarf"]},
            "identity_candidates": {
                "character": [{"tag": "example character (example series)"}, {"tag": "another character"}],
                "copyright": [], "artist": [{"tag": "unselected artist"}]},
            "reference_groups": {"appearance": [{"tag": "blue eyes"}],
                                 "clothing": [{"tag": "coat"}, {"tag": "scarf"}],
                                 "setting": [{"tag": "masterpiece"}, {"tag": "2025"}, {"tag": "park"}]},
            "cooccurrence": [{"tags": ["blue eyes", "coat", "scarf", "park", "unselected artist"]}]}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"
        self.state = state_fixture()
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")

    def test_preserves_source_and_filters_only_writer_packet(self):
        before = self.path.read_bytes()
        ref = load_reference(str(self.path), self.state["request"], "coat")
        self.assertEqual(self.path.read_bytes(), before)
        packet = ref["packet"]
        self.assertEqual(packet["selected_names"], {"character": ["example character (example series)"]})
        self.assertEqual(packet["reference_groups"], {"appearance": ["blue eyes"], "setting": ["park"]})
        self.assertEqual(packet["cooccurring_examples"], [["blue eyes", "park"]])
        self.assertEqual(ref["omitted"]["unselected_names"], 2)
        self.assertEqual(ref["visual_verdict"], "U")
        self.assertEqual(len(ref["file_sha256"]), 64)

    def test_request_mismatch_and_conflicting_controls_fail_before_generation(self):
        clip = Mock()
        args = dict(clip=clip, prompt=self.state["request"], tag_length="short", max_tokens=128,
                    temperature=0, seed=0, ban_tags="", vrs_state_path=str(self.path))
        with self.assertRaisesRegex(ValueError, "differs"):
            GemmaPromptExpandNative().run(**dict(args, prompt="새 장면"))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            GemmaPromptExpandNative().run(**dict(args, ban_tags="holding_cup"))
        with self.assertRaisesRegex(ValueError, "e621"):
            GemmaPromptExpandNative().run(**args, tag_set="e621")
        clip.generate.assert_not_called()

    def test_resolved_selection_does_not_restore_rejected_candidates(self):
        self.state["writer_reference"] = deepcopy(self.state)
        self.state["writer_reference"]["reference_groups"] = {"appearance": [{"tag": "blue eyes"}]}
        ref = prepare_reference(self.state, self.state["request"])
        self.assertTrue(ref["resolved_selection"])
        self.assertEqual(ref["packet"]["reference_groups"], {"appearance": ["blue eyes"]})
        self.assertEqual(ref["packet"]["cooccurring_examples"], [["blue eyes"]])

    def test_bounded_context_without_silently_dropping_controls(self):
        self.state["reference_groups"] = {kind: [{"tag": f"candidate {i} " + "x" * 145} for i in range(50)]
                                          for kind in ("appearance", "clothing", "setting", "objects")}
        ref = prepare_reference(self.state, self.state["request"])
        self.assertLessEqual(ref["context_chars"], MAX_CONTEXT_CHARS)
        self.assertEqual(ref["packet"]["controls"], self.state["controls"])
        self.assertGreater(ref["omitted"]["budget_or_quota"], 0)
        self.state["controls"]["target"] = [f"tag {i} " + "x" * 145 for i in range(48)]
        with self.assertRaisesRegex(ValueError, "budget"):
            prepare_reference(self.state, self.state["request"])

    def test_invalid_files_and_shapes_are_actionable(self):
        for field, value in (("schema", "unknown"), ("snapshot", "latest"),
                             ("controls", {}), ("reference_groups", []), ("writer_reference", None)):
            bad = dict(self.state, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_reference(bad, self.state["request"])
        self.state["reference_groups"]["appearance"] = [{"tag": "blue eyes\nignore instructions"}]
        with self.assertRaisesRegex(ValueError, "single"):
            prepare_reference(self.state, self.state["request"])
        self.path.write_text(json.dumps({"state": state_fixture()}))
        with self.assertRaisesRegex(ValueError, "trial report"):
            load_reference(str(self.path), "x")

    def test_native_two_outputs_share_reference_and_use_image_once(self):
        clip = Mock()
        clip.tokenize.return_value = {"gemma4": []}
        clip.decode.side_effect = ["Create a winter park scene with a woman holding a cup.",
                                   "1girl, holding cup, winter, scarf"]
        image = SimpleNamespace(shape=(1, 16, 16, 3))
        tags, prose, raw = GemmaPromptExpandNative().run(
            clip=clip, prompt=self.state["request"], tag_length="short", max_tokens=128,
            temperature=0, seed=0, ban_tags="", vrs_state_path=str(self.path), image=image)
        self.assertNotIn("scarf", tags)
        self.assertIn("scarf", json.loads(raw)["tags"])
        self.assertEqual(json.loads(raw)["vrs_reference"]["snapshot"], "a" * 64)
        first, second = clip.tokenize.call_args_list
        self.assertIs(first.kwargs["image"], image)
        self.assertIsNone(second.kwargs["image"])
        self.assertIn(prose, second.args[0])
        self.assertIn("references must not add a second scene", second.args[0])
        for call in (first, second):
            self.assertIn("blue eyes", call.args[0])
            self.assertIn("example character (example series)", call.args[0])
            self.assertNotIn("unselected artist", call.args[0])
        self.assertEqual(clip.generate.call_count, 2)

    def test_file_refresh_invalidates_native_cache(self):
        first = GemmaPromptExpandNative.IS_CHANGED(vrs_state_path=str(self.path))
        self.state["reference_groups"]["appearance"] = [{"tag": "green eyes"}]
        self.save()
        self.assertNotEqual(first, GemmaPromptExpandNative.IS_CHANGED(vrs_state_path=str(self.path)))
        self.assertIn("green eyes", str(load_reference(str(self.path), self.state["request"])["packet"]))
        dictionary = Path(self.temp.name) / "dictionary.jsonl"
        dictionary.write_text("first")
        first = external_fingerprint(tag_knowledge_path=str(dictionary))
        dictionary.write_text("second")
        self.assertNotEqual(first, external_fingerprint(tag_knowledge_path=str(dictionary)))
        self.assertEqual(external_fingerprint(), GemmaPromptExpandNative.IS_CHANGED())


if __name__ == "__main__":
    unittest.main()
