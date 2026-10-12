"""Creative-reference boundaries and model ownership, without inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.image_prompt import make_brief, prompt_messages
from experiments.vrs_image import corrected_index_metadata
import prompt_writer_node as writer
import tipo_engine as te


class CreativeReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rows = {
            1: {"id": 1, "tags": ["alice", "alice world", "artist a", "white dress", "solo"]},
            2: {"id": 2, "tags": ["beth", "beth world", "artist b", "black dress", "2girls"]},
        }
        self.categories = {"alice": ["character"], "beth": ["character"],
                           "alice world": ["copyright"], "beth world": ["copyright"],
                           "artist a": ["artist"], "artist b": ["artist"]}
        self.recommendation = {
            "neighbors": [{"id": 1}, {"id": 2}],
            "candidates": [{"tag": t} for t in ("white dress", "alice", "beth world", "solo")],
            "identities": {"character": [{"tag": "alice"}],
                           "copyright": [{"tag": "beth world"}, {"tag": "alice world"}],
                           "artist": [{"tag": "artist b"}, {"tag": "artist a"}]},
        }

    def test_names_use_one_observation_not_independent_top_candidates(self):
        brief = make_brief(self.recommendation, self.rows, self.categories)
        self.assertEqual(brief["borrowed_names"], {
            "artist": ["artist a"], "character": ["alice"], "copyright": ["alice world"]})
        self.assertNotIn("beth world", brief["visual_palette"])
        self.assertEqual(brief["name_evidence_image_id"], 1)
        self.assertEqual(len(self.recommendation["identities"]["copyright"]), 2)

    def test_original_mode_removes_names_from_writer_material_not_memory(self):
        brief = make_brief(self.recommendation, self.rows, self.categories, identity_mode="original")
        self.assertTrue(all(not tags for tags in brief["borrowed_names"].values()))
        self.assertNotIn("alice", brief["visual_palette"])
        self.assertIn("alice", self.rows[1]["tags"])

    def test_creative_direction_is_separate_from_reference_data(self):
        direction = "겨울 축제, 우산 없이"
        brief = make_brief(self.recommendation, self.rows, self.categories, direction)
        messages = prompt_messages(brief, "natural_language")
        self.assertEqual(json.loads(messages[1]["content"])["direction"], direction)
        self.assertNotIn("new_scene_hint", json.loads(messages[1]["content"]))
        self.assertIn("takes precedence", messages[0]["content"])
        self.assertIn("NOT mandatory facts", messages[0]["content"])
        self.assertIn("complete sentences", messages[0]["content"])

    def test_automatic_scene_shared_across_formats_and_changes_with_seed(self):
        brief = make_brief(self.recommendation, self.rows, self.categories)
        tag = json.loads(prompt_messages(brief, "tags", seed=42)[1]["content"])
        prose = json.loads(prompt_messages(brief, "natural_language", seed=42)[1]["content"])
        other = json.loads(prompt_messages(brief, seed=43)[1]["content"])
        self.assertEqual(tag["new_scene_hint"], prose["new_scene_hint"])
        self.assertNotEqual(tag["new_scene_hint"], other["new_scene_hint"])
        self.assertNotIn("new_scene_hint", brief)

    def test_visual_bundle_excludes_other_neighbors_conflicting_clothing(self):
        self.recommendation["candidates"].append({"tag": "black dress"})
        brief = make_brief(self.recommendation, self.rows, self.categories)
        self.assertIn("white dress", brief["visual_palette"])
        self.assertNotIn("black dress", brief["visual_palette"])

    def test_no_neighbors_fails_instead_of_generating_unrelated_prompt(self):
        with self.assertRaisesRegex(ValueError, "No reference"):
            make_brief({"neighbors": []}, {}, {})


class WriterOwnershipTests(unittest.TestCase):
    def generate(self, llm):
        with patch.object(te, "_ensure_deps", return_value=True), \
                patch.object(te, "require_gpu_offload"), \
                patch.object(writer, "_resolve_model", return_value="/model.gguf"), \
                patch.object(te, "load_tipo", return_value=llm) as load:
            result = writer.write_messages([{"role": "user", "content": "reference"}])
        self.assertFalse(load.call_args.kwargs["cache"])
        self.assertEqual(load.call_args.args[1], -1)
        return result

    def test_private_model_closed_on_success(self):
        llm = Mock()
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "white dress, winter"}, "finish_reason": "stop"}]}
        self.assertEqual(self.generate(llm), "white dress, winter")
        llm.close.assert_called_once()

    def test_private_model_closed_on_inference_failure(self):
        llm = Mock()
        llm.create_chat_completion.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.generate(llm)
        llm.close.assert_called_once()

    def test_gpu_failure_is_not_retried_on_cpu(self):
        with patch.object(te, "_ensure_deps", return_value=True), \
                patch.object(te, "require_gpu_offload", side_effect=RuntimeError("no GPU")), \
                patch.object(te, "load_tipo") as load:
            with self.assertRaisesRegex(RuntimeError, "no GPU"):
                writer.write_messages([{"role": "user", "content": "reference"}])
        load.assert_not_called()

    def test_uncached_load_never_reuses_or_replaces_shared_model(self):
        shared, private = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.write_bytes(b"test")
            key = (str(path.resolve()), -1, 4096, 2)
            fake_module = Mock(Llama=Mock(return_value=private))
            with patch.dict("sys.modules", {"llama_cpp": fake_module}), \
                    patch.object(te, "_model_cache", {key: shared}):
                self.assertIs(te.load_tipo(str(path), -1, 4096, 2, cache=False), private)
                self.assertIs(te._model_cache[key], shared)


class IndexProvenanceTests(unittest.TestCase):
    def test_mismatched_encoder_and_modified_vectors_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "receipt.json").write_text("receipt")
            (p / "provenance.json").write_text("encoder")
            (p / "train_selection.json").write_text('[{"id": 1, "split": "train"}]')
            (p / "corrected_vectors.npy").write_bytes(b"vectors")
            digest = lambda name: hashlib.sha256((p / name).read_bytes()).hexdigest()
            report = {"state": "complete", "index_count": 1,
                      "memory_receipt_sha256": digest("receipt.json"),
                      "encoder_provenance_sha256": digest("provenance.json"),
                      "output_sha256": {n: digest(n) for n in ("train_selection.json", "corrected_vectors.npy")}}
            (p / "report.json").write_text(json.dumps(report))
            self.assertEqual(corrected_index_metadata(p, p / "receipt.json", p / "provenance.json")[1][0]["id"], 1)
            (p / "provenance.json").write_text("different encoder")
            with self.assertRaisesRegex(ValueError, "encoder mismatch"):
                corrected_index_metadata(p, p / "receipt.json", p / "provenance.json")
            (p / "corrected_vectors.npy").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                corrected_index_metadata(p, p / "receipt.json")


if __name__ == "__main__":
    unittest.main()
