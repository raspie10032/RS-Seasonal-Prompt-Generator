"""Native memory lifecycle and image binding; no model downloads required."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
try:
    import torch
except ImportError:
    torch = None

from vrs_int.native import build_native, load_image, pixel_digest
from vrs_int.import_snapshot import file_digest
from vrs_int.store import Store
from vrs_native_nodes import VRSNativeMemoryLoader, VRSNativeRecall
from gemma_prompt_expand_node import GemmaPromptExpandNative


@unittest.skipIf(torch is None, "ComfyUI torch needed for IMAGE tensor tests")
class NativeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        Image.fromarray(np.full((8, 9, 3), 80, dtype=np.uint8)).save(self.root / "one.png")
        self.sha = file_digest(self.root / "one.png")
        self.manifest = self.root / "pairs.jsonl"
        self.row = {"source": "generated", "id": self.sha, "image_sha256": self.sha,
                    "image": "one.png", "rating": "u", "tags_general": ["1girl", "blue_hair"],
                    "tags_character": ["example_character"], "tags_artist": ["example_artist"]}
        self.manifest.write_text(json.dumps(self.row) + "\n")
        self.contract = {"encoder": {"provenance_sha256": "a" * 64, "preprocess": "test-native",
            "backend": "comfyui-pytorch-fp32-compute", "quantization": "fp16", "pooling": "mean-tokens-l2-v1"},
            "dimension": 768, "codec": "f32le", "tag_policy": "rs-core-identities-v1"}
        self.encoder = Mock()
        self.encoder.contract.side_effect = lambda codec="f32le": dict(self.contract, codec=codec)
        self.encoder.encode.side_effect = lambda path: np.ones(768, dtype=np.float32) / np.sqrt(768)
        self.encoder.last_pixel_sha256 = pixel_digest(load_image(self.root / "one.png"))
        self.mm = SimpleNamespace(throw_exception_if_processing_interrupted=Mock())
        self.modules = patch.dict(sys.modules, {"comfy": SimpleNamespace(model_management=self.mm),
                                               "comfy.model_management": self.mm})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def build(self, **kwargs):
        return build_native(self.encoder, self.manifest, self.root, self.root / "memory", **kwargs)

    def test_build_update_retains_source_and_old_snapshot(self):
        first = self.build()
        self.row["tags_general"].append("holding_cup")
        self.manifest.write_text(json.dumps(self.row) + "\n")
        with patch.object(Store, "get", side_effect=AssertionError("Avoid per-image bucket lookup")):
            second = self.build(base=first["snapshot"])
        self.assertNotEqual(first["snapshot"], second["snapshot"])
        memory, _ = VRSNativeMemoryLoader().load(str(self.root / "memory"), second["snapshot"])
        old = next(memory.store.entries(first["snapshot"]))
        new = next(memory.store.entries(second["snapshot"]))
        self.assertNotIn("holding cup", old["document"]["tags"])
        self.assertIn("holding cup", new["document"]["tags"])
        self.assertIn("example artist", new["document"]["tags"])
        self.assertEqual(new["audit"][-1]["native_pixel_sha256"], self.encoder.last_pixel_sha256)
        self.assertEqual(file_digest(self.root / "one.png"), self.sha)
        self.assertEqual(memory.store.verify(first["snapshot"])["observations"], 1)

    def test_cancel_never_publishes_and_existing_store_requires_base(self):
        self.mm.throw_exception_if_processing_interrupted.side_effect = RuntimeError("cancel")
        with self.assertRaisesRegex(RuntimeError, "cancel"):
            self.build()
        self.assertFalse(list((self.root / "memory" / "snapshots").glob("*.json")))
        self.mm.throw_exception_if_processing_interrupted.side_effect = None
        self.build()
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.build()

    def test_contract_and_control_mismatch_prevent_encoding(self):
        first = self.build()
        memory, _ = VRSNativeMemoryLoader().load(str(self.root / "memory"), first["snapshot"])
        image = load_image(self.root / "one.png")
        memory.manifest["contract"]["encoder"]["backend"] = "pil"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            VRSNativeRecall().recall(self.encoder, memory, image, "새 장면")
        self.encoder.encode_tensor.assert_not_called()
        memory.manifest["contract"] = deepcopy(self.contract)
        with self.assertRaisesRegex(ValueError, "excluded"):
            VRSNativeRecall().recall(self.encoder, memory, image, "새 장면", keep_tags="coat", exclude_tags="coat")
        self.encoder.encode_tensor.assert_not_called()

    def test_pixel_digest_reencoded_copies_and_invalid_batch(self):
        image = load_image(self.root / "one.png")
        Image.open(self.root / "one.png").save(self.root / "copy.png", compress_level=0)
        self.assertNotEqual(file_digest(self.root / "one.png"), file_digest(self.root / "copy.png"))
        self.assertEqual(pixel_digest(image), pixel_digest(load_image(self.root / "copy.png")))
        self.assertNotEqual(pixel_digest(image), pixel_digest(image + .1))
        with self.assertRaises(ValueError):
            pixel_digest(image.repeat(2, 1, 1, 1))
        with self.assertRaises(ValueError):
            pixel_digest(image * float("nan"))

    def test_recall_forwards_exact_request_and_image_after_existing_outputs(self):
        image = load_image(self.root / "one.png")
        memory = Mock(snapshot="a" * 64, manifest={"contract": deepcopy(self.contract)})
        memory.recommend.return_value = {"snapshot": memory.snapshot, "candidates": []}
        self.encoder.encoder_contract = self.contract["encoder"]
        self.encoder.weights_bytes = 123
        prompt = "  인물은 유지하고 장면을 바꿔줘  "
        result = VRSNativeRecall().recall(self.encoder, memory, image, prompt)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["request"], prompt.strip())
        self.assertEqual(json.loads(result[2])["state"], result[0])
        self.assertEqual(result[3], prompt)
        self.assertIs(result[4], image)
        self.assertEqual(VRSNativeRecall.RETURN_NAMES[:3], ("vrs_state", "candidate_tags", "report"))

    def test_connected_state_matches_image_and_request_before_gemma(self):
        image = load_image(self.root / "one.png")
        from vrs_int.generation_state import build_state
        state = build_state({"snapshot": "a" * 64}, "새 장면")
        state["native_query"] = {"pixel_sha256": pixel_digest(image)}
        args = dict(clip=Mock(), prompt="새 장면", tag_length="short", max_tokens=128,
                    temperature=0, ban_tags="", seed=0, vrs_state=state, image=image)
        with self.assertRaisesRegex(ValueError, "IMAGE differ"):
            GemmaPromptExpandNative().run(**dict(args, image=image + .1))
        with self.assertRaisesRegex(ValueError, "either"):
            GemmaPromptExpandNative().run(**args, vrs_state_path="example.json")
        with patch("gemma_prompt_expand_node.generate_text", side_effect=["A blue-haired woman outdoors.", "1girl, blue hair, outdoors"]):
            tags, natural, raw = GemmaPromptExpandNative().run(**args)
        self.assertIn("1girl", tags)
        self.assertIn("blue-haired", natural)
        self.assertIn("verified", json.loads(raw)["vrs_reference"]["query_image_binding"])
        args["clip"].generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
