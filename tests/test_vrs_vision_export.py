"""The public extractor must never publish nonvision data or replace originals."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_vrs_vision import expected_shapes, prepare


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"), "Requires PyTorch and safetensors")
class VisionExportTests(unittest.TestCase):
    def test_export_contains_only_fp16_vision_and_preserves_input(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        key = "vision_model.encoder.layers.0.input_layernorm.weight"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "native.safetensors", root / "vision.safetensors"
            value = torch.tensor([1, 0.5], dtype=torch.bfloat16)
            save_file({key: value, "model.private_not_for_distribution": torch.tensor([123])}, str(source))
            before = source.read_bytes()
            with patch("scripts.prepare_vrs_vision.expected_shapes", return_value={key: (2,)}):
                report = prepare(source, output)
            self.assertEqual(source.read_bytes(), before)
            with safe_open(str(output), framework="pt", device="cpu") as f:
                self.assertEqual(f.keys(), [key])
                self.assertTrue(torch.equal(f.get_tensor(key), value.half()))
                self.assertEqual(f.get_tensor(key).dtype, torch.float16)
                self.assertNotIn(temp, str(f.metadata()))
            self.assertEqual(report["tensor_count"], 1)
            with self.assertRaises(FileExistsError):
                prepare(source, output)
            self.assertEqual(source.read_bytes(), before)

    def test_incomplete_or_nonfinite_input_is_not_published(self):
        import torch
        from safetensors.torch import save_file
        key = "vision_model.x"
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source.safetensors", Path(temp) / "output.safetensors"
            save_file({key: torch.tensor([1.0])}, str(source))
            with self.assertRaisesRegex(ValueError, "complete native"):
                prepare(source, output)
            save_file({key: torch.tensor([float("inf")])}, str(source))
            with patch("scripts.prepare_vrs_vision.expected_shapes", return_value={key: (1,)}):
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    prepare(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temp).glob("*.part")), [])

    def test_schema_is_complete_e2b_tower_without_projector_or_language(self):
        schema = expected_shapes()
        self.assertEqual(len(schema), 658)
        self.assertEqual(sum(__import__("math").prod(s) for s in schema.values()) * 2, 334730112)
        self.assertTrue(all(k.startswith("vision_model.") for k in schema))


if __name__ == "__main__":
    unittest.main()
