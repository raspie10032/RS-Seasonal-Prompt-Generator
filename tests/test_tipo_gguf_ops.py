"""Small numerical checks; run with ComfyUI on PYTHONPATH, no model download."""
import importlib.util
import unittest
from unittest.mock import patch


RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "gguf", "comfy"))


@unittest.skipUnless(RUNTIME, "Requires ComfyUI's torch and gguf; no model weights")
class PackedMathTests(unittest.TestCase):
    def setUp(self):
        import gguf
        import numpy as np
        import torch
        from tipo_gguf_ops import PackedWeight

        self.torch, self.gguf = torch, gguf
        source = np.sin(np.arange(7 * 64, dtype=np.float32)).reshape(7, 64)
        data = gguf.quants.quantize(source, gguf.GGMLQuantizationType.Q4_0)
        self.reference = torch.from_numpy(gguf.quants.dequantize(data, gguf.GGMLQuantizationType.Q4_0))
        self.weight = PackedWeight(torch.from_numpy(data), gguf.GGMLQuantizationType.Q4_0, source.shape)

    def test_copy_preserves_packed_bytes_and_logical_shape(self):
        torch = self.torch
        for result in (self.weight.to(dtype=torch.float16), self.weight.clone()):
            self.assertEqual(tuple(result.shape), (7, 64))
            self.assertEqual(result.dtype, torch.uint8)
            self.assertTrue(torch.equal(result.as_subclass(torch.Tensor), self.weight.as_subclass(torch.Tensor)))

    def test_gathered_rows_match_independent_decoder_without_numpy_fallback(self):
        from tipo_gguf_ops import unpack
        torch = self.torch
        rows = torch.tensor([[6, 0, 6], [3, 2, 0]])
        with patch.object(self.gguf.quants, "dequantize", side_effect=AssertionError("No NumPy inference")):
            actual = unpack(self.weight, "cpu", torch.float32, rows=rows)
        torch.testing.assert_close(actual, self.reference[rows], rtol=0, atol=0)

    def test_chunked_output_projection_matches_full_reference(self):
        from tipo_gguf_ops import GGUFOps
        torch = self.torch
        embedding = GGUFOps.Embedding(7, 64)
        embedding.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        inputs = torch.arange(128, dtype=torch.float32).reshape(1, 2, 64) / 100
        actual = embedding.project(inputs, chunk=3)
        expected = torch.nn.functional.linear(inputs, self.reference)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_unknown_quantization_fails_instead_of_cpu_fallback(self):
        from tipo_gguf_ops import PackedWeight, unpack
        weight = PackedWeight(self.torch.zeros(16, dtype=self.torch.uint8), -1, (1, 32))
        with self.assertRaisesRegex(ValueError, "CPU fallback is disabled"):
            unpack(weight, "cpu", self.torch.float32)
