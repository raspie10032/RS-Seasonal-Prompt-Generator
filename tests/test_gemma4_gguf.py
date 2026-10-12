"""Weight-copy safety and original-config compatibility for local GGUF exports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.convert_gemma4_gguf import original_config


class ConfigTests(unittest.TestCase):
    def test_original_global_head_dim_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            config = {'model_type': 'gemma4', 'text_config': {'global_head_dim': 512}}
            path.write_text(json.dumps(config))
            before = path.read_bytes()
            self.assertEqual(original_config(temp), config)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                original_config(temp, True)
            path.write_text('{"model_type":"other"}')
            with self.assertRaises(ValueError):
                original_config(temp)


@unittest.skipUnless(importlib.util.find_spec('gguf'), 'Requires llama.cpp gguf-py')
class VisionCopyTests(unittest.TestCase):
    def write_fixture(self, path, extra=False):
        import gguf
        import numpy as np
        writer = gguf.GGUFWriter(path, 'clip')
        writer.add_clip_has_vision_encoder(True)
        writer.add_clip_has_audio_encoder(True)
        writer.add_clip_vision_projector_type(gguf.VisionProjectorType.GEMMA4V)
        writer.add_clip_audio_projector_type(gguf.VisionProjectorType.GEMMA4A)
        writer.add_vision_image_mean([0.0, 0.0, 0.0])
        for name in ['v.test.weight', 'mm.input_projection.weight', 'a.test.weight']:
            writer.add_tensor(name, np.arange(64, dtype=np.float16).reshape(8, 8))
        if extra:
            writer.add_tensor('unknown.weight', np.ones((2, 2), dtype=np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_audio_removed_vision_bytes_and_metadata_preserved(self):
        import gguf
        from experiments.gemma4_gguf_audit import sha256, vision_only
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'source.gguf', Path(temp) / 'vision.gguf'
            self.write_fixture(source)
            before = sha256(source)
            result = vision_only(source, output)
            self.assertTrue(result['all_kept_tensor_payloads_identical'])
            self.assertEqual(result['kept_tensors'], 2)
            self.assertEqual(result['removed_audio_tensors'], 1)
            self.assertEqual(sha256(source), before)
            reader = gguf.GGUFReader(output)
            self.assertFalse(reader.get_field('clip.has_audio_encoder').contents())
            self.assertIsNone(reader.get_field('clip.audio.projector_type'))
            self.assertEqual(reader.get_field('clip.vision.image_mean').contents(), [0., 0., 0.])
            with self.assertRaisesRegex(ValueError, 'must not exist'):
                vision_only(source, output)

    def test_unknown_weights_are_rejected_before_writing(self):
        from experiments.gemma4_gguf_audit import vision_only
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'source.gguf', Path(temp) / 'vision.gguf'
            self.write_fixture(source, extra=True)
            with self.assertRaisesRegex(ValueError, 'Unrecognized tensor'):
                vision_only(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + '.partial').exists())

    def test_unchanged_ple_comparison_is_rejected(self):
        from experiments.gemma4_gguf_audit import compare_ple
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.gguf'
            self.write_fixture(source)
            with self.assertRaisesRegex(ValueError, 'missing PLE'):
                compare_ple(source, source)


if __name__ == '__main__':
    unittest.main()
