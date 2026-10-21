"""Vision setup remains local, lazy, cancellable and independent of the LM."""
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vrs_int.vision_assets import AUTO_VISION, VISION, cached_tipo_vision, has_native_vision, local_vision_path, resolve_vision_path, vision_choices


class VisionSetupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        probe = patch('vrs_int.torch_device._probe')
        probe.start()
        self.addCleanup(probe.stop)

    def header(self, name, keys):
        path = self.root / name
        header = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
        return path

    def test_choices_read_headers_and_exclude_unrelated_or_legacy_models(self):
        files = {"native.safetensors": self.header("native.safetensors", ["vision_model.encoder.layers.0.weight"]),
                 "legacy.safetensors": self.header("legacy.safetensors", ["encoder.layers.0.weight"]),
                 "qwen.safetensors": self.header("qwen.safetensors", ["model.layers.0.weight"])}
        folder = SimpleNamespace(get_filename_list=lambda _: list(files),
            get_full_path=lambda _, name: files[name], get_full_path_or_raise=lambda _, name: files[name])
        with patch.dict(sys.modules, {"folder_paths": folder}):
            self.assertEqual(vision_choices(), [AUTO_VISION, "native.safetensors"])
            self.assertEqual(local_vision_path("native.safetensors"), files["native.safetensors"])
            with self.assertRaisesRegex(ValueError, "auto"):
                local_vision_path("legacy.safetensors")

    def test_truncated_large_or_nonobject_headers_are_not_models(self):
        for content in (b"bad", struct.pack("<Q", 2**63), struct.pack("<Q", 2) + b"[]"):
            path = self.root / "bad.safetensors"
            path.write_bytes(content)
            self.assertFalse(has_native_vision(path))

    def runtime(self):
        mm = SimpleNamespace(get_torch_device=lambda: SimpleNamespace(type="cuda"),
                             throw_exception_if_processing_interrupted=Mock())
        progress = Mock()
        utils = SimpleNamespace(ProgressBar=Mock(return_value=progress))
        modules = {"folder_paths": SimpleNamespace(models_dir=str(self.root)),
            "comfy": SimpleNamespace(model_management=mm, utils=utils),
            "comfy.model_management": mm, "comfy.utils": utils,
            "comfy.text_encoders": SimpleNamespace(gemma4=SimpleNamespace(Gemma4VisionEncoder=object())),
            "gguf": None}  # Standalone vision must work without the parser.
        return mm, progress, modules

    def test_auto_prepares_standalone_vision_without_gguf_or_language_model(self):
        mm, progress, modules = self.runtime()
        interrupt = Mock()
        def prepare(entry, directory, callback):
            self.assertIs(entry, VISION)
            self.assertEqual(directory, self.root / "text_encoders" / "rs_vrs_vision")
            callback(1, 2)
            return directory
        with patch.dict(sys.modules, modules), patch("tipo_native_lora.ensure_files", side_effect=prepare) as fetch:
            path = resolve_vision_path(AUTO_VISION, interrupt)
        self.assertEqual(path.name, "vrs-gemma4-e2b-vision-fp16.safetensors")
        fetch.assert_called_once()
        self.assertEqual(mm.throw_exception_if_processing_interrupted.call_count, 2)
        interrupt.assert_called_once()
        progress.update_absolute.assert_called_once_with(1, 2)

    def cached_mmproj(self):
        path = self.root / "text_encoders" / "rs_gemma_q4" / "mmproj-F16.gguf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"existing mmproj")
        return path

    def test_verified_cached_mmproj_reused_without_network(self):
        path = self.cached_mmproj()
        _, progress, modules = self.runtime()
        modules["gguf"] = SimpleNamespace()
        with patch.dict(sys.modules, modules), patch("tipo_native_lora._check_file") as verify, \
                patch("tipo_native_lora.ensure_files", side_effect=AssertionError("No download")):
            self.assertEqual(resolve_vision_path(AUTO_VISION), path)
        verify.assert_called_once()
        self.assertEqual(progress.update_absolute.call_count, 2)

    def test_missing_parser_or_invalid_mmproj_does_not_block_standalone(self):
        self.cached_mmproj()
        with patch.dict(sys.modules, {"gguf": None}), patch("tipo_native_lora._check_file") as verify:
            self.assertIsNone(cached_tipo_vision(self.root, Mock()))
            verify.assert_not_called()
        with patch.dict(sys.modules, {"gguf": SimpleNamespace()}), \
                patch("tipo_native_lora._check_file", side_effect=ValueError("Corrupt cache")):
            self.assertIsNone(cached_tipo_vision(self.root, Mock()))

    def test_bad_standalone_not_silently_replaced_by_cached_mmproj(self):
        self.cached_mmproj()
        destination = self.root / "text_encoders" / "rs_vrs_vision" / next(iter(VISION["files"]))
        destination.parent.mkdir()
        destination.write_bytes(b"corrupt")
        _, _, modules = self.runtime()
        with patch.dict(sys.modules, modules), patch("vrs_int.vision_assets.cached_tipo_vision") as cached, \
                patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            with self.assertRaisesRegex(ValueError, "wrong size/type"):
                resolve_vision_path(AUTO_VISION)
            cached.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"corrupt")

    def test_cpu_rejected_before_download_or_parser_import(self):
        mm = SimpleNamespace(get_torch_device=lambda: SimpleNamespace(type="cpu"))
        modules = {"comfy": SimpleNamespace(model_management=mm), "comfy.model_management": mm,
            "comfy.text_encoders": SimpleNamespace(gemma4=SimpleNamespace())}
        with patch.dict(sys.modules, modules), patch("tipo_native_lora.ensure_files", side_effect=AssertionError("No download")):
            with self.assertRaisesRegex(RuntimeError, "accelerator"):
                resolve_vision_path(AUTO_VISION)

    def test_accelerator_operation_failure_prevents_download(self):
        _, _, modules = self.runtime()
        with patch.dict(sys.modules, modules), \
                patch('vrs_int.torch_device._probe', side_effect=RuntimeError('Unsupported PyTorch operation')), \
                patch('tipo_native_lora.ensure_files') as fetch:
            with self.assertRaisesRegex(RuntimeError, 'Unsupported PyTorch operation'):
                resolve_vision_path(AUTO_VISION)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
