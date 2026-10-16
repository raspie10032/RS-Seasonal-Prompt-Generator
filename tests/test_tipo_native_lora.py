"""First-use download integrity and old-workflow behavior, with no network."""
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import tipo_native_lora as lora
from gemma_tipo_native_node import GemmaTipoNative
from gemma_prompt_expand_node import GemmaPromptExpandNative


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"a tiny fake adapter"
        self.entry = {"repo": "example/test", "revision": "a" * 40, "files": {
            "adapter_model.safetensors": (len(self.payload), hashlib.sha256(self.payload).hexdigest())}}
        self.catalog = patch.dict(lora.CATALOG, {"test": self.entry})
        self.catalog.start()
        self.addCleanup(self.catalog.stop)

    def test_download_then_offline_reuse(self):
        with patch.object(lora.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)) as fetch:
            folder = lora.ensure_adapter("test", self.root)
        self.assertEqual((folder / "adapter_model.safetensors").read_bytes(), self.payload)
        self.assertIn("/resolve/" + "a" * 40 + "/", fetch.call_args.args[0].full_url)
        with patch.object(lora.urllib.request, "urlopen", side_effect=AssertionError("offline")):
            self.assertEqual(lora.ensure_adapter("test", self.root), folder)

    def test_bad_download_never_becomes_a_cached_model(self):
        for content in (b"short", b"x" * len(self.payload), self.payload * 2):
            with self.subTest(content=content), patch.object(lora.urllib.request, "urlopen", return_value=io.BytesIO(content)):
                with self.assertRaises(ValueError):
                    lora.ensure_adapter("test", self.root)
            self.assertEqual(list((self.root / "test").iterdir()), [])

    def test_network_failure_cleans_partial_file(self):
        with patch.object(lora.urllib.request, "urlopen", side_effect=OSError("disconnected")):
            with self.assertRaises(OSError):
                lora.ensure_adapter("test", self.root)
        self.assertEqual(list((self.root / "test").iterdir()), [])

    def test_cancellation_cleans_partial_file(self):
        with patch.object(lora.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)):
            with self.assertRaises(InterruptedError):
                lora.ensure_adapter("test", self.root, Mock(side_effect=InterruptedError))
        self.assertEqual(list((self.root / "test").iterdir()), [])

    def test_changed_file_invalidates_integrity_cache_even_with_same_mtime(self):
        with patch.object(lora.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)):
            folder = lora.ensure_adapter("test", self.root)
        target = folder / "adapter_model.safetensors"
        lora.ensure_adapter("test", self.root)
        old = target.stat()
        target.write_bytes(b"x" * len(self.payload))
        os.utime(target, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            lora.ensure_adapter("test", self.root)

    def test_unchanged_local_file_reuses_checksum_verification(self):
        target = self.root / "adapter.safetensors"
        target.write_bytes(self.payload)
        expected = self.entry["files"]["adapter_model.safetensors"]
        # Exercise the real change timestamp, including the Windows API in CI.
        self.assertIsNotNone(lora._file_stat(target)[-1])
        lora._check_file(target, expected)
        before = lora._check_digest.cache_info()
        lora._check_file(target, expected)
        after = lora._check_digest.cache_info()
        self.assertEqual(after.hits, before.hits + 1)
        self.assertEqual(after.misses, before.misses)

    def test_unavailable_change_timestamp_never_reuses_checksum(self):
        target = self.root / "adapter.safetensors"
        target.write_bytes(self.payload)
        expected = self.entry["files"]["adapter_model.safetensors"]
        signature = (0, 0, len(self.payload), 0, None)
        with patch.object(lora, "_file_stat", return_value=signature):
            lora._check_file(target, expected)
            target.write_bytes(b"x" * len(self.payload))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                lora._check_file(target, expected)

    def test_existing_corrupt_file_is_preserved_and_rejected(self):
        folder = self.root / "test"
        folder.mkdir()
        target = folder / "adapter_model.safetensors"
        target.write_bytes(b"keep this")
        with patch.object(lora.urllib.request, "urlopen", side_effect=AssertionError):
            with self.assertRaises(ValueError):
                lora.ensure_adapter("test", self.root)
        self.assertEqual(target.read_bytes(), b"keep this")

    def test_none_does_not_touch_clip_or_download(self):
        clip = object()
        with patch.object(lora, "ensure_adapter", side_effect=AssertionError):
            self.assertIs(lora.apply_tipo_lora(clip), clip)
            with self.assertRaises(ValueError):
                lora.apply_tipo_lora(clip, "../../unknown")
            with self.assertRaisesRegex(ValueError, "native Gemma"):
                lora.apply_tipo_lora(clip, "pilot-001")


class NodeTests(unittest.TestCase):
    def test_both_outputs_use_one_adapter_clone_and_preserve_original_input(self):
        import gemma_prompt_expand_node as node
        original, adapted = Mock(), Mock()
        adapted.tokenize.return_value = {"gemma4": []}
        adapted.decode.side_effect = ["A woman stands in a winter park.", "1girl, standing, winter, park"]
        with patch.object(node, "apply_tipo_lora", return_value=adapted) as apply:
            tags, prose, _ = GemmaPromptExpandNative().run(
                clip=original, prompt="겨울 공원", tag_length="short", max_tokens=64,
                temperature=0, ban_tags="", seed=1, tipo_lora="pilot-001")
        apply.assert_called_once_with(original, "pilot-001")
        self.assertEqual(adapted.generate.call_count, 2)
        original.generate.assert_not_called()
        self.assertIn("park", tags)
        self.assertIn("winter", prose)

    def test_warning_and_default_in_both_nodes(self):
        for cls in (GemmaTipoNative, GemmaPromptExpandNative):
            choices, options = cls.INPUT_TYPES()["optional"]["tipo_lora"]
            self.assertEqual(options["default"], "none")
            self.assertIn("9.57 GiB", options["tooltip"])
            self.assertIn("GGUF", options["tooltip"])
            self.assertEqual(choices, list(lora.CHOICES))


if __name__ == "__main__":
    unittest.main()
