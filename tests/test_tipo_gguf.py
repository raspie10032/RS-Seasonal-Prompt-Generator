"""Q4 source selection, compatibility and bounded first-use downloads."""
from pathlib import Path
import os
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import tipo_gguf as q4
from gemma_tipo_native_node import GemmaTipoNative
from gemma_prompt_expand_node import GemmaPromptExpandNative


class SourceTests(unittest.TestCase):
    def test_old_workflows_keep_connected_clip_without_loading(self):
        clip = object()
        with patch.object(q4, "_load", side_effect=AssertionError):
            self.assertIs(q4.resolve_clip(clip), clip)
        for cls in (GemmaTipoNative, GemmaPromptExpandNative):
            self.assertNotIn("clip", cls.INPUT_TYPES()["required"])
            self.assertIn("clip", cls.INPUT_TYPES()["optional"])
            self.assertEqual(cls.INPUT_TYPES()["optional"]["model_source"][1]["default"], "connected_clip")

    def test_ambiguous_sources_and_missing_clip_fail_before_download(self):
        with patch.object(q4, "ensure_files", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "Connect"):
                q4.resolve_clip(None)
            with self.assertRaisesRegex(ValueError, "Disconnect"):
                q4.resolve_clip(object(), "heretic_ara_q4")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                q4.resolve_clip(None, "other")

    def test_image_download_is_lazy_and_local_files_stay_local(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / name for name in ("text.gguf", "tokenizer.json", "vision.gguf")]
            for path in paths:
                path.write_bytes(b"fixture")
            stubs = {"gguf": SimpleNamespace(), "folder_paths": SimpleNamespace(models_dir=temp),
                     "comfy": SimpleNamespace(utils=SimpleNamespace()), "comfy.utils": SimpleNamespace()}
            with patch.dict("sys.modules", stubs), patch.object(q4, "ensure_files", side_effect=AssertionError), \
                    patch.object(q4, "check_native_runtime"), \
                    patch.object(q4, "_load", return_value="clip") as load:
                self.assertEqual(q4.resolve_clip(None, "heretic_ara_q4", gguf_path=str(paths[0]),
                    tokenizer_path=str(paths[1]), mmproj_path="missing-unused"), "clip")
                self.assertIsNone(load.call_args.args[2])
                q4.resolve_clip(None, "heretic_ara_q4", image=object(), gguf_path=str(paths[0]),
                    tokenizer_path=str(paths[1]), mmproj_path=str(paths[2]))
                self.assertEqual(load.call_args.args[2][0], str(paths[2].resolve()))

    def test_new_nodes_do_not_download_on_empty_input(self):
        with patch.object(q4, "ensure_files", side_effect=AssertionError):
            self.assertEqual(GemmaTipoNative().run(model_source="heretic_ara_q4"), ("", ""))
            self.assertEqual(GemmaPromptExpandNative().run(model_source="heretic_ara_q4"), ("", "", ""))

    def test_invalid_manual_tokenizer_fails_before_default_model_download(self):
        with patch.object(q4, "ensure_files", side_effect=AssertionError("No download")):
            with self.assertRaises(FileNotFoundError):
                q4.resolve_clip(None, "heretic_ara_q4", tokenizer_path="/missing/tokenizer.json")

    def test_manual_model_edit_with_restored_size_and_mtime_invalidates_loaded_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"before")
            stamp = path.stat()
            before = q4._signature(path)
            if os.name == "nt":
                # Consecutive writes may share one Windows filesystem clock
                # tick. Exercise a distinct change time, not clock resolution.
                time.sleep(0.1)
            path.write_bytes(b"after!")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertNotEqual(q4._signature(path), before)

    def test_pins_have_full_hashes_sizes_and_immutable_revisions(self):
        for entry in (q4.TEXT, q4.VISION, q4.TOKENIZER):
            self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
            for size, digest in entry["files"].values():
                self.assertGreater(size, 0)
                self.assertRegex(digest, r"^[a-f0-9]{64}$")


class RuntimeTests(unittest.TestCase):
    def modules(self):
        class Clip:
            tokenize = generate = decode = lambda *args: None
        class Gemma:
            tokenizer = object()
        sd = SimpleNamespace(CLIP=Clip)
        ops = SimpleNamespace(manual_cast=type)
        patcher = SimpleNamespace(ModelPatcher=type)
        gemma = SimpleNamespace(Gemma4Base=type, Gemma4_E2B=Gemma, Gemma4_E2B_Config=type,
            Gemma4Transformer=type, gemma4_te=type, Gemma4MultiModalProjector=type, Gemma4VisionEncoder=type)
        encoders = SimpleNamespace(gemma4=gemma)
        return sd, gemma, {"gguf": SimpleNamespace(),
            "comfy": SimpleNamespace(sd=sd, ops=ops, model_patcher=patcher, text_encoders=encoders),
            "comfy.sd": sd, "comfy.ops": ops, "comfy.model_patcher": patcher, "comfy.text_encoders": encoders}

    def test_old_comfy_fails_before_any_model_download(self):
        sd, _, modules = self.modules()
        del sd.CLIP.generate
        with patch.dict("sys.modules", modules), patch.object(q4, "ensure_files") as fetch:
            with self.assertRaisesRegex(RuntimeError, "CLIP.generate"):
                q4.resolve_clip(None, "heretic_ara_q4")
            fetch.assert_not_called()

    def test_missing_comfy_has_actionable_error_before_download(self):
        with patch.dict("sys.modules", {"gguf": SimpleNamespace(), "comfy": None}), \
                patch.object(q4, "ensure_files") as fetch:
            with self.assertRaisesRegex(RuntimeError, "ComfyUI with Gemma 4"):
                q4.resolve_clip(None, "heretic_ara_q4")
            fetch.assert_not_called()

    def test_image_features_required_only_with_image(self):
        _, gemma, modules = self.modules()
        del gemma.Gemma4VisionEncoder
        with patch.dict("sys.modules", modules):
            q4.check_native_runtime()
            with self.assertRaisesRegex(RuntimeError, "Gemma4VisionEncoder"):
                q4.check_native_runtime(has_image=True)
