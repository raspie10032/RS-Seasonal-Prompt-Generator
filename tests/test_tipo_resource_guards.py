"""Legacy TIPO cancellation, offload and lifetime guards without model execution."""
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import tipo_engine as te
import prompt_writer_node as writer


class ModelCacheLifetimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.gguf"
        self.path.write_bytes(b"test")
        self.factory = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        for patcher in (patch.dict(sys.modules, {
                "llama_cpp": SimpleNamespace(Llama=self.factory)}),
                patch.object(te, "_model_cache", {})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def borrow(self, **kwargs):
        # Test-only references are inspected after the scope, never used for
        # inference. Production must finish generation before exiting it.
        with te.tipo_inference():
            return te.load_tipo(str(self.path), **kwargs)

    def borrow_from_another_thread(self, **kwargs):
        results, errors = [], []

        def worker():
            try:
                results.append(self.borrow(**kwargs))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(5)
        self.assertFalse(thread.is_alive(), "Inference or cache lock leaked")
        self.assertEqual(errors, [])
        return results[0]

    def test_bare_cached_load_rejected_before_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "tipo_inference.*cache=False"):
            te.load_tipo(str(self.path))
        self.factory.assert_not_called()
        self.assertEqual(te._model_cache, {})

    def test_same_settings_reuse_and_new_settings_close_before_allocation(self):
        first = self.borrow(n_threads=2)
        self.assertIs(self.borrow(n_threads=2), first)
        first.close.assert_not_called()
        second = Mock()

        def replace(**kwargs):
            first.close.assert_called_once()
            self.assertEqual(te._model_cache, {})
            return second

        self.factory.side_effect = replace
        self.assertIs(self.borrow(n_threads=4), second)
        self.assertEqual(list(te._model_cache.values()), [second])
        self.assertIs(self.borrow(n_threads=4), second)
        self.assertEqual(self.factory.call_count, 2)
        second.close.assert_not_called()

    def test_nested_scope_cannot_close_outer_borrow(self):
        with te.tipo_inference():
            first = te.load_tipo(str(self.path), n_threads=2)
            with te.tipo_inference():
                self.assertIs(te.load_tipo(str(self.path), n_threads=2), first)
                with self.assertRaisesRegex(RuntimeError, "Cannot change cached"):
                    te.load_tipo(str(self.path), n_threads=4)
            first.close.assert_not_called()
            self.assertIs(te.load_tipo(str(self.path), n_threads=2), first)
        self.assertEqual(self.factory.call_count, 1)
        second = self.borrow_from_another_thread(n_threads=4)
        first.close.assert_called_once()
        self.assertEqual(list(te._model_cache.values()), [second])

    def test_constructor_failure_leaves_empty_cache_and_releases_ownership(self):
        first = self.borrow()
        self.factory.side_effect = RuntimeError("allocation failed")
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self.borrow(n_ctx=4096)
        first.close.assert_called_once()
        self.assertEqual(te._model_cache, {})
        replacement = Mock()
        self.factory.side_effect = None
        self.factory.return_value = replacement
        self.assertIs(self.borrow_from_another_thread(n_ctx=4096), replacement)
        self.assertEqual(list(te._model_cache.values()), [replacement])

    def test_close_failure_drops_invalid_handle_and_releases_ownership(self):
        first = self.borrow()
        first.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            self.borrow(n_threads=4)
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(te._model_cache, {})
        replacement = self.borrow_from_another_thread(n_threads=4)
        self.assertIsNot(replacement, first)
        self.assertEqual(list(te._model_cache.values()), [replacement])
        first.close.assert_called_once()

    def test_writer_waits_for_expander_generation_before_replacing_model(self):
        first_generating = threading.Event()
        release_first = threading.Event()
        second_waiting = threading.Event()
        events, results, errors, models = [], {}, [], []
        real_lock = threading.RLock()

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "writer":
                    second_waiting.set()
                real_lock.acquire()

            def __exit__(self, *args):
                real_lock.release()

        def construct(**kwargs):
            label = "first" if kwargs["n_ctx"] == 2048 else "second"
            events.append("load " + label)
            model = Mock(**kwargs)
            model.generating = False
            model.closed = False

            def generate(**settings):
                self.assertFalse(model.closed)
                model.generating = True
                events.append("generate " + label)
                try:
                    if label == "first":
                        first_generating.set()
                        if not release_first.wait(5):
                            raise RuntimeError("Test failed to release first generation")
                    self.assertFalse(model.closed)
                    events.append("finish " + label)
                    return {"choices": [{"message": {"content": "1girl, outdoors"},
                                         "finish_reason": "stop"}]}
                finally:
                    model.generating = False

            def close():
                self.assertFalse(model.generating, "A live model was evicted")
                model.closed = True
                events.append("close " + label)

            model.create_chat_completion.side_effect = generate
            model.close.side_effect = close
            models.append(model)
            return model

        def run(name, function):
            try:
                results[name] = function()
            except BaseException as exc:
                errors.append(exc)

        self.factory.side_effect = construct
        with patch.object(te, "_inference_lock", ObservedLock()), \
                patch.object(te, "_ensure_deps", return_value=True), \
                patch.object(te, "_resolve_model_path", return_value=str(self.path)), \
                patch.object(writer, "_resolve_model", return_value=str(self.path)):
            first = threading.Thread(target=run, args=("expand", lambda: te.expand_prompt(
                "1girl", str(self.path), "short", "", 0.5, 0, sort_preset=None)), daemon=True)
            second = threading.Thread(target=run, name="writer", args=("writer", lambda:
                writer.RSPromptWriter().run("1girl", gpu_layers=0, cpu_threads=4)), daemon=True)
            first.start()
            try:
                self.assertTrue(first_generating.wait(5))
                second.start()
                self.assertTrue(second_waiting.wait(5))
                self.assertEqual(self.factory.call_count, 1)
                models[0].close.assert_not_called()
            finally:
                release_first.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, {"expand": "1girl, outdoors", "writer": ("1girl, outdoors",)})
        self.assertEqual(events, ["load first", "generate first", "finish first", "close first",
                                  "load second", "generate second", "finish second"])
        self.assertEqual(list(te._model_cache.values()), [models[1]])


class DownloadCancellationTests(unittest.TestCase):
    def test_progress_cancellation_stops_reading_and_removes_partial_file(self):
        class InterruptProcessingException(Exception):
            pass

        source = io.BytesIO(b"x" * (2 * 1024 * 1024))
        source.headers = {"Content-Length": str(2 * 1024 * 1024)}
        read = Mock(wraps=source.read)
        source.read = read
        progress = Mock()
        progress.update_absolute.side_effect = InterruptProcessingException
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(te.urllib.request, "urlopen", return_value=source), \
                patch.object(te, "_progress_bar", return_value=progress):
            target = Path(directory) / "model.gguf"
            with self.assertRaises(InterruptProcessingException):
                te._stream_download("https://example.invalid/model", str(target))
            self.assertEqual(read.call_count, 1)
            self.assertEqual(list(Path(directory).iterdir()), [])


class OffloadTests(unittest.TestCase):
    def test_cpu_only_runtime_rejects_gpu_before_model_download(self):
        for layers in (-1, 8):
            with self.subTest(layers=layers), \
                    patch.dict(os.environ, {"RS_TIPO_FORCE_CPU": ""}), \
                    patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(
                        llama_supports_gpu_offload=lambda: False)}), \
                    patch.object(te, "_ensure_deps", return_value=True), \
                    patch.object(te, "_resolve_model_path") as resolve, \
                    patch.object(te, "load_tipo") as load:
                self.assertEqual(te.expand_prompt(
                    "1girl", "", "short", "", 0.5, 0, gpu_layers=layers), "1girl")
                resolve.assert_not_called()
                load.assert_not_called()

    def test_explicit_cpu_modes_do_not_require_gpu_runtime(self):
        llm = Mock()
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "1girl, outdoors"}}]}
        for layers, override in ((0, ""), (-1, "1")):
            with self.subTest(layers=layers, override=override), \
                    patch.dict(os.environ, {"RS_TIPO_FORCE_CPU": override}), \
                    patch.object(te, "_ensure_deps", return_value=True), \
                    patch.object(te, "require_gpu_offload", side_effect=AssertionError), \
                    patch.object(te, "_resolve_model_path", return_value="model.gguf"), \
                    patch.object(te, "load_tipo", return_value=llm) as load:
                self.assertEqual(te.expand_prompt(
                    "1girl", "", "short", "", 0.5, 0, sort_preset=None,
                    gpu_layers=layers), "1girl, outdoors")
                load.assert_called_once_with("model.gguf", 0)


if __name__ == "__main__":
    unittest.main()
