"""Native node contracts and byte-preserving conversion, without GPU/models."""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gemma_tipo_native_node as native
import tipo_engine
from scripts.prepare_gemma4_native import prepare, read_header, native_name


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.clip = Mock()
        self.clip.tokenize.return_value = {"gemma4": [[(1, 1.0)]]}
        self.clip.decode.return_value = "1girl, blue_eyes, scarf, coat"
        self.args = dict(clip=self.clip, prompt="금발 소녀", tag_length="short",
                         max_tokens=64, temperature=0, ban_tags="scarf", seed=42)

    def test_generate_without_optional_install_or_llama(self):
        with patch.object(tipo_engine, "_ensure_formatter", side_effect=AssertionError), \
             patch.object(tipo_engine, "_pip_install", side_effect=AssertionError):
            tags, raw = native.GemmaTipoNative().run(**self.args)
        self.assertEqual(tags, "1girl, blue_eyes, coat")
        self.assertIn("scarf", raw)
        self.assertFalse(self.clip.generate.call_args.kwargs["do_sample"])
        self.assertEqual(self.clip.generate.call_args.kwargs["seed"], 42)
        self.assertEqual(self.clip.generate.call_args.kwargs["repetition_penalty"], 1.05)
        kw = self.clip.tokenize.call_args.kwargs
        self.assertFalse(kw["skip_template"])
        self.assertFalse(kw["thinking"])

    def test_repetition_penalty_reaches_native_sampler_without_filtering_raw(self):
        _, raw = native.GemmaTipoNative().run(**self.args, repetition_penalty=1.15)
        self.assertEqual(self.clip.generate.call_args.kwargs["repetition_penalty"], 1.15)
        self.assertEqual(raw, self.clip.decode.return_value)

    def test_invalid_repetition_penalty_rejected_before_loading(self):
        for value in (float('nan'), float('inf'), 0.99, 2.01, True, '1.15'):
            with self.subTest(value=value), patch.object(native, 'resolve_clip') as resolve:
                with self.assertRaisesRegex(ValueError, 'repetition_penalty'):
                    native.GemmaTipoNative().run(**self.args, repetition_penalty=value)
                resolve.assert_not_called()
        self.clip.generate.assert_not_called()

    def test_image_is_passed_without_host_conversion(self):
        image = SimpleNamespace(shape=(1, 32, 48, 3))
        native.GemmaTipoNative().run(**self.args, image=image)
        self.assertIs(self.clip.tokenize.call_args.kwargs["image"], image)

    def test_two_generations_release_captured_kv_before_next_prompt(self):
        from gemma_prompt_expand_node import GemmaPromptExpandNative
        active_graph = [False]
        def generate(*args, **kwargs):
            if active_graph[0]:
                raise RuntimeError("Stale graph references a previous prompt's KV cache")
            active_graph[0] = True
            return []
        def cleanup():
            active_graph[0] = False
        self.clip.generate.side_effect = generate
        self.clip.decode.side_effect = ["Create an autumn park scene.", "autumn, park"]
        with patch.dict("sys.modules", {"comfy.model_prefetch": SimpleNamespace(cleanup_prefetch_queues=cleanup)}):
            tags, prose, _ = GemmaPromptExpandNative().run(**self.args, output_format="both")
        self.assertEqual(tags, "autumn, park")
        self.assertEqual(prose, "Create an autumn park scene.")
        self.assertFalse(active_graph[0])

    def test_empty_input_does_not_load_model(self):
        self.args["prompt"] = " "
        self.assertEqual(native.GemmaTipoNative().run(**self.args), ("", ""))
        self.clip.generate.assert_not_called()

    def test_wrong_clip_is_rejected_before_inference(self):
        self.clip.tokenize.return_value = {"l": []}
        with self.assertRaisesRegex(ValueError, "Expected a Gemma"):
            native.GemmaTipoNative().run(**self.args)
        self.clip.generate.assert_not_called()

    def test_old_comfy_actionable_error(self):
        self.args["clip"] = object()
        with self.assertRaisesRegex(RuntimeError, "Update ComfyUI"):
            native.GemmaTipoNative().run(**self.args)

    def test_bad_image_shape(self):
        for shape in ((2, 32, 32, 3), (1, 0, 32, 3), (1, 32, 32, 4), (32, 32, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                native.GemmaTipoNative().run(**self.args, image=SimpleNamespace(shape=shape))

    def test_runtime_errors_propagate_without_cpu_retry(self):
        for message in ("interrupted", "CUDA out of memory"):
            self.clip.generate.reset_mock()
            self.clip.generate.side_effect = RuntimeError(message)
            with self.assertRaisesRegex(RuntimeError, message):
                native.GemmaTipoNative().run(**self.args)
            self.assertEqual(self.clip.generate.call_count, 1)

    def test_empty_output_is_not_successful_fallback(self):
        self.clip.decode.return_value = ""
        with self.assertRaisesRegex(RuntimeError, "empty response"):
            native.GemmaTipoNative().run(**self.args)

    def test_reasoning_is_removed_before_tags(self):
        self.assertEqual(native.clean_tags("<think>secret plan, hat</think>\n- coat\n- blue_eyes", "blue eyes", ""), "coat")
        self.clip.decode.return_value = "<think>unfinished"
        with self.assertRaisesRegex(RuntimeError, "only reasoning"):
            native.GemmaTipoNative().run(**self.args)

    def test_expression_tags_do_not_swallow_later_names(self):
        raw = '1girl, :<, >_<, <3, example_hero_(ghost), example_artist, example_series<end_of_turn>'
        self.assertEqual(native.clean_tags(raw, '', ''), raw.removesuffix('<end_of_turn>'))
        self.assertEqual(native.clean_tags('1girl, :<, example_series<end_of_turn', '', ''),
                         '1girl, :<, example_series')

    def test_native_cleanup_keeps_nested_titles_and_still_enforces_bans(self):
        raw = 'title, long title, blue hair, BLUE_HAIR, 2girls, 1girl, skirt, white skirt'
        self.assertEqual(native.clean_tags(raw, 'blue_hair', '1girl'),
                         'title, long title, 1girl, skirt, white skirt')


def write_weights(path, payloads):
    header, body = {}, b""
    for name, data in payloads.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [len(body), len(body) + len(data)]}
        body += data
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": {
            "hidden_size": 1536, "num_hidden_layers": 35, "vocab_size": 262144}}))
        (self.source / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
        self.payloads = {
            "model.language_model.embed_tokens.weight": b"tokens",
            "model.language_model.embed_tokens_per_layer.weight": b"ple",
            "model.embed_vision.embedding_projection.weight": b"paired-vision",
            "model.vision_tower.patch_embedder.input_proj.weight": b"vision",
            **{f"model.language_model.layers.{i}.input_layernorm.weight": bytes([i]) for i in range(35)},
        }
        self.weight = self.source / "model.safetensors"
        write_weights(self.weight, self.payloads)
        self.output = self.root / "native.safetensors"

    def test_payloads_and_tokenizer_preserved(self):
        before = self.weight.read_bytes()
        result = prepare(self.source, self.output)
        self.assertFalse(result["precision_changed"])
        self.assertEqual(before, self.weight.read_bytes())
        header, offset = read_header(self.output)
        raw = self.output.read_bytes()[offset:]
        for name, data in self.payloads.items():
            start, end = header[native_name(name)]["data_offsets"]
            self.assertEqual(raw[start:end], data)
        start, end = header["tokenizer_json"]["data_offsets"]
        self.assertEqual(raw[start:end], (self.source / "tokenizer.json").read_bytes())

    def test_sharded_checkpoint(self):
        items = list(self.payloads.items())
        mapping = {}
        for name, payload in (("model-1.safetensors", dict(items[:20])), ("model-2.safetensors", dict(items[20:]))):
            write_weights(self.source / name, payload)
            mapping.update({k: name for k in payload})
        (self.source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        self.weight.unlink()
        result = prepare(self.source, self.output)
        self.assertEqual(result["tensor_count"], len(self.payloads))

    def test_existing_destination_never_replaced(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep")

    def test_output_in_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare(self.source, self.source / "converted.safetensors")

    def test_missing_vision_rejected(self):
        self.payloads.pop("model.embed_vision.embedding_projection.weight")
        write_weights(self.weight, self.payloads)
        with self.assertRaisesRegex(ValueError, "Missing"):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_index_path_escape_rejected(self):
        (self.source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../elsewhere.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "inside"):
            prepare(self.source, self.output)

    def test_truncated_weights_rejected(self):
        self.weight.write_bytes(self.weight.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "range"):
            prepare(self.source, self.output)

    def test_unsupported_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            native_name("adapter.lora_A.weight")


if __name__ == "__main__":
    unittest.main()
