"""Offline failure, output-contract and provenance tests for the AI writer."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import prompt_writer_node as writer
import tipo_engine as te


class WriterTests(unittest.TestCase):
    def infer(self, output, finish_reason='stop', **kwargs):
        llm = Mock()
        llm.create_chat_completion.return_value = {
            'choices': [{'message': {'content': output}, 'finish_reason': finish_reason}]}
        with patch.object(te, '_ensure_deps', return_value=True), \
                patch.object(te, 'require_gpu_offload'), \
                patch.object(writer, '_resolve_model', return_value='/model.gguf'), \
                patch.object(te, 'load_tipo', return_value=llm) as load:
            result = writer.RSPromptWriter().run('하얀 셔츠를 입은 소녀', **kwargs)
        return result[0], llm.create_chat_completion.call_args.kwargs, load.call_args

    def test_empty_input_has_no_install_or_download(self):
        with patch.object(te, '_ensure_deps') as deps, patch.object(writer, '_resolve_model') as resolve:
            self.assertEqual(writer.RSPromptWriter().run('  '), ('',))
        deps.assert_not_called()
        resolve.assert_not_called()

    def test_tags_remove_reasoning_and_repeats(self):
        result, args, load = self.infer('<think>private</think>1girl, white_shirt, white_shirt, park',
                                       seed=2**64 - 1, gpu_layers=0)
        self.assertEqual(result, '1girl, white_shirt, park')
        self.assertEqual(args['seed'], (2**64 - 1) % (2**31))
        self.assertEqual(load.args, ('/model.gguf', 0))
        self.assertEqual(load.kwargs['n_ctx'], 4096)

    def test_natural_language_preserves_sentence_structure(self):
        prose = 'A girl in a white shirt stands in a park. Soft light falls across the path.'
        result, args, _ = self.infer(prose, output_format='natural_language', creativity='faithful')
        self.assertEqual(result, prose)
        self.assertIn('one fluent English paragraph', args['messages'][0]['content'])
        self.assertIn('Do not invent any details', args['messages'][0]['content'])

    def test_tag_phrases_preserve_subject_relationships(self):
        raw = ('```text\nTags: two adult women sitting opposite each other, '
               'one woman wearing a white sweater, '
               'the other woman wearing a black jacket, café, café\n```')
        result, _, _ = self.infer(raw)
        self.assertEqual(result, 'two adult women sitting opposite each other, '
                         'one woman wearing a white sweater, '
                         'the other woman wearing a black jacket, café')

    def test_tag_generation_receives_the_original_request(self):
        llm = Mock()
        llm.create_chat_completion.return_value = {
            'choices': [{'message': {'content': '2girls, facing each other, sitting'}}]}
        original = '여성 두 명이 마주 앉아 있다.'
        with patch.object(te, '_ensure_deps', return_value=True), \
                patch.object(te, 'require_gpu_offload'), \
                patch.object(writer, '_resolve_model', return_value='/model.gguf'), \
                patch.object(te, 'load_tipo', return_value=llm):
            result = writer.RSPromptWriter().run(original)
        self.assertEqual(result, ('2girls, facing each other, sitting',))
        final_input = llm.create_chat_completion.call_args.kwargs['messages'][-1]['content']
        self.assertEqual(original, final_input)

    def test_missing_dependency_is_visible_and_never_downloads(self):
        with patch.object(te, '_ensure_deps', return_value=False), patch.object(writer, '_resolve_model') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'Generation failed.*llama-cpp-python'):
                writer.RSPromptWriter().run('a park')
        resolve.assert_not_called()

    def test_empty_output_is_failure_instead_of_echoing_input(self):
        with self.assertRaisesRegex(RuntimeError, 'no usable prompt'):
            self.infer('<think>only reasoning</think>')

    def test_partial_output_is_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, 'truncated'):
            self.infer('white shirt, blue', finish_reason='length')

    def test_cpu_override_applies_to_writer(self):
        with patch.dict('os.environ', {'RS_TIPO_FORCE_CPU': '1'}):
            _, _, load = self.infer('white shirt', gpu_layers=-1)
        self.assertEqual(load.args, ('/model.gguf', 0))

    def test_default_writer_requests_gpu_and_limits_cpu_helpers(self):
        _, _, load = self.infer('white shirt')
        self.assertEqual(load.args, ('/model.gguf', -1))
        self.assertEqual(load.kwargs['n_threads'], 2)

    def test_gpu_unavailable_fails_before_model_download(self):
        with patch.object(te, '_ensure_deps', return_value=True), \
                patch.object(te, 'require_gpu_offload', side_effect=RuntimeError('CPU fallback is disabled')), \
                patch.object(writer, '_resolve_model') as resolve, \
                patch.object(te, 'load_tipo') as load:
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                writer.RSPromptWriter().run('a park')
        resolve.assert_not_called()
        load.assert_not_called()

    def test_default_download_is_pinned_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / writer.MODEL_FILE
            path.write_bytes(b'valid-model')
            with patch.object(te, '_gguf_dir', return_value=directory), \
                    patch.object(te, '_stream_download') as download, \
                    patch.object(writer, '_verified', set()), \
                    patch.object(writer, 'MODEL_SHA256', hashlib.sha256(b'valid-model').hexdigest()):
                self.assertEqual(writer._resolve_model(writer.AUTO_WRITER), str(path))
                self.assertIn('/resolve/' + writer.MODEL_REVISION + '/', download.call_args.args[0])
                path.write_bytes(b'damaged')
                with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                    writer._resolve_model(writer.AUTO_WRITER)

    def test_custom_missing_model_never_selects_another_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(te, '_gguf_dir', return_value=directory):
            with self.assertRaises(FileNotFoundError):
                writer._resolve_model('missing.gguf')
            with self.assertRaises(ValueError):
                writer._resolve_model('../model.gguf')


if __name__ == '__main__':
    unittest.main()
