"""Offline contract and regression checks. No model, pip or network required."""
import contextlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import seasonal_fashion_prompt_node as seasonal
import gemma_tipo_node as node
import tipo_engine as te
import tipo_vision_engine as ve
import prompt_writer_node as writer

ROOT = Path(__file__).resolve().parents[1]


def seasonal_inputs(**overrides):
    values = {}
    for key, spec in seasonal.SeasonalFashionPromptNode.INPUT_TYPES()['required'].items():
        options = spec[1] if len(spec) > 1 else {}
        values[key] = options.get('default', spec[0][0])
    values.update(overrides)
    return values


class SeasonalTests(unittest.TestCase):
    def setUp(self):
        self.node = seasonal.SeasonalFashionPromptNode()

    def test_all_seeds_repeat_without_changing_global_random(self):
        state = random.getstate()
        for seed in (0, 1, 42, -1, 10**15, 10**16 + 9, 2**64 - 1):
            with self.subTest(seed=seed):
                args = seasonal_inputs(seed=seed)
                first = self.node.generate_prompt(**args)
                self.assertTrue(first[0])
                self.assertEqual(first, self.node.generate_prompt(**args))
        self.assertEqual(state, random.getstate())

    def test_ui_schema_does_not_consume_random(self):
        state = random.getstate()
        self.node.INPUT_TYPES()
        self.assertEqual(state, random.getstate())

    def test_concurrent_seeded_requests_are_independent(self):
        seeds = list(range(20)) * 3
        expected = [self.node.generate_prompt(**seasonal_inputs(seed=s)) for s in seeds]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda s: self.node.generate_prompt(**seasonal_inputs(seed=s)), seeds))
        self.assertEqual(expected, actual)

    def test_every_clothing_toggle_combination(self):
        categories = ('top', 'bottom', 'one_piece', 'accessory', 'hat', 'shoes', 'socks')
        fixture = {s: [[c, c + '_item'] for c in categories]
                   for s in ('spring', 'summer', 'autumn', 'winter')}
        with patch.object(seasonal, 'seasonal_fashion', fixture):
            for season, bits in itertools.product(fixture, itertools.product((False, True), repeat=7)):
                with self.subTest(season=season, bits=bits):
                    text = self.node.get_fashion(season, *['random' if b else 'off' for b in bits], rng=random.Random(1))
                    tags = set(text.split(', ')) if text else set()
                    for c, enabled in zip(categories, bits):
                        if not enabled:
                            self.assertNotIn(c + '_item', tags)
                    for c, enabled in zip(categories[3:], bits[3:]):
                        self.assertEqual(c + '_item' in tags, enabled)
                    if any(bits[:3]):
                        self.assertTrue(tags & {'top_item', 'bottom_item', 'one_piece_item'})
                    self.assertFalse('one_piece_item' in tags and bool(tags & {'top_item', 'bottom_item'}))

    def test_explicit_choices_and_background_modes(self):
        for mode in ('on', 'off', 'simple'):
            result, = self.node.generate_prompt(**seasonal_inputs(
                season='spring', background_toggle=mode, weather_toggle='TEST_WEATHER',
                time_toggle='TEST_TIME', composition_toggle='TEST_COMPOSITION',
                gaze_toggle='TEST_GAZE', pose_toggle='TEST_POSE', body_direction_toggle='TEST_BODY'))
            for text in ('spring', 'TEST_COMPOSITION', 'TEST_GAZE', 'TEST_POSE', 'TEST_BODY'):
                self.assertIn(text, result)
            self.assertEqual('TEST_WEATHER' in result, mode == 'on')
            self.assertEqual('TEST_TIME' in result, mode == 'on')
            self.assertEqual('simple background' in result, mode == 'simple')

    def test_background_conditions_work_with_weather_and_time_off(self):
        with patch.object(seasonal, 'seasonal_backgrounds', {'autumn': [['outdoors', 'autumn forest']]}):
            background, situation = self.node.get_background_and_situation('autumn', 'off', 'off', 'on')
            self.assertIn('forest', background)
            self.assertIn('surrounded by trees', situation)
            self.assertIn('falling leaves', situation)
            self.assertEqual(self.node.get_background_and_situation('autumn', 'off', 'off', 'off')[1], '')

    def test_compound_situations_require_all_context_terms(self):
        for weather, background, present in (
                ('misty', 'forest', True), ('misty', 'lakeside', False),
                ('sunny', 'forest', False)):
            with self.subTest(weather=weather, background=background):
                conditions = seasonal.additional_situations['misty_forest']['conditions']
                self.assertEqual(seasonal.situation_matches(conditions, [weather, background]), present)
        self.assertTrue(seasonal.situation_matches(['foggy&mountain', 'misty&mountain'], ['misty', 'mountain']))
        self.assertFalse(seasonal.situation_matches(['rain'], ['rainforest']))
        self.assertFalse(seasonal.situation_matches(['', 'misty&'], ['misty']))
        conditions = seasonal.additional_situations['sunrise']['conditions']
        self.assertFalse(seasonal.situation_matches(conditions, ['morning', 'misty']))
        self.assertTrue(seasonal.situation_matches(conditions, ['morning', 'sunny']))

    def test_wanna_shot_keeps_fashion_and_is_repeatable(self):
        args = seasonal_inputs(wanna_shot='Yes', seed=42)
        result = self.node.generate_prompt(**args)
        self.assertIn('cafe, coffee, holding a tumbler', result[0])
        self.assertEqual(result, self.node.generate_prompt(**args))

    def test_empty_and_commented_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.csv'
            path.write_text('')
            self.assertEqual(seasonal.load_data_from_csv(path), {})
            path.write_text('season,category,item\n # comment\n\nspring, top , shirt \n')
            self.assertEqual(seasonal.load_data_from_csv(path), {'spring': [['top', 'shirt']]})

    def test_bundled_fashion_has_all_categories_and_no_duplicate_rows(self):
        expected = {'top', 'bottom', 'one_piece', 'accessory', 'hat', 'shoes', 'socks'}
        self.assertEqual(set(seasonal.seasonal_fashion), {'spring', 'summer', 'autumn', 'winter'})
        for season, rows in seasonal.seasonal_fashion.items():
            with self.subTest(season=season):
                self.assertEqual({row[0] for row in rows}, expected)
                self.assertEqual(len(rows), len({tuple(row) for row in rows}))
                self.assertTrue(all(row[1].strip() for row in rows))


class TagTests(unittest.TestCase):
    def setUp(self):
        self.formatter = patch.object(te, '_ensure_formatter', return_value=False)
        self.formatter.start()
        self.addCleanup(self.formatter.stop)

    def test_dedup_keeps_counts_and_character_identities(self):
        self.assertEqual(te._dedup('1girl, 2girls, 3girls, 2B, 9S, character (a), character (b)'),
                         '1girl, 2girls, 3girls, 2B, 9S, character (a), character (b)')
        self.assertEqual(te._dedup('blue_hair, Blue Hair, (blue hair:1.2)'), 'blue_hair')

    def test_user_count_wins_after_cleanup(self):
        self.assertEqual(te._postprocess('2girls, 3girls, multiple_girls, 1boy', '', 'danbooru', '3girls'),
                         '3girls, 1boy')

    def test_extraction_keeps_dotted_names_and_decimal_weights(self):
        raw = '1girl, d.va_(overwatch), (blue hair:1.2), 2b_(nier:automata)'
        self.assertEqual(te._extract_tags(raw), raw)
        self.assertEqual(te._postprocess(te._extract_tags(raw), 'blue_hair', 'danbooru'),
                         '1girl, d.va_(overwatch), 2b_(nier:automata)')

    def test_bans_are_enforced_without_formatter(self):
        result = te._postprocess('blue_hair, (red hair:1.2), green hair, 1girl', 'BLUE HAIR, red_hair', 'danbooru')
        self.assertEqual(result, 'green hair, 1girl')

    def test_specific_tags_suppress_only_redundant_suffixes(self):
        result = te._postprocess('skirt, white skirt, dress, dress shirt', '', 'danbooru')
        self.assertEqual(result, 'white skirt, dress, dress shirt')
        self.assertEqual(te._drop_redundant('skirt, WHITE_SKIRT, dress, dress shirt'),
                         'WHITE_SKIRT, dress, dress shirt')

    def test_reasoning_is_removed_in_both_modes(self):
        cases = {
            '<think>blue hair, thinking</think>1girl, red hair': '1girl, red hair',
            '<think>unfinished, hidden': '',
            '<|channel>thought hidden, plan<channel|>1girl, smile<end_of_turn>': '1girl, smile',
            '<|channel>thought hidden, plan': '',
            '<|channel|>analysis hidden, plan<|channel|>final1girl, smile': '1girl, smile',
            'Here are your tags:\n- 1girl, clothing: white shirt\n2. blue eyes': '1girl, white shirt, blue eyes',
            '<|general|>1girl, smile<|quality|>masterpiece': '1girl, smile, masterpiece',
            '': '',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(te._extract_tags(raw), expected)
                self.assertEqual(ve._extract_tags(raw), expected)

    def test_text_generation_keeps_input_tags_and_passes_sampling(self):
        llm = Mock()
        llm.create_chat_completion.return_value = {'choices': [{'message': {'content': '3girls, blue_hair, outdoors'}}]}
        with patch.object(te, '_ensure_deps', return_value=True), patch.object(te, 'load_tipo', return_value=llm) as load:
            result = te.expand_prompt('2girls', '/model.gguf', 'short', 'blue hair', 0.7, 42, gpu_layers=7)
        self.assertEqual(result, '2girls, outdoors')
        load.assert_called_once_with('/model.gguf', 7)
        self.assertEqual(llm.create_chat_completion.call_args.kwargs['seed'], 42)
        self.assertEqual(llm.create_chat_completion.call_args.kwargs['temperature'], 0.7)

    def test_banning_all_successful_output_returns_empty(self):
        llm = Mock()
        llm.create_chat_completion.return_value = {'choices': [{'message': {'content': '1girl'}}]}
        with patch.object(te, '_ensure_deps', return_value=True), patch.object(te, 'load_tipo', return_value=llm):
            self.assertEqual(te.expand_prompt('1girl', '/model.gguf', 'short', '1girl', .5, 0), '')

    def test_korean_input_is_not_merged_into_successful_tags(self):
        llm = Mock()
        llm.create_chat_completion.return_value = {'choices': [{'message': {'content': '1girl, white shirt'}}]}
        with patch.object(te, '_ensure_deps', return_value=True), patch.object(te, 'load_tipo', return_value=llm):
            self.assertEqual(te.expand_prompt('하얀 옷을 입은 소녀', '/model.gguf', 'short', '', .5, 0), '1girl, white shirt')

    def test_text_failure_preserves_original(self):
        with patch.object(te, '_ensure_deps', side_effect=RuntimeError('unavailable')):
            self.assertEqual(te.expand_prompt('original', '', 'short', '', .5, 0), 'original')

    def test_available_inference_does_not_depend_on_optional_sorter(self):
        with patch.object(te, '_imported_llama_version', return_value=(0, 3, 23)), \
                patch.object(te, '_have', return_value=False), patch.object(te, '_pip_install') as pip:
            self.assertTrue(te._ensure_deps())
            pip.assert_not_called()

    def test_model_cache_includes_offload_and_context_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            path.write_bytes(b'test')
            llama = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
            with patch.dict(sys.modules, {'llama_cpp': SimpleNamespace(Llama=llama)}), patch.object(te, '_model_cache', {}):
                cpu = te.load_tipo(str(path), 0)
                gpu = te.load_tipo(str(path), 8)
                longer = te.load_tipo(str(path), 0, n_ctx=4096)
                self.assertIs(cpu, te.load_tipo(str(path), 0))
                self.assertIsNot(cpu, gpu)
                self.assertIsNot(cpu, longer)
                self.assertEqual(longer.n_ctx, 4096)
                self.assertEqual(gpu.n_gpu_layers, 8)
                self.assertEqual(gpu.n_threads, 2)
                self.assertEqual(gpu.n_threads_batch, 2)
                self.assertEqual(llama.call_count, 3)
                more_threads = te.load_tipo(str(path), 8, n_threads=4)
                self.assertIsNot(gpu, more_threads)
                self.assertEqual(more_threads.n_threads_batch, 4)

    def test_cpu_only_backend_rejects_requested_gpu(self):
        with patch.dict(sys.modules, {'llama_cpp': SimpleNamespace(llama_supports_gpu_offload=lambda: False)}):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                te.require_gpu_offload()


class DependencyTests(unittest.TestCase):
    def test_installed_version_does_not_require_pip_or_importing_native_code(self):
        with patch.object(te.importlib.metadata, 'version', return_value='0.3.35'), \
                patch.object(te.subprocess, 'check_output') as command:
            self.assertEqual(te._installed_llama_version(), (0, 3, 35))
            command.assert_not_called()

    def test_uv_installs_into_the_running_python_when_pip_is_absent(self):
        with patch.object(te.importlib.util, 'find_spec', return_value=None), \
                patch.object(te.shutil, 'which', return_value='/bin/uv'), \
                patch.object(te.subprocess, 'check_call') as command:
            te._pip_install(['tipo-kgen', '--no-deps'])
        command.assert_called_once_with([
            '/bin/uv', 'pip', 'install', '--python', sys.executable,
            'tipo-kgen', '--no-deps'])

    def test_broken_installed_gpu_wheel_is_reported_without_reinstallation(self):
        with patch.object(te, '_installed_llama_version', return_value=(0, 3, 35)), \
                patch.object(te, '_imported_llama_version', return_value=None), \
                patch.object(te, '_restart_needed', return_value=False), \
                patch.object(te, '_DEPS_ATTEMPTED', False), \
                patch.object(te, '_LLAMA_IMPORT_ERROR', 'libcudart.so.13 is missing'), \
                patch.object(te, '_pip_install') as install, \
                contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertFalse(te._ensure_deps())
        install.assert_not_called()
        self.assertIn('libcudart.so.13 is missing', log.getvalue())


class DownloadTests(unittest.TestCase):
    def response(self, content, total):
        stream = io.BytesIO(content)
        stream.headers = {'Content-Length': str(total)}
        return stream

    def test_complete_file_is_published_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.gguf'
            with patch.object(te.urllib.request, 'urlopen', return_value=self.response(b'hello', 5)) as urlopen:
                te._stream_download('https://example.invalid/model', str(target))
                te._stream_download('https://example.invalid/model', str(target))
            self.assertEqual(target.read_bytes(), b'hello')
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(list(Path(directory).glob('*.part')), [])

    def test_truncated_empty_and_interrupted_downloads_are_not_published(self):
        for payload, total in ((b'hello', 10), (b'', 0)):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'model.gguf'
                with patch.object(te.urllib.request, 'urlopen', return_value=self.response(payload, total)):
                    with self.assertRaises(IOError):
                        te._stream_download('https://example.invalid/model', str(target))
                self.assertEqual(list(Path(directory).iterdir()), [])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(te.urllib.request, 'urlopen', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    te._stream_download('https://example.invalid/model', str(Path(directory) / 'model'))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_parallel_requests_publish_once(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / 'model.gguf')
            with patch.object(te.urllib.request, 'urlopen', side_effect=lambda *a, **k: self.response(b'hello', 5)) as urlopen:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda _: te._stream_download('https://example.invalid/model', target), range(8)))
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(Path(target).read_bytes(), b'hello')

    def test_archive_cannot_write_outside_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache'
            cache.mkdir()
            def download(url, destination):
                with zipfile.ZipFile(destination, 'w') as archive:
                    archive.writestr('../outside.txt', 'escape')
            with patch.object(te, '_stream_download', side_effect=download):
                with self.assertRaises(ValueError):
                    ve._download_extract('https://example.invalid/runtime', str(cache), 'zip')
            self.assertFalse((Path(directory) / 'outside.txt').exists())
            self.assertEqual(list(cache.iterdir()), [])

    def test_failed_cuda_runtime_does_not_publish_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            def extract(url, dest, kind):
                if 'cudart' in url:
                    raise IOError('interrupted')
                Path(dest).mkdir(parents=True, exist_ok=True)
                (Path(dest) / 'llama-mtmd-cli.exe').write_bytes(b'fake exe')
            with patch.object(ve, '_bin_dir', return_value=directory), patch.object(ve.sys, 'platform', 'win32'), patch.object(ve.platform, 'machine', return_value='AMD64'), patch.object(ve, '_download_extract', side_effect=extract), patch.dict(os.environ, {'LLAMA_MTMD_CLI': ''}):
                with self.assertRaises(IOError):
                    ve._resolve_mtmd_cli(('cuda', '12.4'))
            self.assertEqual(list(Path(directory).iterdir()), [])


class VisionTests(unittest.TestCase):
    def test_cpu_fallback_disables_model_and_projector_offload(self):
        with patch.object(ve, '_resolve_mtmd_cli', return_value='/tmp/mtmd'), patch.object(ve.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='1girl', stderr='')) as run:
            ve._run_cli(None, '/m', '/p', '/image', 'tags', 25, .5, 42, 5)
        args = run.call_args.args[0]
        self.assertEqual(args[args.index('-ngl') + 1], '0')
        self.assertEqual(args[args.index('--device') + 1], 'none')
        self.assertIn('--no-mmproj-offload', args)
        self.assertEqual(args[args.index('-t') + 1], '2')
        self.assertEqual(args[args.index('-tb') + 1], '2')
        self.assertEqual(run.call_args.kwargs['stdin'], subprocess.DEVNULL)

    def test_cuda_driver_must_support_the_exact_runtime_minor(self):
        for cuda, expected in (((12, 3), None), ((12, 4), ('cuda', '12.4')), ((13, 0), ('cuda', '12.4')), ((13, 1), ('cuda', '13.1'))):
            with self.subTest(cuda=cuda), patch.object(ve.sys, 'platform', 'win32'), patch.object(ve.platform, 'machine', return_value='AMD64'), patch.object(ve, '_nvidia_max_cuda', return_value=cuda), patch.object(ve, '_vulkan_available', return_value=False), patch.dict(os.environ, {'RS_TIPO_FORCE_CPU': ''}):
                self.assertEqual(ve._detect_gpu_backend(), expected)

    def test_force_cpu_overrides_cached_gpu(self):
        with patch.object(ve, '_GPU_BACKEND', ('vulkan', None)), patch.dict(os.environ, {'RS_TIPO_FORCE_CPU': '1'}):
            self.assertIsNone(ve._gpu_backend())

    def test_custom_binary_works_even_without_platform_prebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            cli = Path(directory) / 'custom-cli'
            cli.write_bytes(b'custom')
            with patch.dict(os.environ, {'LLAMA_MTMD_CLI': str(cli)}), patch.object(ve.platform, 'machine', return_value='custom_arch'):
                self.assertEqual(ve._resolve_mtmd_cli(None), str(cli))

    def test_vision_cpu_mode_and_gpu_failure_fallback(self):
        for layers, expected in ((0, [None]), (-1, [('vulkan', None), None]), (12, [('vulkan', None), None])):
            with self.subTest(layers=layers), tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / 'model.gguf'; model.write_bytes(b'model')
                proj = Path(directory) / 'mmproj.gguf'; proj.write_bytes(b'proj')
                png = Path(directory) / 'input.png'; png.write_bytes(b'png')
                calls = []
                def infer(backend, *args):
                    calls.append(backend)
                    if backend:
                        raise RuntimeError('GPU out of memory')
                    return '1girl, blue_hair, 3girls'
                with patch.object(ve, '_save_image', return_value=str(png)), patch.object(ve, '_gpu_backend', return_value=('vulkan', None)), patch.object(ve, '_run_cli', side_effect=infer), patch.object(te, '_ensure_formatter', return_value=False):
                    tags = ve.image_to_tags(object(), str(model), str(proj), 'short', 'blue hair', .5, 0, gpu_layers=layers)
                self.assertEqual(calls, expected)
                self.assertEqual(tags, '1girl')
                self.assertFalse(png.exists())

    def test_invalid_image_never_downloads_models(self):
        with patch.object(ve, '_save_image', side_effect=ValueError('bad image')), patch.object(ve, '_resolve_vision_model') as resolve:
            self.assertEqual(ve.image_to_tags(object(), '', '', 'short', '', .5, 0), '')
            resolve.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'), 'optional image dependencies not installed')
    def test_real_png_conversion_and_batch_rejection(self):
        import numpy as np
        from PIL import Image
        for channels in (1, 3, 4):
            path = ve._save_image(np.ones((1, 2, 3, channels)))
            try:
                with Image.open(path) as image:
                    self.assertEqual(image.mode, 'RGB')
                    self.assertEqual(image.size, (3, 2))
                    self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            finally:
                os.remove(path)
        for arr in (np.zeros((2, 2, 3, 3)), np.zeros((0, 2, 3, 3)), np.full((2, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                ve._save_image(arr)


class NodeContractTests(unittest.TestCase):
    def test_api_examples_match_input_contracts(self):
        classes = {'SeasonalFashionPromptNode': seasonal.SeasonalFashionPromptNode,
                   'GemmaTipoNode': node.GemmaTipoNode,
                   'RSPromptWriter': writer.RSPromptWriter}
        examples = sorted((ROOT / 'examples').glob('*.api.json'))
        self.assertGreaterEqual(len(examples), 3)
        for path in examples:
            graph = json.loads(path.read_text())['prompt']
            for entry in graph.values():
                if entry['class_type'] == 'PreviewAny':
                    continue
                cls = classes[entry['class_type']]
                with patch.object(te, '_gguf_dir', return_value='/no-models-needed'):
                    schema = cls.INPUT_TYPES()
                self.assertTrue(set(schema['required']) <= set(entry['inputs']), path.name)
                for key, value in entry['inputs'].items():
                    if isinstance(value, list):
                        self.assertIn(value[0], graph)
                    else:
                        spec = (schema['required'] | schema.get('optional', {}))[key]
                        if isinstance(spec[0], list):
                            self.assertIn(value, spec[0])
            if graph['1']['class_type'] == 'SeasonalFashionPromptNode':
                result = seasonal.SeasonalFashionPromptNode().generate_prompt(**graph['1']['inputs'])
                self.assertTrue(result[0].startswith('spring,'))

    def test_discovery_is_zero_dependency_and_versions_agree(self):
        script = '''import importlib.util, sys
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('rs_nodes', root / '__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert set(module.NODE_CLASS_MAPPINGS) == {'SeasonalFashionPromptNode', 'GemmaTipoNode', 'RSPromptWriter'}
assert all(name not in sys.modules for name in ('torch', 'numpy', 'PIL', 'llama_cpp', 'kgen'))
assert ('version = "' + module.__version__ + '"') in (root / 'pyproject.toml').read_text()
'''
        subprocess.run([sys.executable, '-S', '-c', script], cwd=ROOT, check=True)

    def test_projectors_and_directories_are_not_language_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('model.gguf', 'model.mmproj.gguf', 'MMPROJ-F16.GGUF'):
                (Path(directory) / name).write_bytes(b'test')
            (Path(directory) / 'directory.gguf').mkdir()
            with patch.object(te, '_gguf_dir', return_value=directory):
                choices = node.GemmaTipoNode.INPUT_TYPES()['required']['model'][0]
            self.assertEqual(choices, [node.AUTO_MODEL, 'model.gguf'])

    def test_node_dispatch_and_text_gpu_setting(self):
        text, vision = Mock(), Mock()
        text.expand_prompt.return_value = 'text tags'
        vision.image_to_tags.return_value = 'image tags'
        with patch.object(node, '_engines', return_value=(text, vision)):
            generator = node.GemmaTipoNode()
            self.assertEqual(generator.run('input', node.AUTO_MODEL, 'short', 'simple', .5, '', 42, gpu_layers=8), ('text tags',))
            self.assertEqual(text.expand_prompt.call_args.args[-1], 8)
            self.assertEqual(generator.run('ignored', node.AUTO_MODEL, 'short', 'simple', .5, '', 42, image=[1]), ('image tags',))
            self.assertEqual(generator.run('', node.AUTO_MODEL, 'short', 'simple', .5, '', 42), ('',))


if __name__ == '__main__':
    unittest.main()
