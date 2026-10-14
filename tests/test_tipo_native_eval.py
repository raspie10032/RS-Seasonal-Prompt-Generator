"""Native evaluation must keep private inputs local and record both outputs."""
import unittest
from unittest.mock import patch

from experiments.run_tipo_native_eval import extract_output, graph_for, idle, local_url


class NativeEvaluationTests(unittest.TestCase):
    def test_remote_and_credential_urls_rejected(self):
        self.assertEqual(local_url('http://127.0.0.1:8188'), 'http://127.0.0.1:8188/')
        for url in ('https://example.com', 'http://127.0.0.1@evil.test',
                    'http://localhost/path', 'http://localhost/?token=secret'):
            with self.assertRaises(ValueError):
                local_url(url)

    def test_busy_queue_stops_before_submission(self):
        for queue in ({'queue_running': [1], 'queue_pending': []},
                      {'queue_running': [], 'queue_pending': [1]}):
            with patch('experiments.run_tipo_native_eval.api', return_value=queue):
                with self.assertRaisesRegex(RuntimeError, 'occupied'):
                    idle('http://localhost/')

    def test_image_must_not_be_downgraded(self):
        row = {'input': {'prompt': 'winter', 'image': '/private/image.png'}}
        with self.assertRaisesRegex(ValueError, 'staged'):
            graph_for(row, 'model.safetensors', {})
        graph = graph_for(row, 'model.safetensors', {}, 'staged/test.png')
        self.assertEqual(graph['2']['inputs']['image'], ['5', 0])
        self.assertEqual(graph['3']['inputs']['source'], ['2', 0])
        self.assertEqual(graph['4']['inputs']['source'], ['2', 1])

    def test_raw_preserved_and_termination_unknown(self):
        history = {'status': {'status_str': 'success', 'completed': True},
                   'outputs': {'3': {'text': ['winter']}, '4': {'text': ['winter, scarf']}}}
        result = extract_output(history)
        self.assertEqual(result['raw'], 'winter, scarf')
        self.assertEqual(result['postprocessed'], 'winter')
        self.assertIsNone(result['truncated'])
        history['status']['completed'] = False
        with self.assertRaises(RuntimeError):
            extract_output(history)


if __name__ == '__main__':
    unittest.main()
