"""Progress must describe outcomes without broadcasting private input text."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tipo_status import node_status


class StatusTests(unittest.TestCase):
    def events(self, value=None, failure=None):
        send = Mock()
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(send_sync=send)))

        @node_status
        def run(prompt, unique_id=None):
            if failure:
                raise failure
            return value

        with patch.dict(sys.modules, {"server": server}):
            if failure:
                with self.assertRaises(type(failure)) as caught:
                    run("PRIVATE PROMPT", unique_id="42")
                self.assertIs(caught.exception, failure)
            else:
                self.assertIs(run("PRIVATE PROMPT", unique_id="42"), value)
        return [call.args for call in send.call_args_list]

    def test_empty_and_success_are_distinct(self):
        for result, stage in [(('tags', '', 'raw'), 'complete'), (('', '', ''), 'empty')]:
            self.assertEqual(self.events(result), [
                ('rs_tipo_status', {'node': '42', 'stage': 'validating'}),
                ('rs_tipo_status', {'node': '42', 'stage': stage}),
            ])

    def test_failure_and_cancellation_preserve_exception_without_private_details(self):
        for error in [ValueError('PRIVATE ERROR'), KeyboardInterrupt('PRIVATE ERROR')]:
            self.assertEqual(self.events(failure=error), [
                ('rs_tipo_status', {'node': '42', 'stage': 'validating'}),
                ('rs_tipo_status', {'node': '42', 'stage': 'failed'}),
            ])

    def test_no_ui_server_is_supported(self):
        @node_status
        def run(unique_id=None):
            return ('local',)
        with patch.dict(sys.modules, {'server': None}):
            self.assertEqual(run(unique_id='42'), ('local',))

    def test_broken_ui_transport_cannot_fail_inference_or_mask_original_error(self):
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(
            send_sync=Mock(side_effect=RuntimeError("PRIVATE TRANSPORT ERROR")))))
        @node_status
        def run(unique_id=None, failure=None):
            if failure:
                raise failure
            return ('local',)
        with patch.dict(sys.modules, {'server': server}), self.assertLogs(level='WARNING') as log:
            self.assertEqual(run(unique_id='42'), ('local',))
            error = ValueError('original failure')
            with self.assertRaises(ValueError) as caught:
                run(unique_id='42', failure=error)
            self.assertIs(caught.exception, error)
        self.assertNotIn('PRIVATE', ' '.join(log.output))
