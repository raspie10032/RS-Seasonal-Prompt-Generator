"""Native preflight uses the selected PyTorch backend; no NVIDIA detection."""
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vrs_int import torch_device as td


class DeviceTests(unittest.TestCase):
    def runtime(self, kind):
        device = SimpleNamespace(type=kind)
        mm = SimpleNamespace(get_torch_device=lambda: device,
                             throw_exception_if_processing_interrupted=Mock())
        return device, {'comfy': SimpleNamespace(model_management=mm), 'comfy.model_management': mm}

    def test_other_accelerators_are_probed_instead_of_rejected_by_name(self):
        for kind in ('cuda', 'mps', 'xpu', 'privateuseone'):
            device, modules = self.runtime(kind)
            with self.subTest(kind=kind), patch.dict(sys.modules, modules), patch.object(td, '_probe') as probe:
                self.assertIs(td.require_accelerator(), device)
                probe.assert_called_once_with(str(device))

    def test_cpu_and_meta_fail_without_allocating_or_downloading(self):
        for kind in ('cpu', 'meta'):
            _, modules = self.runtime(kind)
            with self.subTest(kind=kind), patch.dict(sys.modules, modules), patch.object(td, '_probe') as probe:
                with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                    td.require_accelerator()
                probe.assert_not_called()

    def test_failed_probe_propagates_without_changing_device(self):
        _, modules = self.runtime('xpu')
        with patch.dict(sys.modules, modules), patch.object(td, '_probe', side_effect=RuntimeError('unsupported')) as probe:
            with self.assertRaisesRegex(RuntimeError, 'unsupported'):
                td.require_accelerator()
            self.assertEqual(probe.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_real_basic_math_and_failure_not_cached(self):
        import torch
        td._probe.cache_clear()
        self.addCleanup(td._probe.cache_clear)
        td._probe('cpu')  # Only exercise the math; the public guard rejects CPU.
        with patch.object(torch, 'ones', side_effect=NotImplementedError('unsupported operator')) as allocate:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'No CPU retry'):
                    td._probe('meta')
            self.assertEqual(allocate.call_count, 2)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_cpu_requires_a_boolean_explicit_choice_and_does_not_probe_gpu(self):
        mm = SimpleNamespace(get_torch_device=Mock(side_effect=AssertionError('No GPU selection')),
                             throw_exception_if_processing_interrupted=Mock())
        with patch.dict(sys.modules, {'comfy': SimpleNamespace(model_management=mm), 'comfy.model_management': mm}), \
                patch.object(td, '_probe') as probe:
            self.assertEqual(td.require_accelerator(use_cpu=True).type, 'cpu')
            probe.assert_called_once_with('cpu')
            for invalid in ('true', 1, None):
                with self.assertRaisesRegex(ValueError, 'boolean'):
                    td.require_accelerator(use_cpu=invalid)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_tagger_non_cuda_scoring_never_enters_cuda_autocast(self):
        import numpy as np
        import torch
        from vrs_int.generated_tagger import LocalPixAI
        tagger = LocalPixAI.__new__(LocalPixAI)
        tagger.device = 'cpu'
        tagger.labels = ['a', 'b']
        tagger.provenance = {}
        tagger.model = lambda batch: batch
        with patch.object(torch, 'autocast', side_effect=AssertionError('No CUDA context')):
            result = tagger.score_prepared(torch.tensor([[0., 1.]]))
        np.testing.assert_allclose(result, [[.5, .7310586]], rtol=1e-6)
        self.assertEqual(tagger.provenance['precision'], 'fp32-weights-fp32-sigmoid')


if __name__ == '__main__':
    unittest.main()
