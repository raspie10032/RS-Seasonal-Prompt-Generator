"""Queue responsiveness and image-boundary handoff, without a live GPU job."""
import asyncio
import os
import sys
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import test_vrs_folders as fixtures
from vrs_folder_service import enqueue_if_idle, generation_waiting
from vrs_int.store import Store
from vrs_folder_service import VRSFolderMemory


class VolatileQueue(fixtures.Queue):
    def get_current_queue(self):
        raise AssertionError("Do not deep-copy full user workflows")

    def get_current_queue_volatile(self):
        with self.mutex:
            return self.running[:], self.pending[:]


class FolderServiceTests(unittest.TestCase):
    setUp = fixtures.FolderTests.setUp
    image = fixtures.FolderTests.image
    scan = fixtures.FolderTests.scan
    service = fixtures.FolderTests.service

    def test_queue_arbitration_uses_lightweight_view(self):
        queue = VolatileQueue()
        self.assertTrue(enqueue_if_idle(queue, (1e15, "import")))
        self.assertFalse(generation_waiting(queue, "import"))
        queue.put((1, "user"))
        self.assertTrue(generation_waiting(queue, "import"))
        self.assertFalse(enqueue_if_idle(queue, (1e15, "other")))
        self.assertEqual(len(queue.pending), 2)

    def test_user_waiting_at_start_skips_encoder_and_preserves_pending(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "import"
        service.server.prompt_queue.put((1, "user"))
        self.assertEqual(service.execute("import"), {"state": "yielded", "added": 0})
        service.encoder.encode_tensor.assert_not_called()
        self.assertEqual(self.catalog.status()["records"], {"pending": 1})
        self.assertEqual(service.server.prompt_queue.pending, [(1, "user")])
        self.assertFalse(service.stopped.is_set())

    def test_user_arriving_during_image_yields_after_its_commit(self):
        self.image()
        self.image("two.png", color="green")
        self.scan()
        service = self.service()
        service.ticket = "import"
        def encode(image):
            service.server.prompt_queue.put((1, "user"))
            return np.ones(768, dtype=np.float32)
        service.encoder.encode_tensor.side_effect = encode
        result = service.execute("import")
        self.assertEqual(result["added"], 1)
        self.assertEqual(self.catalog.status()["records"], {"staged": 1, "pending": 1})
        self.assertIsNone(result["snapshot"])
        published = self.catalog.publish()
        self.assertEqual(Store(self.catalog.memory).verify(published)["vectors"], 1)
        self.assertEqual(service.server.prompt_queue.pending, [(1, "user")])
        self.assertIsNone(service.ticket)
        self.assertFalse(service.stopped.is_set())

    def test_user_arriving_during_model_load_skips_first_image(self):
        self.image()
        self.scan()
        service = self.service()
        encoder, service.encoder = service.encoder, None
        service.ticket = "import"
        sys.modules["folder_paths"].get_full_path_or_raise = Mock(return_value="gemma.safetensors")
        def load(path):
            service.server.prompt_queue.put((1, "user"))
            return encoder
        with patch("vrs_folder_service.NativeVision", side_effect=load):
            self.assertEqual(service.execute("import")["state"], "yielded")
        encoder.encode_tensor.assert_not_called()
        self.assertEqual(self.catalog.status()["records"], {"pending": 1})

    def test_interruption_keeps_previously_encoded_image(self):
        self.image()
        self.image("two.png", color="green")
        self.scan()
        service = self.service()
        service.ticket = "import"
        service.encoder.encode_tensor.side_effect = [np.ones(768, dtype=np.float32), RuntimeError("interrupted")]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            service.execute("import")
        self.assertEqual(self.catalog.status()["records"], {"pending": 1, "staged": 1})
        snapshot = self.catalog.publish(force=True)
        self.assertEqual(Store(self.catalog.memory).verify(snapshot)["vectors"], 1)

    def test_pending_check_runs_off_event_loop_and_stop_prevents_submit(self):
        service = self.service()
        main_thread = threading.get_ident()
        worker_threads = []
        def pending():
            worker_threads.append(threading.get_ident())
            service.stopped.set()
            return True
        service.catalog.scan_steps = Mock(return_value=iter(()))
        service.catalog.has_pending = pending
        service.submit = AsyncMock()
        asyncio.run(service.run())
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], main_thread)
        service.submit.assert_not_awaited()
        self.assertEqual(service.state, "stopped")

    def test_completed_ticket_is_not_reported_as_disappeared(self):
        service = self.service()
        service.catalog.scan_steps = Mock(return_value=iter(()))
        service.ticket = "import"
        queue = VolatileQueue()
        service.server.prompt_queue = queue
        def completed():
            # Simulate execute() finishing between the ticket read and lookup.
            service.ticket = None
            service.stopped.set()
            return [], []
        queue.get_current_queue_volatile = completed
        asyncio.run(service.run())
        self.assertIsNone(service.error)
        self.assertEqual(service.state, "stopped")

    def test_append_does_not_repeat_per_entry_store_lookup(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "first"
        first = service.execute("first")["snapshot"]
        self.image("two.png", color="green")
        self.scan()
        service.ticket = "second"
        with patch.object(Store, "get", side_effect=AssertionError("Redundant lookup")):
            second = service.execute("second")["snapshot"]
        self.assertNotEqual(first, second)
        self.assertEqual(Store(self.catalog.memory).verify(second)["observations"], 2)

    def test_prepared_head_advances_only_after_success_and_preserves_pinned_readers(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "first"
        first = service.execute("first")["snapshot"]
        with patch.dict(os.environ, {"RS_VRS_CACHE_DIR": str(self.root / "cache")}), \
                patch("vrs_folder_service.get_service", return_value=service):
            service.prepare_search()
            self.assertEqual(self.catalog.status()["search_count"], 1)
            original, snapshot, _ = VRSFolderMemory().load(self.catalog.config["id"])
            self.addCleanup(original.close)
            self.assertEqual(snapshot, first)
            self.image("two.png", color="green")
            self.scan()
            service.ticket = "second"
            second = service.execute("second")["snapshot"]
            self.assertEqual(self.catalog.search_snapshot(), first)
            self.assertEqual(VRSFolderMemory.IS_CHANGED(self.catalog.config["id"]), first)
            # An arriving generation must prevent expensive search preparation.
            service.server.prompt_queue.put((1, "user"))
            with patch("vrs_int.search_index.SearchIndex", side_effect=AssertionError("generation is active")):
                service.prepare_search()
            self.assertEqual(self.catalog.search_snapshot(), first)
            service.server.prompt_queue.pending.clear()
            with patch("vrs_int.search_index.SearchIndex", side_effect=OSError("disk full")):
                service.prepare_search()
            self.assertEqual(self.catalog.search_snapshot(), first)
            self.assertIn("disk full", service.status()["search_error"])
            self.assertFalse(service.stopped.is_set())
            service.search_retry_at = 0
            service.prepare_search()
            self.assertEqual(self.catalog.search_snapshot(), second)
            self.assertEqual(self.catalog.status()["search_count"], 2)
            self.assertEqual(original.snapshot, first)
            self.assertFalse(self.catalog.search_ready(first))
            self.assertEqual(self.catalog.search_snapshot(), second)

    def test_stop_during_preparation_keeps_previous_ready_head(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "first"
        first = service.execute("first")["snapshot"]
        with patch.dict(os.environ, {"RS_VRS_CACHE_DIR": str(self.root / "cache")}):
            service.prepare_search()
            self.image("two.png", color="green")
            self.scan()
            service.ticket = "second"
            service.execute("second")
            def stopped(store, snapshot, checkpoint):
                service.stopped.set()
                checkpoint()
            with patch("vrs_int.search_index.SearchIndex", side_effect=stopped):
                service.prepare_search()
            self.assertEqual(self.catalog.search_snapshot(), first)
            self.assertIsNone(service.search_error)


if __name__ == "__main__":
    unittest.main()
