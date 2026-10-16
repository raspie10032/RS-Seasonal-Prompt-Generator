"""Folder preservation, resumption, queue arbitration and snapshot binding."""
import asyncio
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, PngImagePlugin

from vrs_int.folders import FolderCatalog, collection_path
from vrs_int.import_snapshot import file_digest
from vrs_int.store import Store, canonical
from vrs_folder_service import FolderService, VRSFolderMemory, enqueue_if_idle


class Queue:
    def __init__(self):
        self.mutex = threading.RLock()
        self.running, self.pending = [], []

    def get_current_queue(self):
        with self.mutex:
            return self.running[:], self.pending[:]

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.pending, item)

    def delete_queue_item(self, predicate):
        with self.mutex:
            for item in self.pending:
                if predicate(item):
                    self.pending.remove(item)
                    heapq.heapify(self.pending)
                    return True
        return False


class FolderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.catalog = FolderCatalog.create(self.root / "collections", [self.images], "gemma.safetensors")

    def image(self, name="one.png", prompt="1girl, blue hair", negative="bad anatomy", color="blue"):
        path = self.images / name
        info = PngImagePlugin.PngInfo()
        if prompt is not None:
            info.add_text("Comment", json.dumps({"prompt": prompt, "uc": negative}))
        Image.new("RGB", (32, 24), color).save(path, pnginfo=info)
        os.utime(path, (time.time() - 60, time.time() - 60))
        return path

    def scan(self):
        return list(self.catalog.scan_steps(settle_seconds=0))

    def service(self):
        queue = Queue()
        service = FolderService(SimpleNamespace(prompt_queue=queue), self.root / "collections")
        service.catalog = self.catalog
        service.stopped.clear()
        service.encoder_name = "gemma.safetensors"
        encoder = Mock()
        encoder.contract.return_value = {"encoder": {"provenance_sha256": "a" * 64,
            "preprocess": "test-native", "backend": "comfyui-pytorch-fp32-compute",
            "quantization": "fp16", "pooling": "mean-tokens-l2-v1"},
            "dimension": 768, "codec": "f32le", "tag_policy": "rs-core-identities-v1"}
        encoder.encode_tensor.return_value = np.ones(768, dtype=np.float32)
        service.encoder = encoder
        mm = SimpleNamespace(throw_exception_if_processing_interrupted=Mock())
        context = patch.dict(sys.modules, {"folder_paths": SimpleNamespace(),
            "comfy": SimpleNamespace(model_management=mm), "comfy.model_management": mm})
        context.start()
        self.addCleanup(context.stop)
        # These tests exercise the queue/catalog, not ComfyUI tensor conversion.
        # Keep them runnable in CI with only NumPy/Pillow. The actual IMAGE
        # conversion and pixel binding are covered by test_vrs_native.
        def pixels(path):
            with Image.open(path) as source:
                return np.array(source.convert("RGB"), dtype=np.uint8)[None]
        def pixel_hash(image):
            return hashlib.sha256(canonical(list(image.shape)) + image.tobytes()).hexdigest()
        for boundary, implementation in (("load_image", pixels), ("pixel_digest", pixel_hash)):
            patcher = patch("vrs_folder_service." + boundary, side_effect=implementation)
            patcher.start()
            self.addCleanup(patcher.stop)
        return service

    def test_metadata_duplicates_no_metadata_and_read_only_scan(self):
        original = self.image()
        before = file_digest(original)
        shutil.copy2(original, self.images / "copy.png")
        self.image("plain.png", prompt=None)
        self.scan()
        status = self.catalog.status()
        self.assertEqual(status["records"], {"pending": 1})
        self.assertEqual(status["files"]["duplicate-bytes"], 1)
        self.assertEqual(status["files"]["no-generation-metadata"], 1)
        row = self.catalog.pending(1)[0][2]
        self.assertEqual(row["generation"]["negative"], ["bad anatomy"])
        self.assertIn("blue hair", row["caption"])
        self.assertEqual(file_digest(original), before)
        with patch("vrs_int.folders._scan_image", side_effect=AssertionError("unchanged reread")):
            self.scan()

    def test_recent_files_wait(self):
        path = self.image()
        os.utime(path, None)
        self.assertEqual(list(self.catalog.scan_steps(settle_seconds=60)), [None])

    def test_symlinks_not_followed(self):
        path = self.image()
        os.utime(path, (1, 1))
        try:
            (self.images / "loop").symlink_to(self.images, target_is_directory=True)
            (self.images / "alias.png").symlink_to(path)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege is unavailable")
            raise
        self.scan()
        self.assertEqual(self.catalog.status()["files"], {"included": 1})

    def test_changes_are_new_experience_and_missing_original_is_preserved(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "first"
        first = service.execute("first")["snapshot"]
        self.image(prompt="1girl, green hair", color="green")
        self.scan()
        service.ticket = "second"
        second = service.execute("second")["snapshot"]
        self.assertNotEqual(first, second)
        self.assertEqual(Store(self.catalog.memory).verify(first)["observations"], 1)
        self.assertEqual(Store(self.catalog.memory).verify(second)["observations"], 2)
        (self.images / "one.png").unlink()
        self.scan()
        reopened = FolderCatalog(self.catalog.directory)
        self.assertEqual(reopened.snapshot(), second)
        self.assertEqual(reopened.status()["records"], {"encoded": 2})

    def test_failed_publish_does_not_mark_records_complete(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "first"
        with patch.object(self.catalog, "commit", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                service.execute("first")
        self.assertIsNone(self.catalog.snapshot())
        self.assertEqual(self.catalog.status()["records"], {"staged": 1})
        self.assertTrue(service.stopped.is_set())
        self.assertIsNone(service.ticket)
        reopened = FolderCatalog(self.catalog.directory)
        self.assertFalse(reopened.has_pending())
        snapshot = reopened.publish(force=True)
        self.assertEqual(Store(reopened.memory).verify(snapshot)["vectors"], 1)
        self.assertEqual(reopened.status()["records"], {"encoded": 1})

    def test_stop_during_encoding_commits_only_completed_image(self):
        self.image()
        self.image("two.png", color="green")
        self.scan()
        service = self.service()
        def encode(image):
            service.stop()
            return np.ones(768, dtype=np.float32)
        service.encoder.encode_tensor.side_effect = encode
        service.ticket = "first"
        result = service.execute("first")
        self.assertEqual(result["added"], 1)
        self.assertEqual(self.catalog.status()["records"], {"encoded": 1, "pending": 1})
        self.assertEqual(Store(self.catalog.memory).verify(result["snapshot"])["vectors"], 1)

    def test_stale_ticket_cannot_start_or_clear_work(self):
        service = self.service()
        service.ticket = "current"
        with self.assertRaisesRegex(ValueError, "Expired"):
            service.execute("old")
        self.assertEqual(service.ticket, "current")
        service.encoder.encode_tensor.assert_not_called()

    def test_stop_removes_only_import_never_generation(self):
        service = self.service()
        service.ticket = "import"
        queue = service.server.prompt_queue
        queue.put((1e15, "import"))
        queue.put((1, "user"))
        service.stop()
        self.assertEqual(queue.pending, [(1, "user")])
        self.assertIsNone(service.ticket)

    def test_encoder_contract_change_rejected_without_advancing(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = "first"
        first = service.execute("first")["snapshot"]
        self.image("two.png", color="green")
        self.scan()
        service.encoder.contract.return_value["encoder"]["provenance_sha256"] = "b" * 64
        service.ticket = "second"
        with self.assertRaisesRegex(ValueError, "changed"):
            service.execute("second")
        self.assertEqual(self.catalog.snapshot(), first)

    def test_memory_loader_follows_completed_snapshot_only(self):
        self.image()
        self.scan()
        service = self.service()
        with patch("vrs_folder_service.get_service", return_value=service):
            cid = self.catalog.config["id"]
            with self.assertRaisesRegex(ValueError, "No completed"):
                VRSFolderMemory().load(cid)
            service.ticket = "first"
            result = service.execute("first")
            memory, snapshot, report = VRSFolderMemory().load(cid)
            self.assertEqual(memory.snapshot, result["snapshot"])
            self.assertEqual(snapshot, VRSFolderMemory.IS_CHANGED(cid))

    def test_invalid_and_nested_roots(self):
        nested = self.images / "output"
        nested.mkdir()
        catalog = FolderCatalog.create(self.root / "collections", [self.images, nested], "gemma.safetensors")
        self.assertEqual(catalog.config["roots"], [str(self.images.resolve())])
        with self.assertRaises(ValueError):
            collection_path(self.root, "../escape")
        with self.assertRaises(ValueError):
            FolderCatalog.create(self.root, [], "gemma.safetensors")

    def test_idle_check_does_not_append_to_busy_generation_queue(self):
        queue = Queue()
        queue.running.append((1, "user"))
        self.assertFalse(enqueue_if_idle(queue, (1e15, "import")))
        queue.running.clear()
        queue.put((1, "user"))
        self.assertFalse(enqueue_if_idle(queue, (1e15, "import")))
        queue.pending.clear()
        self.assertTrue(enqueue_if_idle(queue, (1e15, "import")))
        queue.put((2, "later-user"))
        self.assertEqual(heapq.heappop(queue.pending)[1], "later-user")

    def test_missing_queue_handle_pauses_instead_of_resubmitting(self):
        service = self.service()
        service.ticket = "removed"
        asyncio.run(service.run())
        self.assertTrue(service.stopped.is_set())
        self.assertIn("disappeared", service.error)


if __name__ == "__main__":
    unittest.main()
