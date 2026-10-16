"""Durable batching, restart, duplicate scans, and SQLite thread ownership."""
from concurrent.futures import ThreadPoolExecutor
import shutil
import sqlite3
import time
import unittest
from unittest.mock import patch

import test_vrs_folders as fixtures
from vrs_int.folders import FolderCatalog
from vrs_int.generated import entry_from_record
from vrs_int.store import Store
from vrs_int.vectors import encode_vector
import numpy as np


class StagingTests(unittest.TestCase):
    setUp = fixtures.FolderTests.setUp
    image = fixtures.FolderTests.image
    scan = fixtures.FolderTests.scan
    service = fixtures.FolderTests.service

    def stage_pending(self, count):
        rows = self.catalog.pending(count)
        contract = self.service().encoder.contract()
        vector = encode_vector(np.ones(768, dtype=np.float32), contract["codec"])
        self.catalog.stage([entry_from_record(r, vector) for _, _, r in rows],
                           [sha for sha, _, _ in rows], contract)

    def test_scan_connections_bounded_and_resume_on_other_thread(self):
        for i in range(65):
            self.image(f"{i:03}.png", prompt=f"1girl, blue hair, seed {i}")
        scan = self.catalog.scan_steps(settle_seconds=0)
        with patch("sqlite3.connect", wraps=sqlite3.connect) as connect:
            # A generator yield must not retain a connection bound to thread 1.
            with ThreadPoolExecutor(1) as first:
                first.submit(next, scan).result()
                with ThreadPoolExecutor(1) as second:
                    self.assertEqual(len(second.submit(list, scan).result()), 64)
            self.assertEqual(connect.call_count, 3)
        self.assertEqual(self.catalog.status()["records"], {"pending": 65})

    def test_batched_publication_and_history_preserved(self):
        for i in range(132):
            self.image(f"{i:03}.png", prompt=f"1girl, blue hair, seed {i}")
        self.scan()
        self.stage_pending(4)
        first = self.catalog.publish()
        self.assertEqual(Store(self.catalog.memory).verify(first)["vectors"], 4)
        for _ in range(31):
            self.stage_pending(4)
            self.assertIsNone(self.catalog.publish())
        self.stage_pending(4)
        last = self.catalog.publish()
        self.assertEqual(Store(self.catalog.memory).verify(last)["vectors"], 132)
        self.assertEqual(Store(self.catalog.memory).verify(first)["vectors"], 4)
        self.assertEqual(len(list((self.catalog.memory / "snapshots").glob("*.json"))), 2)
        self.assertEqual(self.catalog.status()["records"], {"encoded": 132})

    def test_staged_duplicate_stays_staged_and_recovers_without_source(self):
        source = self.image()
        self.scan()
        self.stage_pending(1)
        shutil.copy2(source, self.images / "duplicate.png")
        self.scan()
        self.assertEqual(self.catalog.status()["records"], {"staged": 1})
        source.unlink()
        (self.images / "duplicate.png").unlink()
        reopened = FolderCatalog(self.catalog.directory)
        self.assertFalse(reopened.has_pending())
        result = reopened.publish()
        self.assertEqual(Store(reopened.memory).verify(result)["vectors"], 1)

    def test_publication_time_limit_and_force_drain(self):
        self.image()
        self.scan()
        self.stage_pending(1)
        self.catalog.publish()
        self.image("second.png", color="green")
        self.scan()
        self.stage_pending(1)
        self.assertIsNone(self.catalog.publish())
        with self.catalog.db() as db:
            db.execute("UPDATE staged SET created=?", (time.time() - 61,))
        self.assertIsNotNone(self.catalog.publish())
        self.image("third.png", color="red")
        self.scan()
        self.stage_pending(1)
        self.assertIsNotNone(self.catalog.publish(force=True))

    def test_staged_encoder_contract_cannot_be_changed(self):
        self.image()
        self.scan()
        self.stage_pending(1)
        self.image("second.png", color="green")
        self.scan()
        sha, _, row = self.catalog.pending(1)[0]
        contract = self.service().encoder.contract()
        contract["encoder"]["provenance_sha256"] = "b" * 64
        entry = entry_from_record(row, encode_vector(np.ones(768, dtype=np.float32), "f32le"))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.catalog.stage([entry], [sha], contract)
        self.assertEqual(self.catalog.status()["records"], {"pending": 1, "staged": 1})


if __name__ == "__main__":
    unittest.main()
