"""Persistent exact search: fresh processes, immutable views, recovery and limits."""
import copy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zlib
from unittest.mock import patch

import numpy as np

from test_vrs_search_index import fixture, eligible_reference
from vrs_int.search_index import SearchIndex
from vrs_int.search_cache import file_hash, tree_bytes
from vrs_int.store import Store, revision


class PersistentSearchTests(unittest.TestCase):
    def setUp(self):
        disk = patch("vrs_int.search_cache.shutil.disk_usage",
                     return_value=SimpleNamespace(total=100 * 1024 ** 3, free=80 * 1024 ** 3))
        disk.start()
        self.addCleanup(disk.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        env = patch.dict(os.environ, {"RS_VRS_CACHE_DIR": str(self.cache), "RS_VRS_CACHE_MAX_GIB": "8"})
        env.start()
        self.addCleanup(env.stop)
        self.store = Store(self.root / "source")
        self.contract, self.entries = fixture()
        self.snapshot = self.store.build(self.entries, self.contract, "test", bucket_count=7)

    def index(self, snapshot=None):
        value = SearchIndex(self.store, snapshot or self.snapshot)
        self.addCleanup(value.close)
        return value

    def added(self, n=1):
        entries = []
        for i in range(n):
            entry = copy.deepcopy(self.entries[0])
            entry["document"]["id"] = i + 1000
            entry["key"] = f"danbooru:{i + 1000}"
            entry["document"]["image_sha256"] = hashlib.sha256(entry["key"].encode()).hexdigest()
            entries.append(entry)
        return self.store.update(self.snapshot, [dict(op="upsert", key=e["key"], entry=e,
                                                       expected_revision=None) for e in entries], "add")

    def test_reopen_does_not_decode_source_or_reinterpret(self):
        old = self.index()
        path = old.path
        docs = old.documents(range(old.count))
        old.close()
        with patch.object(Store, "read_bucket", side_effect=AssertionError("unpack")), \
                patch("vrs_int.search_cache.convert", side_effect=AssertionError("convert")):
            new = self.index()
        self.assertEqual(path, new.path)
        self.assertEqual(new.stats["converted_entries"], 0)
        self.assertEqual(new.stats["reused_objects"], 7)
        self.assertEqual(docs, new.documents(range(new.count)))

    def test_compressed_documents_preserve_duplicates_empty_and_chunked_order(self):
        index = self.index()
        expected = index.documents(range(index.count))
        with closing(sqlite3.connect(index.path / "votes.sqlite3")) as db:
            raw, = db.execute("SELECT value FROM records LIMIT 1").fetchone()
        self.assertEqual(json.loads(zlib.decompress(raw))["key"], expected[0]["key"])
        order = [int(i % index.count) for i in range(700, -1, -1)]
        self.assertEqual(index.documents(order), [expected[i] for i in order])
        self.assertEqual(index.documents([]), [])

    def test_compact_store_yields_identical_search_documents_and_vector_bits(self):
        old = self.index()
        report = self.store.repack(self.snapshot, self.root / "compact")
        compact = SearchIndex(Store(report["store"]), report["snapshot"])
        self.addCleanup(compact.close)
        self.assertEqual(old.documents(range(old.count)), compact.documents(range(compact.count)))
        np.testing.assert_array_equal(old.vectors[:].view("u1"), compact.vectors[:].view("u1"))
        self.assertEqual(compact.stats["converted_entries"], 0)

    def test_large_single_bucket_batches_revisions_and_winning_documents(self):
        rows = []
        for number in range(1100):
            row = copy.deepcopy(self.entries[0])
            row["key"] = f"danbooru:{number + 1000}"
            row["document"]["id"] = number + 1000
            row["document"]["image_sha256"] = hashlib.sha256(row["key"].encode()).hexdigest()
            rows.append(row)
        snapshot = self.store.build(rows, self.contract, "batched", bucket_count=1)
        index = self.index(snapshot)
        self.assertEqual(index.count, 1100)
        self.assertEqual(index.stats["converted_entries"], 1100)
        expected = index.documents(range(index.count))
        order = list(range(1099, -1, -1)) + [42, 42, 0]
        self.assertEqual(index.documents(order), [expected[i] for i in order])
        changed = copy.deepcopy(rows[42])
        changed["document"]["tags"].append("new_scene")
        updated = self.store.update(snapshot, [dict(op="upsert", key=changed["key"],
            entry=changed, expected_revision=revision(rows[42]))], "one edit")
        new = self.index(updated)
        self.assertEqual(new.stats["converted_entries"], 1)
        self.assertEqual(new.stats["reused_entries"], 1099)
        self.assertEqual(index.documents(order), [expected[i] for i in order])

    def test_fresh_process_reuses_without_conversion(self):
        self.index().close()
        code = """
import json,sys
from types import SimpleNamespace
from unittest.mock import patch
from vrs_int.search_index import SearchIndex
from vrs_int.store import Store
with patch.object(Store, 'read_bucket', side_effect=AssertionError('unpack')), patch(
    'vrs_int.search_cache.shutil.disk_usage',
    return_value=SimpleNamespace(total=100 * 1024 ** 3, free=80 * 1024 ** 3)):
    index=SearchIndex(Store(sys.argv[1]), sys.argv[2])
print(json.dumps(index.stats)); index.close()
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.store.root), self.snapshot],
                                capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(json.loads(result.stdout)["converted_entries"], 0)

    def test_one_four_and_128_additions_only_convert_new_rows(self):
        old = self.index()
        before = old.vectors[np.arange(old.count)].copy()
        docs = old.documents(range(old.count))
        for n in (1, 4, 128):
            new = self.index(self.added(n))
            self.assertEqual(new.path, old.path)
            self.assertEqual(new.stats["converted_entries"], {1: 1, 4: 3, 128: 124}[n])
            self.assertEqual(new.count, old.count + n)
            self.assertEqual((old.path / "vectors.f32").stat().st_size, (old.count + n) * 8 * 4)
            np.testing.assert_array_equal(old.vectors[np.arange(old.count)], before)
            self.assertEqual(old.documents(range(old.count)), docs)

    def test_modified_deleted_and_nontraining_rows_are_snapshot_local(self):
        old = self.index()
        edited = copy.deepcopy(self.entries[0])
        edited["document"]["tags"] = ["green hair"]
        edited["categories"] = {"green hair": ["general"]}
        val = copy.deepcopy(self.entries[1])
        val["document"]["split"] = "val"
        operations = [dict(op="upsert", key=e["key"], entry=e, expected_revision=revision(self.entries[i]))
                      for i, e in enumerate((edited, val))]
        operations.append(dict(op="delete", key=self.entries[2]["key"], reason="test",
                               expected_revision=revision(self.entries[2])))
        snapshot = self.store.update(self.snapshot, operations, "modify")
        new = self.index(snapshot)
        docs = {d["key"]: d for d in new.documents(range(new.count))}
        self.assertEqual(docs[edited["key"]]["document"]["tags"], ["green hair"])
        self.assertNotIn(val["key"], docs)
        self.assertNotIn(self.entries[2]["key"], docs)
        self.assertEqual(new.stats["converted_entries"], 1)
        self.assertNotEqual(old.documents(range(old.count)), new.documents(range(new.count)))
        expected = [e["key"] for e in eligible_reference(self.store, snapshot)]
        self.assertEqual([d["key"] for d in new.documents(new.eligible(None, None, ()))], expected)

    def test_source_damage_is_rejected_even_with_complete_cache(self):
        old = self.index()
        source = self.store.object_path(next(iter(self.store.manifest(self.snapshot)["objects"].values())))
        source.write_bytes(source.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.index()
        self.assertTrue(old.path.exists())

    def test_corrupt_vector_or_vote_cache_rebuilds_without_overwriting_reader(self):
        for name in ("vectors.f32", "votes.sqlite3", "seal.json"):
            old = self.index()
            path = old.path / name
            raw = path.read_bytes()
            path.write_bytes(b"X" + raw[1:])
            new = self.index()
            self.assertNotEqual(new.path, old.path)
            self.assertEqual(new.count, old.count)
            self.assertGreater(new.stats["converted_entries"], 0)
            self.assertTrue(old.path.exists())

    def test_cancelled_delta_rolls_back_and_old_cache_remains_usable(self):
        old = self.index()
        original_hash = file_hash(old.path / "vectors.f32")
        snapshot = self.added(4)
        from vrs_int.search_cache import convert
        calls = 0
        def cancel(entry, dimension):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("cancelled delta")
            return convert(entry, dimension)
        with patch("vrs_int.search_cache.convert", side_effect=cancel):
            with self.assertRaisesRegex(RuntimeError, "cancelled delta"):
                self.index(snapshot)
        self.assertEqual(file_hash(old.path / "vectors.f32"), original_hash)
        with patch.object(Store, "read_bucket", side_effect=AssertionError("lost previous cache")):
            reopened = self.index()
        self.assertEqual(old.path, reopened.path)
        self.assertEqual(self.index(snapshot).stats["converted_entries"], 4)

    def test_interrupted_seal_does_not_get_accepted(self):
        old = self.index()
        with patch("vrs_int.search_cache.seal", side_effect=OSError("crash at seal")):
            with self.assertRaisesRegex(OSError, "crash at seal"):
                self.index(self.added())
        new = self.index()
        self.assertNotEqual(old.path, new.path)
        self.assertEqual(old.documents(range(old.count)), new.documents(range(new.count)))

    def test_changed_rules_rebuild(self):
        old = self.index()
        with patch("vrs_int.search_cache.rules_digest", return_value="new parser version"):
            new = self.index()
        self.assertNotEqual(old.path, new.path)
        self.assertEqual(old.documents(range(old.count)), new.documents(range(new.count)))

    def test_quota_never_deletes_active_reader_and_reclaims_closed_generation(self):
        old = self.index()
        cap = (tree_bytes(old.path) + 8000) / 1024 ** 3
        env = {"RS_VRS_CACHE_MAX_GIB": str(cap)}
        with patch.dict(os.environ, env), patch("vrs_int.search_cache.rules_digest", return_value="other"):
            with self.assertRaisesRegex(OSError, "RS_VRS_CACHE_MAX_GIB"):
                self.index()
            self.assertTrue(old.path.exists())
            old.close()
            new = self.index()
        self.assertFalse(old.path.exists())
        self.assertGreater(new.count, 0)

    def test_disk_reserve_rejects_low_space_without_losing_active_cache(self):
        old = self.index()
        docs = old.documents(range(old.count))
        with patch("vrs_int.search_cache.shutil.disk_usage",
                   return_value=SimpleNamespace(total=100 * 1024 ** 3, free=19 * 1024 ** 3)):
            with self.assertRaisesRegex(OSError, "20% free disk reserve"):
                self.index()
        self.assertTrue(old.path.exists())
        self.assertEqual(old.documents(range(old.count)), docs)
        self.assertEqual(self.index().path, old.path)

    def test_two_processes_serialize_preparation_and_share_one_pool(self):
        code = """
import json,sys
from types import SimpleNamespace
from unittest.mock import patch
from vrs_int.search_index import SearchIndex
from vrs_int.store import Store
with patch('vrs_int.search_cache.shutil.disk_usage',
           return_value=SimpleNamespace(total=100 * 1024 ** 3, free=80 * 1024 ** 3)):
    index=SearchIndex(Store(sys.argv[1]), sys.argv[2])
print(json.dumps(dict(path=str(index.path), stats=index.stats))); index.close()
"""
        command = [sys.executable, "-c", code, str(self.store.root), self.snapshot]
        workers = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        results = []
        try:
            for worker in workers:
                out, err = worker.communicate(timeout=30)
                self.assertEqual(worker.returncode, 0, err)
                results.append(json.loads(out))
        finally:
            # Release file locks before removing the fixture even on failure.
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate()
        self.assertEqual(results[0]["path"], results[1]["path"])
        self.assertEqual(sum(r["stats"]["converted_entries"] for r in results), self.index().count)
        self.assertEqual(sorted(r["stats"]["built_objects"] for r in results), [0, 7])


if __name__ == "__main__":
    unittest.main()
