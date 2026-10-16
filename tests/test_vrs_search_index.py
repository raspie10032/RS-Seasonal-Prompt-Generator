"""Exact retrieval semantics, bounded cache ownership and failure recovery."""
import gc
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vrs_int.recommend import Recommender
from vrs_int.search_index import MappedVectors, SearchIndex, best_in_block
from vrs_int.store import Store, revision
from vrs_int.vectors import decode_vector, encode_vector

try:
    import torch
except ImportError:
    torch = None


def fixture(codec="f32le"):
    contract = {"encoder": {"provenance_sha256": "a" * 64, "preprocess": "fixture", "backend": "fixture",
                            "quantization": "none", "pooling": "fixture"},
                "dimension": 8, "codec": codec, "tag_policy": "fixture"}
    rng = np.random.default_rng(81)
    entries = []
    for i in range(1, 86):
        sha = hashlib.sha256(str(i // 3).encode()).hexdigest()  # Cross-rating copies.
        vector = rng.normal(size=8) if i < 40 else np.ones(8)  # Many cutoff ties.
        entries.append({"key": f"danbooru:{i}", "document": {
            "id": i, "image_sha256": sha, "split": "val" if i % 11 == 0 else "train",
            "rating": "g" if i % 2 else "s", "tags": ["red hair", "sample artist", f"scene {i}", "highres"]},
            "categories": {"sample artist": ["artist"], "red hair": ["general"]},
            "vector": encode_vector(vector, codec) if i % 13 else None,
            "audit": [{"native_pixel_sha256": "a" * 64 if i % 5 else "b" * 64}]})
    return contract, entries


def eligible_reference(store, snapshot, image=None, pixel=None, ratings=()):
    seen, result = set(), []
    for entry in store.entries(snapshot):
        doc = entry["document"]
        if entry["vector"] is None or doc["split"] != "train" or doc["image_sha256"] == image:
            continue
        pixels = [a["native_pixel_sha256"] for a in entry["audit"] if a.get("native_pixel_sha256")]
        if pixel and pixels and pixels[-1] == pixel:
            continue
        if ratings and doc["rating"] not in ratings:
            continue
        if doc["image_sha256"] not in seen:
            result.append(entry)
            seen.add(doc["image_sha256"])
    return result


class SearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.env = patch.dict("os.environ", {"RS_VRS_CACHE_DIR": str(self.cache)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store = Store(self.root / "memory")
        self.contract, self.entries = fixture()
        self.snapshot = self.store.build(self.entries, self.contract, "test", bucket_count=7)

    def index(self):
        index = SearchIndex(self.store, self.snapshot)
        self.addCleanup(index.close)
        return index

    def test_filters_deduplicate_after_rating_and_pixel_exclusions(self):
        index = self.index()
        for ratings in ((), ("g",), ("s",), ("u",), ("g", "s")):
            for pixel in (None, "a" * 64, "b" * 64, "a" * 65, "한" * 64):
                for image in (None, self.entries[0]["document"]["image_sha256"]):
                    with self.subTest(ratings=ratings, pixel=pixel, image=image):
                        expected = eligible_reference(self.store, self.snapshot, image, pixel, ratings)
                        rows = index.eligible(image, pixel, ratings)
                        self.assertEqual([v["key"] for v in index.documents(rows)], [e["key"] for e in expected])
                        for row, entry in zip(rows, expected):
                            np.testing.assert_array_equal(index.vectors[row], decode_vector(entry["vector"], 8))

    def test_only_winner_documents_read_and_sources_unchanged(self):
        before = {str(p): p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()}
        index = self.index()
        with patch.object(Store, "read_bucket", side_effect=AssertionError("Should not reread sources")):
            winners = index.documents(index.eligible(None, None, ())[:3])
        self.assertEqual(len(winners), 3)
        self.assertNotIn("vector", winners[0])
        self.assertNotIn("audit", winners[0])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()})

    def test_long_valid_observation_keys_keep_exact_tie_order(self):
        contract, entries = fixture()
        for i, entry in enumerate(entries):
            entry["document"]["id"] = int("9" * 100 + str(i + 1))
            entry["key"] = "danbooru:" + str(entry["document"]["id"])
        snapshot = self.store.build(entries, contract, "long-keys", bucket_count=7)
        index = SearchIndex(self.store, snapshot)
        self.addCleanup(index.close)
        documents = index.documents(range(index.count))
        order = np.argsort(index.meta["order"])
        self.assertEqual([documents[i]["key"] for i in order], sorted(d["key"] for d in documents))

    def test_cancel_preparation_cleans_partial_files(self):
        calls = 0
        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 12:
                raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            SearchIndex(self.store, self.snapshot, interrupt)
        self.assertEqual(list(self.cache.rglob("generation-*")), [])
        self.assertGreater(self.index().count, 0)

    def test_corruption_never_exposes_index(self):
        manifest = self.store.manifest(self.snapshot)
        path = self.store.object_path(manifest["objects"]["6"])
        path.write_bytes(path.read_bytes() + b"broken")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            SearchIndex(self.store, self.snapshot)
        self.assertEqual(list(self.cache.rglob("generation-*")), [])

    def test_failed_disk_write_cleans_and_can_retry(self):
        with patch("vrs_int.search_cache.convert", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                SearchIndex(self.store, self.snapshot)
        self.assertEqual(list(self.cache.rglob("generation-*")), [])
        self.assertGreater(self.index().count, 0)

    def test_close_and_garbage_collection_release_leases_keep_cache(self):
        index = self.index()
        path = index.path
        index.close()
        index.close()
        self.assertTrue(path.exists())
        self.assertEqual(list(path.glob("lease-*")), [])
        other = SearchIndex(self.store, self.snapshot)
        path = other.path
        del other
        gc.collect()
        self.assertTrue(path.exists())
        self.assertEqual(list(path.glob("lease-*")), [])

    def test_no_vectors_and_new_snapshot_do_not_reuse_stale_rows(self):
        old = self.index()
        changes = [{"op": "delete", "key": e["key"], "expected_revision": revision(e), "reason": "test"}
                   for e in self.entries]
        snapshot = self.store.update(self.snapshot, changes, "deleted")
        new = SearchIndex(self.store, snapshot)
        self.addCleanup(new.close)
        self.assertEqual(len(new.eligible(None, None, ())), 0)
        self.assertGreater(len(old.eligible(None, None, ())), 0)

    def test_topk_keeps_lexical_cutoff_ties_and_negative_scores(self):
        rng = np.random.default_rng(77)
        for size in (1, 5, 500):
            scores = rng.choice(np.array([-1., -.1, 0., .5, 1.], dtype=np.float32), size=size)
            keys = np.array([f"danbooru:{i}".encode() for i in range(size)])
            for count in (1, 4, 32, 256):
                expected = sorted(range(size), key=lambda i: (-scores[i], keys[i]))[:count]
                self.assertEqual(best_in_block(scores, keys, count).tolist(), expected)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA retrieval integration")
    def test_cuda_matches_independent_full_sort_both_codecs(self):
        for codec in ("f32le", "i8"):
            contract, entries = fixture(codec)
            snapshot = self.store.build(entries, contract, codec, bucket_count=7)
            memory = Recommender(self.store.root, snapshot)
            self.addCleanup(memory.close)
            for image, pixel, ratings in ((None, None, ()), (None, "a" * 64, ("s",)),
                                         (entries[0]["document"]["image_sha256"], None, ("g",)),
                                         (None, None, ("u",))):
                rows = eligible_reference(self.store, snapshot, image, pixel, ratings)
                query = torch.ones(8, device="cuda")
                query /= query.norm()
                expected = []
                for start in range(0, len(rows), 7):
                    batch = rows[start:start + 7]
                    matrix = torch.tensor(np.stack([decode_vector(e["vector"], 8) for e in batch]), device="cuda")
                    scores = (torch.nn.functional.normalize(matrix, dim=1) @ query).cpu().tolist()
                    expected.extend(zip(scores, (e["key"] for e in batch)))
                expected.sort(key=lambda item: (-item[0], item[1]))
                result = memory.recommend(np.ones(8), image, neighbors=4, batch_size=7,
                                          ratings=ratings, pixel_sha256=pixel)
                self.assertEqual([(e["similarity"], e["key"]) for e in result["neighbors"]], expected[:4])
                if expected:
                    self.assertEqual(result["identities"]["artist"][0]["tag"], "sample artist")
                    self.assertNotIn("highres", [e["tag"] for e in result["candidates"]])
            # A warmed query neither decompresses source buckets nor decodes vectors.
            with patch.object(Store, "read_bucket", side_effect=AssertionError("source reread")), \
                    patch("vrs_int.search_cache.decode_vector", side_effect=AssertionError("redecode")):
                memory.recommend(np.ones(8), None)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA retrieval integration")
    def test_cancelled_query_can_resume_and_close_reuses(self):
        memory = Recommender(self.store.root, self.snapshot)
        self.addCleanup(memory.close)
        result = memory.recommend(np.ones(8), None)
        with self.assertRaisesRegex(RuntimeError, "cancel"):
            memory.recommend(np.ones(8), None, interrupt=lambda: (_ for _ in ()).throw(RuntimeError("cancel")))
        self.assertEqual(memory.recommend(np.ones(8), None), result)
        path = memory._index.path
        memory.close()
        self.assertTrue(path.exists())
        self.assertEqual(list(path.glob("lease-*")), [])
        self.assertEqual(memory.recommend(np.ones(8), None), result)


class FileVectorTests(unittest.TestCase):
    """Exercise the Windows file-read path on every CI host without mocking OS."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "vectors.f32"
        self.data = np.arange(72, dtype="<f4").reshape(12, 6)
        self.data.tofile(self.path)
        self.rows = np.array([8, 2, 3, 9, 0, 7], dtype=np.int64)
        with patch("vrs_int.search_index.USE_MMAP", False), \
                patch("numpy.memmap", side_effect=AssertionError("No retained mapping on Windows")):
            self.vectors = MappedVectors(self.path, self.rows, 12, 6)
        self.addCleanup(self.vectors.close)

    def test_file_reads_match_scalar_slices_order_duplicates_and_empty_batches(self):
        for selected in (0, -1, slice(None), slice(None, None, -1), [], [3, 0, 3, 1],
                         np.array([[4, 1], [4, 0]]), self.rows % 2 == 0):
            with self.subTest(selected=selected):
                np.testing.assert_array_equal(self.vectors[selected], self.data[self.rows[selected]])
        self.vectors.release_pages()

    def test_append_and_rollback_preserve_open_view_without_retaining_handles(self):
        expected = self.vectors[:].copy()
        with self.path.open("ab") as stream:
            stream.write(np.ones((4, 6), dtype="<f4").tobytes())
        np.testing.assert_array_equal(self.vectors[:], expected)
        with self.path.open("r+b") as stream:
            stream.truncate(self.data.nbytes)
        np.testing.assert_array_equal(self.vectors[:], expected)

    def test_short_read_is_rejected_and_close_is_idempotent(self):
        self.path.write_bytes(b"bad")
        with self.assertRaisesRegex(OSError, "Truncated"):
            self.vectors[0]
        self.vectors.close()
        self.vectors.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            self.vectors[0]


if __name__ == "__main__":
    unittest.main()
