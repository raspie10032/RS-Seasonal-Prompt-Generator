"""Immutable versions, transfers, conflicts, metadata retention, vector codecs."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vrs_int.store import Store, canonical, digest, pack, revision, unpack
from vrs_int.vectors import decode_vector, encode_vector


def contract(codec="f32le"):
    return {"encoder": {"provenance_sha256": "a" * 64, "preprocess": "float-v1", "backend": "pil",
                        "quantization": "nf4", "pooling": "mean-l2"},
            "dimension": 4, "codec": codec, "tag_policy": "core-names-v1"}


def entry(number=1):
    return {"key": f"danbooru:{number}", "document": {
        "id": number, "image_sha256": digest(str(number).encode()), "split": "train", "rating": "g",
        "image": "not-shipped/image.png", "source_ids": [number],
        "tags": ["rare artist (한글)", "singleton character", "new work", "red hair"]},
        "categories": {"rare artist (한글)": ["artist"], "singleton character": ["character"],
                       "new work": ["copyright"]},
        "vector": encode_vector([.1, -.2, .3, .9]),
        "audit": [{"source_record": '{"tags": ["highres", "rare artist (한글)"]}',
                   "removed_tags": '{"highres":"metadata"}'}]}


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "producer")
        self.records = [entry(i) for i in range(1, 20)]
        self.base = self.store.build(self.records, contract(), "initial", bucket_count=32)

    def tearDown(self):
        self.temp.cleanup()

    def change(self):
        row = copy.deepcopy(self.records[0])
        row["document"]["tags"].append("new scene")
        return {"op": "upsert", "key": row["key"], "expected_revision": revision(self.records[0]), "entry": row}

    def test_lossless_all_fields_and_singletons(self):
        actual = {e["key"]: e for e in self.store.entries(self.base)}
        self.assertEqual(actual, {e["key"]: e for e in self.records})
        self.assertEqual(self.store.verify(self.base)["source_records"], 19)

    def test_order_independent_build_and_noop(self):
        again = self.store.build(reversed(self.records), contract(), "initial", bucket_count=32)
        self.assertEqual(again, self.base)
        op = dict(self.change(), entry=self.records[0])
        self.assertEqual(self.store.update(self.base, [op], "no change"), self.base)

    def test_update_only_touches_one_bucket(self):
        new = self.store.update(self.base, [self.change()], "updated")
        left, right = self.store.manifest(self.base), self.store.manifest(new)
        self.assertEqual(sum(left["objects"].get(k) != v for k, v in right["objects"].items()), 1)
        self.assertEqual(self.store.verify(new)["observations"], 19)
        self.assertEqual(self.store.get(self.base, self.records[0]["key"]), self.records[0])
        self.assertIn("new scene", self.store.get(new, self.records[0]["key"])["document"]["tags"])

    def test_conflict_and_missing_revision_do_not_publish(self):
        op = self.change()
        op["expected_revision"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.store.update(self.base, [op], "wrong")
        del op["expected_revision"]
        with self.assertRaises(ValueError):
            self.store.update(self.base, [op], "missing")
        self.assertEqual(len(list((self.store.root / "snapshots").iterdir())), 1)

    def test_add_delete_restore_and_retained_history(self):
        row = entry(100)
        new = self.store.update(self.base, [
            {"op": "upsert", "key": row["key"], "expected_revision": None, "entry": row},
            {"op": "delete", "key": self.records[0]["key"], "expected_revision": revision(self.records[0]), "reason": "source withdrawn"}], "delta")
        tombstone = self.store.get(new, self.records[0]["key"])
        self.assertTrue(tombstone["deleted"])
        self.assertEqual(self.store.verify(new)["observations"], 19)
        self.assertEqual(self.store.verify(new)["tombstones"], 1)
        self.assertNotIn(self.records[0], list(self.store.entries(new)))
        self.assertEqual(self.store.get(self.base, self.records[0]["key"]), self.records[0])
        restored = self.store.update(new, [{"op": "upsert", "key": self.records[0]["key"],
            "expected_revision": revision(tombstone), "entry": self.records[0]}], "restore")
        self.assertEqual(self.store.verify(restored)["observations"], 20)

    def test_incompatible_encoder_and_codec_rejected(self):
        for field in ("encoder", "codec", "tag_policy"):
            other = contract()
            if field == "encoder":
                other[field]["preprocess"] = "different"
            else:
                other[field] = "different"
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                self.store.update(self.base, [self.change()], "wrong", other)

    def test_complete_transfer_delta_activation_and_rollback(self):
        receiver = Store(self.root / "receiver")
        full_package = self.root / "full"
        full = self.store.export(self.base, full_package)
        self.assertEqual(receiver.install(full_package), self.base)
        self.assertIsNone(receiver.current())
        receiver.activate(self.base, None)
        newer = self.store.update(self.base, [self.change()], "next")
        delta = self.store.export(newer, self.root / "delta", self.base)
        self.assertLess(delta["package_bytes"], full["package_bytes"])
        self.assertEqual(delta["transferred_objects"], 1)
        receiver.install(self.root / "delta")
        self.assertEqual(receiver.current(), self.base)
        receiver.activate(newer, self.base)
        with self.assertRaisesRegex(ValueError, "Active snapshot changed"):
            receiver.activate(self.base, self.base)
        self.assertEqual(receiver.current(), newer)
        receiver.activate(self.base, newer)
        self.assertEqual(receiver.current(), self.base)

    def test_missing_parent_and_corrupt_transfer_keep_active(self):
        newer = self.store.update(self.base, [self.change()], "next")
        package = self.root / "delta"
        self.store.export(newer, package, self.base)
        receiver = Store(self.root / "receiver")
        with self.assertRaises(FileNotFoundError):
            receiver.install(package)
        self.store.activate(self.base, None)
        next((package / "objects").iterdir()).write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.store.install(package)
        self.assertEqual(self.store.current(), self.base)

    def test_foreign_branch_delta_rejected(self):
        newer = self.store.update(self.base, [self.change()], "next")
        fork_op = self.change()
        fork_op["entry"]["document"]["tags"].append("local only")
        local = self.store.update(self.base, [fork_op], "local")
        with self.assertRaisesRegex(ValueError, "exact immediate parent"):
            self.store.export(newer, self.root / "bad", local)

    def test_valid_hash_invalid_counts_not_published(self):
        package = self.root / "incorrect-counts"
        self.store.export(self.base, package)
        manifest = self.store.manifest(self.base)
        manifest["counts"]["observations"] += 1
        raw = canonical(manifest)
        forged_id = digest(raw)
        (package / "snapshots" / (forged_id + ".json")).write_bytes(raw)
        transfer = json.loads((package / "package.json").read_text())
        transfer["snapshot"] = forged_id
        (package / "package.json").write_bytes(canonical(transfer))
        receiver = Store(self.root / "receiver")
        with self.assertRaisesRegex(ValueError, "Incoming snapshot counts"):
            receiver.install(package)
        self.assertFalse(receiver.manifest_path(forged_id).exists())
        self.assertIsNone(receiver.current())

    def test_duplicate_base_and_operations_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.store.build([entry(), entry()], contract(), "duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.store.update(self.base, [self.change(), self.change()], "duplicate")

    def test_build_closes_spool_on_success_and_failure(self):
        connect = sqlite3.connect

        def broken_input():
            yield entry()
            raise ValueError("input interrupted")

        for records, error in (([entry()], None), ([entry(), entry()], "Duplicate"),
                               (broken_input(), "input interrupted")):
            with self.subTest(error=error):
                connections = []

                def track_connection(*args, **kwargs):
                    db = connect(*args, **kwargs)
                    connections.append(db)
                    return db

                try:
                    with patch("vrs_int.store.sqlite3.connect", side_effect=track_connection):
                        if error:
                            with self.assertRaisesRegex(ValueError, error):
                                self.store.build(records, contract(), "close-on-error")
                        else:
                            self.store.build(records, contract(), "close-on-success")
                    self.assertEqual(len(connections), 1)
                    with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
                        connections[0].execute("SELECT 1")
                    self.assertEqual(list(self.store.root.glob(".build-*")), [])
                finally:
                    for db in connections:
                        db.close()

    def test_manifest_corruption_and_path_traversal(self):
        with self.assertRaises(ValueError):
            self.store.manifest("../../bad")
        self.store.manifest_path(self.base).write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.store.verify(self.base)

    def test_unknown_format_fails(self):
        manifest = self.store.manifest(self.base)
        manifest["format"] = "vrs-int/999"
        raw = canonical(manifest)
        snapshot = digest(raw)
        self.store.manifest_path(snapshot).write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.store.verify(snapshot)

    def test_corrupt_object_prevents_activation(self):
        self.store.activate(self.base, None)
        newer = self.store.update(self.base, [self.change()], "next")
        old_objects = set(self.store.manifest(self.base)["objects"].values())
        new_objects = set(self.store.manifest(newer)["objects"].values())
        self.store.object_path((new_objects - old_objects).pop()).write_bytes(b"truncated")
        with self.assertRaises(ValueError):
            self.store.activate(newer, self.base)
        self.assertEqual(self.store.current(), self.base)


class CodecTests(unittest.TestCase):
    def test_fp32_exact_bits_and_int8_error(self):
        vector = np.random.default_rng(42).normal(size=768).astype("<f4")
        vector /= np.linalg.norm(vector)
        exact = decode_vector(encode_vector(vector), 768)
        self.assertEqual(exact.tobytes(), vector.tobytes())
        compact = decode_vector(encode_vector(vector, "i8"), 768)
        self.assertGreater(float(vector @ compact / np.linalg.norm(compact)), .9999)

    def test_invalid_vectors(self):
        for values in ([0., 0.], [float("nan"), 1], [float("inf"), 1]):
            with self.assertRaises(ValueError):
                encode_vector(values)
        with self.assertRaises(ValueError):
            decode_vector(encode_vector([1., 2.]), 3)

    def test_dictionary_preserves_order_unicode_and_category(self):
        rows = [entry(10), entry(20)]
        self.assertEqual(unpack(pack(rows)), sorted(rows, key=lambda e: e["key"]))


if __name__ == "__main__":
    unittest.main()
