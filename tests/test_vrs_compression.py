"""Lossless experience compaction, compatibility, bounded decode and recovery."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_vrs_int import contract, entry
from vrs_int.store import Store, COMPACT_FORMAT, canonical, digest, pack, unpack, revision
from vrs_int.recommend import Recommender
from vrs_native_nodes import VRSNativeMemoryCompact


class CompactStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root / "source")
        self.rows = [entry(i) for i in range(1, 20)]
        self.base = self.store.build(self.rows, contract(), "test", bucket_count=7)
        disk = patch("shutil.disk_usage", return_value=SimpleNamespace(total=1000000000, free=800000000))
        disk.start()
        self.addCleanup(disk.stop)

    def repack(self, source=None):
        return self.store.repack(source or self.base, self.root / "compact")

    def test_all_fields_revisions_and_source_files_survive(self):
        before = {p: p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()}
        report = self.repack()
        target = Store(report["store"])
        self.assertEqual(target.manifest(report["snapshot"])["format"], COMPACT_FORMAT)
        self.assertEqual(target.manifest(report["snapshot"])["contract"], contract())
        self.assertIsNone(target.current())
        for row in target.entries(report["snapshot"]):
            self.assertEqual(canonical(row), canonical(self.store.get(self.base, row["key"])))
            self.assertEqual(revision(row), revision(self.store.get(self.base, row["key"])))
        self.assertEqual(before, {p: p.read_bytes() for p in self.store.root.rglob("*") if p.is_file()})

    def test_tombstones_transfer_and_compact_delta(self):
        deleted = self.store.update(self.base, [dict(op="delete", key=self.rows[0]["key"],
            expected_revision=revision(self.rows[0]), reason="removed by user")], "deleted")
        report = self.repack(deleted)
        target = Store(report["store"])
        snapshot = report["snapshot"]
        self.assertEqual(target.get(snapshot, self.rows[0]["key"]), self.store.get(deleted, self.rows[0]["key"]))
        target.export(snapshot, self.root / "package")
        receiver = Store(self.root / "receiver")
        self.assertEqual(receiver.install(self.root / "package"), snapshot)
        edited = copy.deepcopy(self.rows[1])
        edited["document"]["tags"].append("새로운 장면")
        update = target.update(snapshot, [dict(op="upsert", key=edited["key"], entry=edited,
            expected_revision=revision(self.rows[1]))], "edit")
        self.assertEqual(target.manifest(update)["format"], COMPACT_FORMAT)
        target.export(update, self.root / "delta", snapshot)
        self.assertEqual(receiver.install(self.root / "delta"), update)
        self.assertEqual(receiver.get(update, edited["key"]), edited)
        self.assertEqual(receiver.get(snapshot, edited["key"]), self.rows[1])
        target.activate(update, None)
        self.assertEqual(target.current(), update)

    def test_generated_upsert_cannot_downgrade_compact_format(self):
        report = self.repack()
        target = Store(report["store"])
        generated = copy.deepcopy(self.rows[0])
        sha = generated["document"]["image_sha256"]
        generated["key"] = "generated:" + sha
        generated["document"]["source"] = "generated"
        generated["document"]["id"] = sha
        generated["document"]["rating"] = "u"
        new = target.update(report["snapshot"], [dict(op="upsert", key=generated["key"],
            entry=generated, expected_revision=None)], "add generated")
        self.assertEqual(target.manifest(new)["format"], COMPACT_FORMAT)
        self.assertEqual(target.get(new, generated["key"]), generated)

    def test_noop_update_stays_same_snapshot(self):
        report = self.repack()
        target = Store(report["store"])
        self.assertEqual(target.update(report["snapshot"], [dict(op="upsert", key=self.rows[0]["key"],
            entry=self.rows[0], expected_revision=revision(self.rows[0]))], "noop"), report["snapshot"])

    def test_native_compact_node_needs_no_encoder_and_checks_cancellation(self):
        calls = []
        mm = SimpleNamespace(throw_exception_if_processing_interrupted=lambda: calls.append(True))
        with patch.dict(sys.modules, {"comfy": SimpleNamespace(model_management=mm),
                                     "comfy.model_management": mm}):
            output = VRSNativeMemoryCompact().compact(Recommender(self.store.root, self.base),
                                                     str(self.root / "compact"))
        memory, snapshot, raw = output["result"]
        self.assertTrue(calls)
        self.assertEqual(snapshot, memory.snapshot)
        self.assertEqual(list(memory.store.entries(snapshot)), list(self.store.entries(self.base)))
        self.assertEqual(output["ui"]["text"], [raw])

    def test_compact_objects_cannot_pretend_to_be_older_format(self):
        report = self.repack()
        target = Store(report["store"])
        manifest = target.manifest(report["snapshot"])
        manifest["format"] = "vrs-int/1"
        raw = canonical(manifest)
        snapshot = digest(raw)
        target.manifest_path(snapshot).write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "require vrs-int/3"):
            target.verify(snapshot)

    def test_bad_target_cancel_and_low_disk_never_publish(self):
        for target in (self.store.root, self.store.root / "nested", self.root):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.store.repack(self.base, target)
        target = self.root / "existing"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            self.store.repack(self.base, target)

    def test_cancel_and_low_disk_leave_no_manifest(self):
        target = self.root / "cancelled"
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.store.repack(self.base, target, lambda: (_ for _ in ()).throw(RuntimeError("cancelled")))
        self.assertEqual(list(target.rglob("*.json")), [])
        with patch("shutil.disk_usage", return_value=SimpleNamespace(total=1000000, free=190000)):
            with self.assertRaisesRegex(OSError, "20%"):
                self.repack()
        self.assertEqual(list((self.root / "compact").rglob("*.json")), [])
        self.assertEqual(self.store.verify(self.base)["observations"], len(self.rows))

    def test_source_corruption_or_false_counts_never_publish(self):
        manifest = self.store.manifest(self.base)
        manifest["counts"]["observations"] += 1
        raw = canonical(manifest)
        bad = digest(raw)
        self.store.manifest_path(bad).write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "counts mismatch"):
            self.repack(bad)
        self.assertEqual(list((self.root / "compact").rglob("*.json")), [])

    def test_written_object_corruption_never_publishes(self):
        from vrs_int.store import write_once
        def corrupt_write(path, raw):
            write_once(path, raw)
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with patch("vrs_int.store.write_once", side_effect=corrupt_write):
            with self.assertRaisesRegex(ValueError, "Lossless bucket verification"):
                self.repack()
        self.assertEqual(list((self.root / "compact" / "snapshots").glob("*.json")), [])
        self.assertEqual(self.store.verify(self.base)["observations"], len(self.rows))

    def test_documented_cli_creates_verified_independent_copy(self):
        from vrs_int.__main__ import main
        output = io.StringIO()
        argv = ["vrs_int", "repack", "--store", str(self.store.root),
                "--snapshot", self.base, "--output", str(self.root / "cli-copy")]
        with patch.object(sys, "argv", argv), redirect_stdout(output):
            main()
        report = json.loads(output.getvalue())
        self.assertTrue(report["lossless"])
        self.assertEqual(Store(report["store"]).verify(report["snapshot"])["observations"], len(self.rows))

    def test_bounded_decoder_rejects_truncated_trailing_and_oversized_xz(self):
        raw = pack(self.rows, "xz")
        self.assertEqual(unpack(raw), sorted(self.rows, key=lambda e: e["key"]))
        for invalid in (raw[:-2], raw + b"trailing", raw + raw):
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                unpack(invalid)
        with patch("vrs_int.store.MAX_OBJECT_BYTES", 128):
            with self.assertRaisesRegex(ValueError, "oversized"):
                unpack(raw)
        # The original zlib format remains readable and the default writer.
        self.assertEqual(unpack(pack(self.rows)), unpack(raw))


if __name__ == "__main__":
    unittest.main()
