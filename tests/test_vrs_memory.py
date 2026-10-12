"""Offline data-integrity tests; no CUDA initialization or model downloads."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.vrs_memory import (contained, observation_tags, open_ledger,
                                     read_rows, record_observation)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.db = open_ledger(self.path / "ledger.sqlite3")
        self.addCleanup(self.db.close)
        self.row = {"id": 1, "flat_tags": ["white_dress"], "identity_tags": [],
                    "rating": "g", "split": "train"}

    def insert(self, row=None, image="sha-image", cache="sha-cache"):
        return record_observation(self.db, row or self.row, image, cache, b"vector")

    def test_replay_does_not_add_votes(self):
        self.assertEqual(self.insert(), "added")
        self.assertEqual(self.insert(), "replayed")
        self.assertEqual(self.db.execute("SELECT count(*) FROM observations").fetchone()[0], 1)

    def test_same_image_under_different_id_does_not_add_votes(self):
        self.insert()
        self.assertEqual(self.insert(dict(self.row, id=2)), "duplicate_image")
        self.assertEqual(self.db.execute("SELECT count(*) FROM observations").fetchone()[0], 1)

    def test_changed_labels_require_explicit_version(self):
        self.insert()
        with self.assertRaisesRegex(ValueError, "Changed observation"):
            self.insert(dict(self.row, flat_tags=["blue_dress"]))

    def test_changed_image_or_cache_is_not_silently_accepted(self):
        self.insert()
        with self.assertRaisesRegex(ValueError, "Changed observation"):
            self.insert(image="different-image")
        with self.assertRaisesRegex(ValueError, "Changed observation"):
            self.insert(cache="different-cache")

    def test_new_character_is_preserved_in_observation(self):
        row = dict(self.row, id=3, identity_tags=["new_character (new_series)"])
        self.insert(row)
        self.assertIn("new character (new series)", observation_tags(row))

    def test_manifest_selection_cannot_mix_test_or_ratings(self):
        rows = [self.row, dict(self.row, id=2, split="test"), dict(self.row, id=3, rating="e")]
        manifest = self.path / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(r) for r in rows))
        self.assertEqual([r["id"] for r in read_rows(manifest)], [1])

    def test_conflicting_duplicate_manifest_ids_fail(self):
        manifest = self.path / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(r) for r in [self.row, dict(self.row, flat_tags=[])]))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            read_rows(manifest)

    def test_dataset_traversal_and_absolute_escape_are_rejected(self):
        for path in ("../escape.pt", str(self.path.parent / "escape.pt")):
            with self.assertRaisesRegex(ValueError, "escapes"):
                contained(self.path, path)

    def test_symlink_escape_is_rejected(self):
        link = self.path / "outside"
        try:
            link.symlink_to(self.path.parent, target_is_directory=True)
        except OSError:
            self.skipTest("Symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "escapes"):
            contained(self.path, "outside/escape.pt")


if __name__ == "__main__":
    unittest.main()
