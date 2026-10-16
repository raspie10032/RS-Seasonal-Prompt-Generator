"""Core identities and immutable full-corpus ingestion, without GPU initialization."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from experiments.tag_policy import filter_tags, raw_tags
from experiments.vrs_full import assigned_split, ingest_full, source_rows


class CoreTagPolicyTests(unittest.TestCase):
    def test_both_source_serializations_and_typed_only_names_survive(self):
        old = {"tags": "hatsune miku, vocaloid, white dress, highres"}
        modern = {"tags": "hatsune_miku vocaloid white_dress highres", "tags_artist": "rare_artist"}
        self.assertEqual(set(filter_tags(old)[0]), {"hatsune miku", "vocaloid", "white dress"})
        self.assertEqual(set(filter_tags(modern)[0]), {"hatsune miku", "vocaloid", "white dress", "rare artist"})

    def test_explicit_identity_overrides_noise_rule(self):
        row = {"tags": "signature highres", "tags_artist": "signature", "tags_character": "highres",
               "tags_copyright": "original"}
        kept, removed = filter_tags(row)
        self.assertEqual(set(kept), {"signature", "highres", "original"})
        self.assertEqual(removed, {})

    def test_category_hints_protect_names_in_untyped_records(self):
        kept, removed = filter_tags({"tags": "signature, watermark, new character (series)"},
                                     {"signature": ["artist"]})
        self.assertIn("signature", kept)
        self.assertIn("new character (series)", kept)
        self.assertEqual(set(removed), {"watermark"})

    def test_visual_style_and_copyright_are_not_blanket_meta_drops(self):
        row = {"tags": "original blue_archive traditional_media official_art monochrome japanese_commentary highres watermark signature",
               "tags_copyright": "original blue_archive",
               "tags_meta": "traditional_media official_art japanese_commentary highres"}
        kept, removed = filter_tags(row)
        self.assertTrue({"original", "blue archive", "traditional media", "official art", "monochrome"} <= set(kept))
        self.assertEqual(set(removed), {"japanese commentary", "highres", "watermark", "signature"})

    def test_unknown_names_and_visual_properties_are_preserved(self):
        row = {"tags": "unheard_of_artist, unheard_of_character, low angle, side lighting, english text"}
        self.assertEqual(len(filter_tags(row)[0]), 5)


class FullIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "processed_v1").mkdir()
        self.out = self.root / "output"
        self.out.mkdir()

    def write(self, rows, splits):
        (self.data / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (self.data / "processed_v1/manifest_visual_expand.jsonl").write_text(
            "\n".join(json.dumps({"id": k, "split": v}) for k, v in splits.items()))

    def test_duplicate_images_keep_source_records_without_extra_votes(self):
        (self.data / "a.jpg").write_bytes(b"same image")
        (self.data / "b.jpg").write_bytes(b"same image")
        rows = [{"id": 1, "image": "a.jpg", "rating": "g", "tags": "rare_artist character_a series_a highres",
                 "tags_artist": "rare_artist", "tags_character": "character_a", "tags_copyright": "series_a"},
                {"id": 2, "image": "b.jpg", "rating": "g", "tags": "evaluation_only, highres"}]
        self.write(rows, {1: "train", 2: "test"})
        docs, hints, report = ingest_full(self.data, self.out)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["id"], 1)
        self.assertIn("rare artist", docs[0]["tags"])
        self.assertIn("evaluation only", docs[0]["tags"])
        self.assertEqual(docs[0]["source_ids"], [1, 2])
        self.assertEqual(docs[0]["split"], "train")
        self.assertEqual(hints["rare artist"], ["artist"])
        self.assertEqual(report["duplicate_image_records"], 1)
        with closing(sqlite3.connect(self.out / "observations.sqlite3")) as db:
            raw = db.execute("SELECT source_record,removed_tags FROM observations WHERE id=1").fetchone()
            self.assertEqual(json.loads(raw[0]), rows[0])
            self.assertIn("highres", json.loads(raw[1]))
            self.assertEqual(db.execute("SELECT count(*) FROM observations").fetchone()[0], 2)

    def test_missing_image_fails_instead_of_silently_dropping_identity(self):
        self.write([{"id": 1, "image": "missing.jpg", "rating": "g", "tags": "character"}], {1: "train"})
        with self.assertRaises(FileNotFoundError):
            ingest_full(self.data, self.out)

    def test_conflicting_source_id_fails(self):
        self.write([{"id": 1, "tags": "first"}, {"id": 1, "tags": "changed"}], {1: "train"})
        with self.assertRaisesRegex(ValueError, "Conflicting source ID"):
            source_rows(self.data)

    def test_previous_split_and_additional_record_assignment_are_stable(self):
        self.assertEqual(assigned_split(123, {123: "test"}), "test")
        self.assertEqual(assigned_split(123, {}), assigned_split(123, {}))
        self.assertIn(assigned_split(123, {}), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
