"""Personal collections preserve identities and can be extended in batches."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vrs_int.build import build_personal
from vrs_int.store import Store


class FakeEncoder:
    def __init__(self, directory):
        pass

    def encode(self, path):
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1
        return vector


class PersonalBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "one.png").write_bytes(b"mock image bytes; no live encoder in unit test")
        self.encoder = self.root / "encoder"
        self.encoder.mkdir()
        (self.encoder / "provenance.json").write_text("{}")
        self.manifest = self.root / "input.jsonl"
        self.store = Store(self.root / "store")
        self.row = {"id": 123, "image": "one.png", "rating": "e",
                    "tags": ["red_hair", "highres", "signature"],
                    "tags_artist": ["signature"], "tags_character": ["단일_캐릭터"],
                    "tags_copyright": ["new_work"]}

    def tearDown(self):
        self.temp.cleanup()

    def run_build(self, rows, **kwargs):
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return build_personal(self.manifest, self.images, self.encoder, self.store.root,
                              encoder_factory=FakeEncoder, **kwargs)

    def test_names_rating_original_metadata_and_filter_reasons_survive(self):
        report = self.run_build([self.row])
        self.assertEqual(report["observations"], 1)
        self.assertEqual(report["vectors"], 1)
        saved = self.store.get(report["snapshot"], "danbooru:123")
        self.assertEqual(saved["categories"]["signature"], ["artist"])
        self.assertIn("단일 캐릭터", saved["document"]["tags"])
        self.assertIn("new work", saved["document"]["tags"])
        self.assertNotIn("highres", saved["document"]["tags"])
        self.assertEqual(saved["audit"][0]["source_record"], self.row)
        self.assertIn("highres", saved["audit"][0]["removed_tags"])
        self.assertEqual(saved["document"]["rating"], "e")
        self.assertIsNone(self.store.current())
        self.assertTrue((self.images / "one.png").exists())
        self.assertFalse(list(self.store.root.rglob("*.png")))

    def test_increment_replay_and_revision_replace(self):
        base = self.run_build([self.row])["snapshot"]
        added = dict(self.row, id=124, rating="q")
        next_id = self.run_build([added], base=base)["snapshot"]
        self.assertEqual(self.store.verify(next_id)["observations"], 2)
        self.assertEqual(self.run_build([added], base=next_id)["snapshot"], next_id)
        changed = copy.deepcopy(self.row)
        changed["tags"].append("blue_sky")
        latest = self.run_build([changed], base=next_id)["snapshot"]
        self.assertEqual(self.store.verify(latest)["observations"], 2)
        self.assertNotIn("blue sky", self.store.get(base, "danbooru:123")["document"]["tags"])
        self.assertIn("blue sky", self.store.get(latest, "danbooru:123")["document"]["tags"])

    def test_encoder_mismatch_before_encoding(self):
        base = self.run_build([self.row])["snapshot"]
        (self.encoder / "provenance.json").write_text('{"different": true}')
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.run_build([self.row], base=base)
        self.assertEqual(len(list((self.store.root / "snapshots").iterdir())), 1)

    def test_invalid_inputs_never_publish(self):
        variants = [dict(self.row, image="../outside.png"), dict(self.row, rating="x"),
                    dict(self.row, tags="ambiguous string"), dict(self.row, id=True),
                    dict(self.row, image_sha256="0" * 64), dict(self.row, split="typo")]
        for row in variants:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.run_build([row])
        for rows in ([], [self.row, self.row]):
            with self.assertRaises(ValueError):
                self.run_build(rows)
        self.assertFalse(list((self.store.root / "snapshots").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
