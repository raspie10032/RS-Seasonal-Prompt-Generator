"""Legacy import remains self-contained; fixtures are entirely synthetic."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from vrs_int.import_snapshot import corrected_index_metadata, file_digest, migrate
from vrs_int.store import Store


class MigrationTests(unittest.TestCase):
    def test_import_preserves_source_and_rejects_changed_encoder_or_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory, corrected, encoder = (root / name for name in ("legacy", "corrected", "encoder"))
            for path in (memory, corrected, encoder):
                path.mkdir()
            doc = {"id": 1, "image": "not-shipped.png", "image_sha256": "a" * 64,
                   "rating": "g", "split": "train", "source_ids": [1], "tags": ["example artist"]}
            source = {"id": 1, "tags_artist": ["example_artist"], "tags_meta": ["highres"]}
            (memory / "documents.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            (memory / "tag_categories.json").write_text('{"example artist": ["artist"]}', encoding="utf-8")
            db = sqlite3.connect(memory / "observations.sqlite3")
            db.execute("CREATE TABLE observations (id INTEGER, canonical_id INTEGER, source_record TEXT)")
            db.execute("INSERT INTO observations VALUES (1, 1, ?)", (json.dumps(source),))
            db.commit()
            db.close()
            receipt = {"state": "complete", "ingest": {"filter_policy": "fictional-v1"},
                       "files_sha256": {name: file_digest(memory / name) for name in (
                           "documents.jsonl", "tag_categories.json", "observations.sqlite3")}}
            (memory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            (encoder / "provenance.json").write_text('{"fictional": true}', encoding="utf-8")
            (corrected / "train_selection.json").write_text(json.dumps([doc]), encoding="utf-8")
            vectors = np.zeros((1, 768), dtype="<f4")
            vectors[0, 0] = 1
            np.save(corrected / "corrected_vectors.npy", vectors)
            report = {"state": "complete", "index_count": 1,
                      "memory_receipt_sha256": file_digest(memory / "receipt.json"),
                      "encoder_provenance_sha256": file_digest(encoder / "provenance.json"),
                      "output_sha256": {name: file_digest(corrected / name) for name in (
                          "train_selection.json", "corrected_vectors.npy")}}
            (corrected / "report.json").write_text(json.dumps(report), encoding="utf-8")
            result = migrate(memory, corrected, encoder, root / "new")
            stored = Store(root / "new").get(result["snapshot"], "danbooru:1")
            self.assertEqual(stored["document"], doc)
            self.assertEqual(json.loads(stored["audit"][0]["source_record"]), source)
            (encoder / "provenance.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "encoder mismatch"):
                corrected_index_metadata(corrected, memory / "receipt.json", encoder / "provenance.json")
            (corrected / "corrected_vectors.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                corrected_index_metadata(corrected, memory / "receipt.json")


if __name__ == "__main__":
    unittest.main()
