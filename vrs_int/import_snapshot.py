"""Migrate the local experiment without reading any original image or old cache."""
import json
from pathlib import Path
import sqlite3

from .store import Store, digest
from .vectors import encode_vector


def file_digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corrected_index_metadata(directory, memory_receipt, encoder_provenance=None):
    """Check the immutable repair artifact before mixing any features."""
    path = Path(directory)
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report.get("state") != "complete":
        raise ValueError("Incomplete corrected index")
    if report["memory_receipt_sha256"] != file_digest(memory_receipt):
        raise ValueError("Corrected index belongs to another VRS snapshot")
    if encoder_provenance is not None and report["encoder_provenance_sha256"] != file_digest(encoder_provenance):
        raise ValueError("Corrected index encoder mismatch")
    for name in ("train_selection.json", "corrected_vectors.npy"):
        if file_digest(path / name) != report["output_sha256"][name]:
            raise ValueError(f"Corrected index digest mismatch: {name}")
    rows = json.loads((path / "train_selection.json").read_text(encoding="utf-8"))
    if len(rows) != report["index_count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Corrected index row count mismatch")
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Corrected index must contain train observations only")
    return report, rows



def migrate(memory, corrected, encoder, destination, codec="f32le", label="local-initial", bucket_count=256):
    import numpy as np
    memory, corrected, encoder = map(Path, (memory, corrected, encoder))
    receipt = json.loads((memory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("state") != "complete":
        raise ValueError("Incomplete source snapshot")
    for name in ("documents.jsonl", "tag_categories.json", "observations.sqlite3"):
        if file_digest(memory / name) != receipt["files_sha256"][name]:
            raise ValueError("Source artifact mismatch: " + name)
    report, selected = corrected_index_metadata(corrected, memory / "receipt.json", encoder / "provenance.json")
    vectors = np.load(corrected / "corrected_vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(selected), 768) or not np.isfinite(vectors).all():
        raise ValueError("Invalid corrected vectors")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.01):
        raise ValueError("Unnormalized corrected vectors")
    positions = {row["id"]: i for i, row in enumerate(selected)}
    categories = json.loads((memory / "tag_categories.json").read_text(encoding="utf-8"))
    contract = {"encoder": {"provenance_sha256": report["encoder_provenance_sha256"],
                "preprocess": "gemma4-float-pixels-v1", "backend": "pil", "quantization": "nf4",
                "pooling": "mean-tokens-l2-v1"}, "dimension": 768,
                "codec": codec, "tag_policy": receipt["ingest"]["filter_policy"]}
    # Read-only source ledger. Preserve the exact original row fields, including
    # source_record JSON text and excluded tags, rather than just an external path.
    db = sqlite3.connect((memory / "observations.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    source_count = 0
    seen_vectors = set()

    def entries():
        nonlocal source_count
        with (memory / "documents.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                doc = json.loads(line)
                pos = positions.get(doc["id"])
                if pos is not None and doc != selected[pos]:
                    raise ValueError("Corrected vector observation mismatch")
                audit = [dict(row) for row in db.execute(
                    "SELECT * FROM observations WHERE canonical_id=? ORDER BY id", (doc["id"],))]
                if sorted(r["id"] for r in audit) != sorted(doc["source_ids"]):
                    raise ValueError("Missing source ledger records")
                source_count += len(audit)
                if pos is not None:
                    seen_vectors.add(doc["id"])
                yield {"key": f"danbooru:{doc['id']}", "document": doc,
                       "categories": {tag: categories[tag] for tag in doc["tags"] if tag in categories},
                       "vector": None if pos is None else encode_vector(vectors[pos], codec), "audit": audit}
        if len(seen_vectors) != len(selected) or source_count != db.execute("SELECT count(*) FROM observations").fetchone()[0]:
            raise ValueError("Source coverage mismatch")

    try:
        store = Store(destination)
        snapshot = store.build(entries(), contract, label, bucket_count=bucket_count, provenance={
            "memory_receipt_sha256": file_digest(memory / "receipt.json"),
            "corrected_report_sha256": file_digest(corrected / "report.json"),
            "image_coverage": "corrected subset only; legacy uint8 vectors excluded",
            "images_included": False, "source_metadata": "all original ledger fields included",
            "omitted_derived_artifacts": ["legacy vectors", "pair graph", "membership matrix", "evaluation logs"],
        })
        return store.verify(snapshot)
    finally:
        db.close()
