"""Full metadata preservation audit and local full/delta transfer simulation."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sqlite3
import time

from vrs_int.store import Store, canonical, digest, revision
from vrs_int.vectors import decode_vector


def run(args):
    import numpy as np
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    base = Store(args.base)
    compact = Store(args.compact)
    left = next((args.base / "snapshots").glob("*.json")).stem
    right = next((args.compact / "snapshots").glob("*.json")).stem
    expected_categories = json.loads((args.memory / "tag_categories.json").read_text())
    selected = json.loads((args.corrected / "train_selection.json").read_text())
    original_vectors = np.load(args.corrected / "corrected_vectors.npy", allow_pickle=False)
    original_positions = {r["id"]: i for i, r in enumerate(selected)}
    document_hashes, audit_hashes, metadata_hashes = {}, {}, {}
    names = {k: set() for k in ("artist", "character", "copyright")}
    singleton_frequency = Counter()
    ratings = Counter()
    exact_vectors = 0
    witness = None
    left_features = []
    for entry in base.entries(left):
        doc = entry["document"]
        witness = witness or entry
        document_hashes[doc["id"]] = digest(canonical(doc))
        expected = {t: expected_categories[t] for t in doc["tags"] if t in expected_categories}
        assert entry["categories"] == expected
        for tag, categories in expected.items():
            for category in categories:
                if category in names:
                    names[category].add(tag)
            singleton_frequency[tag] += 1
        for row in entry["audit"]:
            assert row["id"] not in audit_hashes
            audit_hashes[row["id"]] = digest(canonical(row))
        ratings[doc["rating"]] += 1
        metadata_hashes[entry["key"]] = digest(canonical(dict(entry, vector=None)))
        if entry["vector"] is not None:
            vector = decode_vector(entry["vector"], 768)
            assert vector.tobytes() == original_vectors[original_positions[doc["id"]]].tobytes()
            exact_vectors += 1
            left_features.append((entry["key"], vector.copy()))
    checked_documents = 0
    with (args.memory / "documents.jsonl").open() as stream:
        for line in stream:
            doc = json.loads(line)
            assert document_hashes.pop(doc["id"]) == digest(canonical(doc))
            checked_documents += 1
    assert not document_hashes
    with sqlite3.connect((args.memory / "observations.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT * FROM observations"):
            assert audit_hashes.pop(row["id"]) == digest(canonical(dict(row)))
    assert not audit_hashes
    right_features = []
    for entry in compact.entries(right):
        assert metadata_hashes.pop(entry["key"]) == digest(canonical(dict(entry, vector=None)))
        if entry["vector"] is not None:
            right_features.append((entry["key"], decode_vector(entry["vector"], 768).copy()))
    assert not metadata_hashes
    assert [k for k, _ in left_features] == [k for k, _ in right_features]
    cosines = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
               for (_, a), (_, b) in zip(left_features, right_features)]
    # A clearly marked synthetic metadata change exercises distribution. It is
    # not claimed to be a new Danbooru observation and never activates upstream.
    modified = copy.deepcopy(witness)
    modified["document"]["local_note"] = "Synthetic VRS-INT transfer test; not a Danbooru annotation"
    child = base.update(left, [{"op": "upsert", "key": witness["key"],
        "expected_revision": revision(witness), "entry": modified}], "transfer-simulation-only",
        provenance={"synthetic": True, "purpose": "delta transfer and rollback test"})
    full_report = base.export(left, args.output / "full-package")
    delta_report = base.export(child, args.output / "delta-package", left)
    receiver = Store(args.output / "receiver")
    assert receiver.install(args.output / "full-package") == left
    assert receiver.current() is None
    receiver.activate(left, None)
    assert receiver.install(args.output / "delta-package") == child
    assert receiver.current() == left
    receiver.activate(child, left)
    assert receiver.get(child, witness["key"]) == modified
    assert receiver.get(left, witness["key"]) == witness
    receiver.activate(left, child)
    assert receiver.current() == left
    assert base.current() is None
    report = {"state": "complete", "base": base.verify(left), "i8": compact.verify(right),
        "compared_documents": checked_documents, "compared_source_records": base.manifest(left)["counts"]["source_records"],
        "all_categories_preserved": True, "protected_names": {k: len(v) for k, v in names.items()},
        "protected_singletons": {k: sum(singleton_frequency[t] == 1 for t in v) for k, v in names.items()},
        "ratings": dict(ratings), "f32_vectors_bit_exact": exact_vectors,
        "i8_metadata_identical": True, "i8_vector_cosine_min": min(cosines),
        "i8_vector_cosine_mean": sum(cosines) / len(cosines),
        "canonical_jsonl_input_bytes": base.manifest(left)["provenance"]["canonical_input_jsonl_bytes"],
        "full_transfer": full_report, "delta_transfer": delta_report,
        "install_did_not_activate": True, "rollback_passed": True,
        "source_images_read": 0, "elapsed_seconds": time.perf_counter() - start,
        "scope": "Metadata preservation and package behavior; compression is not measured versus omitted legacy artifacts."}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "compact", "memory", "corrected", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    run(parser.parse_args())
