"""Reproducible held-out tag retrieval metrics from immutable stored features.

No image decoding, encoder, weight download, activation, or snapshot writes.
Scores remain neighbor votes. Ground truth is used only after retrieval.
"""
from collections import Counter
import json
import platform
from pathlib import Path

from .recommend import Recommender
from .store import canonical, digest
from .tag_policy import IDENTITY, raw_tags
from .vectors import decode_vector


def _source_ids(entry):
    doc = entry["document"]
    ids = [doc["id"], *doc.get("source_ids", [])]
    if any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError("Evaluation requires positive integer source IDs")
    return set(ids)


def _original_tags(entry):
    """Use original audit tags when available; never invent missing source labels."""
    tags, covered = set(), False
    for audit in entry["audit"]:
        row = audit.get("source_record") if isinstance(audit, dict) else None
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except ValueError:
                row = None
        if isinstance(row, dict) and any(k in row for k in ("tags", "tags_artist", "tags_character",
                                                           "tags_copyright", "tags_general", "tags_meta")):
            try:
                tags.update(raw_tags(row))
            except (TypeError, AttributeError):
                continue
            covered = True
    return (tags if covered else set(entry["document"]["tags"])), covered


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(counts):
    """Empty truth is excluded from macro recall; short lists still divide by k."""
    n = counts["queries"]
    return {**counts,
            "precision_micro": _ratio(counts["hits"], counts["slots"]),
            "recall_micro": _ratio(counts["hits"], counts["truth"]),
            "precision_macro": _ratio(counts["precision_sum"], n),
            "recall_macro": _ratio(counts["recall_sum"], counts["queries_with_truth"])}


def _empty():
    return dict(queries=0, queries_with_truth=0, hits=0, truth=0, predictions=0, slots=0,
                precision_sum=0., recall_sum=0.)


def _add(counts, predicted, truth, slots):
    hits = len(set(predicted) & truth)
    counts["queries"] += 1
    counts["queries_with_truth"] += bool(truth)
    counts["hits"] += hits
    counts["truth"] += len(truth)
    counts["predictions"] += len(predicted)
    counts["slots"] += slots
    counts["precision_sum"] += hits / slots
    if truth:
        counts["recall_sum"] += hits / len(truth)


def _preservation(original_df, preserved_df, rare_max):
    rare = {t for t, n in original_df.items() if n <= rare_max}
    preserved = {t for t in rare if preserved_df[t]}
    total = sum(original_df[t] for t in rare)
    kept = sum(preserved_df[t] for t in rare)
    return {"original_rare_vocabulary": len(rare), "preserved_rare_vocabulary": len(preserved),
            "vocabulary_preservation": _ratio(len(preserved), len(rare)),
            "original_rare_occurrences": total, "preserved_rare_occurrences": kept,
            "occurrence_preservation": _ratio(kept, total)}


def evaluate(store, snapshot, *, split="test", ks=(5, 10, 20), neighbors=32,
             ratings=(), rare_max=5, backend="numpy", batch_size=2048, include_cases=False):
    """Pin a snapshot and exclude *all* val/test hashes and source IDs from train.

    Holdout duplicates are counted once by image hash (stable key order). Missing
    query vectors are reported as skipped, never silently included as successes.
    No train-only snapshot is manufactured and no active pointer is touched.
    """
    import numpy as np
    ks = tuple(sorted(set(ks)))
    if split not in ("val", "test") or backend not in ("numpy", "cuda"):
        raise ValueError("Evaluation requires val/test and numpy/cuda backend")
    if not ks or any(type(k) is not int or not 1 <= k <= 100 for k in ks):
        raise ValueError("Top-k values must be integers in 1..100")
    if type(rare_max) is not int or rare_max < 1:
        raise ValueError("rare_max must be a positive integer")
    if not 1 <= neighbors <= 256 or not 1 <= batch_size <= 4096:
        raise ValueError("Invalid retrieval limits")
    if any(r not in ("g", "s", "q", "e") for r in ratings):
        raise ValueError("Invalid ratings")
    memory = Recommender(store, snapshot)
    verified = memory.store.verify(snapshot)
    heldout = [e for e in memory.store.entries(snapshot) if e["document"]["split"] != "train"]
    excluded_images = {e["document"]["image_sha256"] for e in heldout}
    excluded_ids = set().union(*(_source_ids(e) for e in heldout)) if heldout else set()
    targets = sorted((e for e in heldout if e["document"]["split"] == split
                      and (not ratings or e["document"]["rating"] in ratings)), key=lambda e: e["key"])
    if not targets:
        raise ValueError("No holdout observations for the requested split/ratings")
    # Frequency is computed only from leak-free train observations, one vote/image.
    train = list(memory._entries(None, ratings, excluded_images, excluded_ids))
    train_all = [e for e in memory._visual_entries if not ratings or e["document"]["rating"] in ratings]
    retained_df, original_df, preserved_df, categories = Counter(), Counter(), Counter(), {}
    audit_covered = storage_audit_covered = storage_train_count = missing_train_vectors = 0
    storage_original_df, storage_preserved_df = Counter(), Counter()
    eligible_keys = {e["key"] for e in train}
    for e in memory.store.entries(snapshot):
        doc = e["document"]
        if doc["split"] != "train" or (ratings and doc["rating"] not in ratings):
            continue
        original, covered = _original_tags(e)
        storage_train_count += 1
        missing_train_vectors += e["vector"] is None
        storage_audit_covered += covered
        storage_original_df.update(original)
        storage_preserved_df.update(original & set(doc["tags"]))
        if e["key"] not in eligible_keys:
            continue
        audit_covered += covered
        original_df.update(original)
        retained_df.update(set(e["document"]["tags"]))
        preserved_df.update(original & set(e["document"]["tags"]))
        for tag, kinds in e["categories"].items():
            categories.setdefault(tag, set()).update(kinds)
    vocabulary = set(retained_df)
    excluded_images, excluded_ids = frozenset(excluded_images), frozenset(excluded_ids)
    rare = {t for t, n in retained_df.items() if n <= rare_max}
    original_rare = {t for t, n in original_df.items() if n <= rare_max}
    rare_preserved = original_rare & vocabulary
    overall = {k: _empty() for k in ks}
    category_counts = {c: {k: _empty() for k in ks} for c in IDENTITY}
    rare_counts = {k: _empty() for k in ks}
    unseen_counts = {k: _empty() for k in ks}
    identities = {c: _empty() for c in IDENTITY}
    tag_truth, tag_hits = Counter(), {k: Counter() for k in ks}
    errors = {k: Counter() for k in ks}
    cases, evaluated_keys, skipped, seen = [], [], [], set()
    retrieve = memory.recommend_offline if backend == "numpy" else memory.recommend
    for e in targets:
        doc = e["document"]
        if e["vector"] is None:
            skipped.append({"key": e["key"], "reason": "missing_vector"})
            continue
        if doc["image_sha256"] in seen:
            skipped.append({"key": e["key"], "reason": "duplicate_holdout_image"})
            continue
        seen.add(doc["image_sha256"])
        vector = decode_vector(e["vector"], memory.manifest["contract"]["dimension"])
        result = retrieve(vector, doc["image_sha256"], neighbors=neighbors, limit=max(ks),
                          ratings=ratings, batch_size=batch_size, exclude_images=excluded_images,
                          exclude_source_ids=excluded_ids)
        truth = set(doc["tags"])
        typed = {c: {t for t in truth if c in e["categories"].get(t, [])} for c in IDENTITY}
        predicted = [c["tag"] for c in result["candidates"]]
        tag_truth.update(truth)
        evaluated_keys.append(e["key"])
        for c in IDENTITY:
            _add(identities[c], [x["tag"] for x in result["identities"].get(c, [])], typed[c], 3)
        details = {}
        for k in ks:
            top = predicted[:k]
            selected = set(top)
            _add(overall[k], top, truth, k)
            _add(rare_counts[k], [t for t in top if t in rare], truth & rare, k)
            _add(unseen_counts[k], [t for t in top if t not in retained_df], truth - vocabulary, k)
            for c in IDENTITY:
                _add(category_counts[c][k], [x["tag"] for x in result["candidates"][:k]
                                            if c in x["categories"]], typed[c], k)
            tag_hits[k].update(selected & truth)
            fp, fn = selected - truth, truth - selected
            errors[k]["queries_with_false_positives"] += bool(fp)
            errors[k]["queries_with_false_negatives"] += bool(fn)
            errors[k]["no_recommendation"] += not top
            errors[k]["zero_hit_with_recommendations"] += bool(top) and not (selected & truth)
            errors[k]["false_positive_tags"] += len(fp)
            errors[k]["false_negative_tags"] += len(fn)
            if include_cases:
                details[str(k)] = {"predicted": top, "false_positives": sorted(fp),
                                   "false_negatives": sorted(fn),
                                   "unseen_truth": sorted(truth - vocabulary),
                                   "rare_misses": sorted((truth & rare) - selected),
                                   "identity_errors": {c: {
                                       "false_positives": sorted({x["tag"] for x in result["candidates"][:k]
                                                                  if c in x["categories"]} - typed[c]),
                                       "false_negatives": sorted(typed[c] - {x["tag"] for x in result["candidates"][:k]
                                                                           if c in x["categories"]})}
                                                       for c in IDENTITY}}
        if include_cases:
            cases.append({"key": e["key"], "truth": sorted(truth), "top_k": details,
                          "candidates": result["candidates"], "identities": result["identities"],
                          "neighbor_keys": [n["key"] for n in result["neighbors"]]})
    if not evaluated_keys:
        raise ValueError("No evaluable holdout vectors; encode holdout images in a separate build stage")
    implementation = {name: digest((Path(__file__).parent / name).read_bytes())
                      for name in ("evaluate.py", "recommend.py", "vectors.py", "store.py", "tag_policy.py", "__main__.py")}
    report = {
        "format": "vrs-int-evaluation/1", "snapshot": snapshot,
        "parent": memory.manifest["parent"], "contract": memory.manifest["contract"],
        "contract_sha256": digest(canonical(memory.manifest["contract"])),
        "implementation_sha256": implementation,
        "runtime": {"numpy": np.__version__, "backend": backend,
                    "python": platform.python_version(), "platform": platform.platform()},
        "config": {"split": split, "ks": list(ks), "neighbors": neighbors,
                   "ratings": sorted(set(ratings)), "rare_max": rare_max, "batch_size": batch_size,
                   "include_cases": include_cases},
        "coverage": {"selected_holdout": len(targets), "evaluated": len(evaluated_keys),
                     "evaluated_keys_sha256": digest(canonical(evaluated_keys)), "skipped": skipped,
                     "snapshot_observations": verified["observations"],
                     "train_visual_observations": len(train_all), "train_without_vectors": missing_train_vectors,
                     "eligible_train_images": len(train),
                     "excluded_or_duplicate_train_observations": len(train_all) - len(train),
                     "holdout_images_blocked": len(excluded_images),
                     "holdout_source_ids_blocked": len(excluded_ids), "train_audit_tag_coverage": audit_covered},
        "top_k": {str(k): _metrics(overall[k]) for k in ks},
        "category_top_k": {c: {str(k): _metrics(v) for k, v in values.items()}
                           for c, values in category_counts.items()},
        "identity_list_at_3": {c: _metrics(v) for c, v in identities.items()},
        "storage_preservation": {
            **_preservation(storage_original_df, storage_preserved_df, rare_max),
            "train_observations": storage_train_count, "audit_tag_coverage": storage_audit_covered,
            "scope": "All live train observations passing ratings, including vectorless/blocked/duplicate rows; "
                     "frequency here counts observations, not deduplicated retrieval images. Audit fallback applies."},
        "rare": {"definition": "1 <= eligible train image document frequency <= rare_max",
                 "train_retained_vocabulary": len(rare), "original_rare_vocabulary": len(original_rare),
                 "preserved_original_rare_vocabulary": len(rare_preserved),
                 "vocabulary_preservation": _ratio(len(rare_preserved), len(original_rare)),
                 "original_rare_occurrences": sum(original_df[t] for t in original_rare),
                 "preserved_original_rare_occurrences": sum(preserved_df[t] for t in original_rare),
                 "occurrence_preservation": _ratio(sum(preserved_df[t] for t in original_rare),
                                                   sum(original_df[t] for t in original_rare)),
                 "preservation_scope": "Audit-original tags vs stored tags in eligible vector-bearing train; "
                                       "intentional noise removals count as removals. Missing audit uses retained tags.",
                 "top_k": {str(k): _metrics(rare_counts[k]) for k in ks}},
        "unseen_top_k": {str(k): _metrics(unseen_counts[k]) for k in ks},
        "errors": {str(k): dict(errors[k]) for k in ks},
        "scope": "Stored-feature holdout retrieval only. Scores are weighted neighbor votes, not "
                 "probabilities or SWEGCA A/R/U judgments. No encoder/image/CUDA quality claim from NumPy.",
        "limitations": ["Exact hashes and source IDs do not detect unrecorded near-duplicates.",
                        "Tags absent from annotations are not proof a recommendation is semantically wrong.",
                        "Snapshot contracts do not prove the encoder was never trained on holdout images.",
                        "No correctness threshold or calibrated probability is asserted."]}
    if backend == "cuda":
        import torch
        report["runtime"].update(torch=torch.__version__, cuda=torch.version.cuda,
                                 device=torch.cuda.get_device_name())
    if include_cases:
        report["cases"] = cases
        report["per_tag"] = [{"tag": t, "train_df": retained_df[t], "categories": sorted(categories.get(t, [])),
                              "rare": t in rare, "source_train_df": original_df[t],
                              "preserved_train_occurrences": preserved_df[t],
                              "holdout_occurrences": tag_truth[t],
                              "hits": {str(k): tag_hits[k][t] for k in ks},
                              "recall": {str(k): _ratio(tag_hits[k][t], tag_truth[t]) for k in ks}}
                             for t in sorted(set(tag_truth) | rare | original_rare)]
    report["report_sha256"] = digest(canonical(report))
    return report
