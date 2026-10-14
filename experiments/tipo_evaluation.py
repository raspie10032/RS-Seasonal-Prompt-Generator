"""Freeze private TIPO evaluation inputs and score constraints, never creativity.

No inference, downloads or training. Expected tags are user requirements, not
image-caption ground truth. Save private JSONL files outside the checkout.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from tipo_prompt import VERSION as PROMPT_CONTRACT
from vrs_int.tag_policy import canonical, filter_recommendation_tags

VERSION = "tipo-evaluation/1"
TASKS = ("partial_tags", "korean", "edit", "image", "vrs")
HASH = re.compile(r"[a-f0-9]{64}\Z")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def source_keys(row):
    source = row["source"]
    return {(key, source[key]) for key in ("image_sha256", "intent_sha256", "family_id")
            if source.get(key)}


def validate(rows):
    if not rows:
        raise ValueError("Empty evaluation")
    ids, seen_sources, groups = set(), {}, {}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(row["id"])
        if row.get("purpose") != "evaluation" or row.get("prompt_contract") != PROMPT_CONTRACT:
            raise ValueError("Expected versioned evaluation-only case")
        if row["task"] not in TASKS or row["split"] not in ("dev", "final"):
            raise ValueError("Unknown task/split")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("Source family crosses evaluation splits")
        if not row["input"]["prompt"].strip() or not row["required"]:
            raise ValueError("Empty request or constraints")
        required = {canonical(t) for t in row["required"]}
        excluded = {canonical(t) for t in row["excluded"]}
        if required & excluded:
            raise ValueError("Required and excluded overlap")
        if not set(row.get("protected_names", {})) <= required:
            raise ValueError("Protected identity missing from required tags")
        if (row["task"] == "image") != bool(row["input"].get("image")):
            raise ValueError("Only image task carries pixels")
        if row["task"] == "vrs" and not row.get("reference", {}).get("sha256"):
            raise ValueError("VRS reference must have provenance")
        if any(key in row for key in ("answer", "messages")):
            raise ValueError("Evaluation contains a training target or hidden template")
        for field in ("image_sha256", "intent_sha256"):
            if not HASH.fullmatch(row["source"][field]):
                raise ValueError("Invalid source digest")
        for key in source_keys(row):
            if seen_sources.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Source or generation intent crosses evaluation splits")
    return {"cases": len(rows), "groups": len(groups),
            "tasks": dict(Counter(r["task"] for r in rows)),
            "splits": dict(Counter(r["split"] for r in rows))}


def assert_training_disjoint(training, reserved):
    """Future training must carry source provenance; missing evidence fails."""
    blocked = set().union(*(source_keys(r) for r in reserved))
    hashes = {r["source"].get("dhash64") for r in reserved} - {None}
    groups = {r["group_id"] for r in reserved}
    for row in training:
        if row.get("purpose") == "evaluation" or row.get("group_id") in groups:
            raise ValueError("Evaluation row/family used for training")
        source = row.get("source")
        if not isinstance(source, dict) or not all(source.get(k) for k in
                ("image_sha256", "intent_sha256", "family_id", "dhash64")):
            raise ValueError("Training row lacks source and duplicate-screening provenance")
        if source_keys(row) & blocked:
            raise ValueError("Training source overlaps reserved evaluation")
        if any((int(source["dhash64"], 16) ^ int(h, 16)).bit_count() <= 6 for h in hashes):
            raise ValueError("Possible near-duplicate of reserved image; review required")


def score(raw, row):
    """Keep raw repetitions/format errors visible; aliases are case-specific."""
    tags = [canonical(t) for t in raw.split(",") if t.strip()]
    unique = set(tags)
    aliases = row.get("aliases", {})
    present = lambda tag: bool(unique & {canonical(t) for t in [tag, *aliases.get(tag, [])]})
    missing = [tag for tag in row["required"] if not present(tag)]
    forbidden = [tag for tag in row["excluded"] if present(tag)]
    names = row.get("protected_names", {})
    _, quality = filter_recommendation_tags(tags, {tag: [kind] for tag, kind in names.items()})
    conflicts = [values for values in row.get("exclusive_sets", [])
                 if sum(present(tag) for tag in values) > 1]
    duplicates = {tag: count for tag, count in Counter(tags).items() if count > 1}
    format_problem = bool(re.search(r"```|<\|?|^\s*(?:tags|prompt)\s*:|^\s*[-*]\s", raw, re.I))
    length_ok = row.get("min_tags", 10) <= len(unique) <= row.get("max_tags", 80)
    given = {canonical(t) for t in row.get("given_tags", [])}
    return {"missing": missing, "forbidden": forbidden, "quality_year": quality,
            "conflicts": conflicts, "duplicate_tags": duplicates, "format_problem": format_problem,
            "unique_tags": len(unique), "length_ok": length_ok,
            "new_tag_count": len(unique - given),
            "new_tag_count_is_not_creative_quality": True,
            "constraint_pass": bool(tags) and not any((missing, forbidden, quality, conflicts,
                                                        duplicates, format_problem, not length_ok)),
            "creative_quality": None, "visual_fidelity": None}


def score_results(rows, outputs):
    by_id = {r["id"]: r for r in rows}
    seen, results = set(), []
    for output in outputs:
        key = output["id"]
        if key not in by_id or key in seen:
            raise ValueError("Unknown or repeated result ID")
        seen.add(key)
        result = {"id": key, "task": by_id[key]["task"],
                  "raw": score(output["raw"], by_id[key]),
                  "postprocessed": score(output["postprocessed"], by_id[key])
                  if "postprocessed" in output else None}
        if output.get("truncated"):
            result["raw"]["constraint_pass"] = False
            result["raw"]["truncated"] = True
            if result["postprocessed"] is not None:
                result["postprocessed"]["constraint_pass"] = False
                result["postprocessed"]["truncated"] = True
        results.append(result)
    return {"evaluated": len(results), "expected": len(rows), "complete": seen == set(by_id),
            "raw_constraint_passes": sum(r["raw"]["constraint_pass"] for r in results),
            "creative_quality_review": "pending; never inferred from tag counts or condition scores",
            "results": results}


def freeze(rows, output):
    summary = validate(rows)
    # Compare the selected sources, not every image in the user's collection.
    sources = {r["source"]["image_sha256"]: r for r in rows}
    sources = list(sources.values())
    for i, left in enumerate(sources):
        for right in sources[i + 1:]:
            a, b = left["source"].get("dhash64"), right["source"].get("dhash64")
            if a and b and left["split"] != right["split"] and (int(a, 16) ^ int(b, 16)).bit_count() <= 6:
                raise ValueError("Possible near-duplicate crosses evaluation splits")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in ("dev", "final"):
        raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                      for row in rows if row["split"] == split).encode()
        (output / (split + ".jsonl")).write_bytes(raw)
        files[split + ".jsonl"] = hashlib.sha256(raw).hexdigest()
    manifest = {"schema": VERSION, "prompt_contract": PROMPT_CONTRACT,
                **summary, "files": files, "corpus_sha256": digest(rows),
                "review_status": "authored inputs; human preference and visual review pending",
                "scope": "Creative expansion evaluation, not image tagging or identity accuracy. "
                         "Generation intent is not pixel ground truth. Exact/intent/dHash screens "
                         "are bounded heuristics, not a proof against every crop or edit."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("freeze")
    build.add_argument("--draft", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("score")
    check.add_argument("--cases", type=Path, required=True)
    check.add_argument("--outputs", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(read_rows(args.draft), args.output), ensure_ascii=False, indent=2))
    else:
        rows = read_rows(args.cases)
        validate(rows)
        report = score_results(rows, read_rows(args.outputs))
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
