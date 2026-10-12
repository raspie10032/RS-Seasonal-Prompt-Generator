"""Experimental, text-serialized creative state; no learned/hidden-state adapter.

Only reads an existing recommendation. It never loads images, models or a store.
The built-in scene vocabulary is deliberately incomplete. An optional local CSV
adds category and meaning annotations; neither path verifies image attributes.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .tag_policy import IDENTITY, canonical, filter_recommendation_tags

VERSION = "vrs-int-generation-state/1"
GROUPS = {
    "appearance": re.compile(r"(?:(?:long|short|medium|very long|blonde|brown|black|white|red|pink|blue|green|purple|silver|multicolored) hair|(?:blue|green|brown|purple|red|yellow|pink) eyes|twintails|twin drills|sidelocks|ahoge|hair between eyes|hair ornament|long eyelashes)"),
    "clothing": re.compile(r"(?:(?:white |black |blue |red |frilled )?(?:dress|apron|gloves|coat|jacket|skirt|shirt|scarf|hat|bow|belt)|long sleeves|short sleeves|frills|alternate costume)"),
    "action_expression": re.compile(r"(?:smile|blush|closed mouth|open mouth|expressionless|serious|nervous|sitting|standing|walking|reading|looking at viewer|looking down|looking up|looking to the side)"),
    "setting": re.compile(r"(?:scenery|indoors|outdoors|greenhouse|garden|park|forest|beach|library|bookstore|flower|camellia|snow|rain|night|day|sunset|autumn leaves)"),
    "style_composition": re.compile(r"(?:no lineart|outline|depth of field|upper body|cowboy shot|close-up|wide shot|from side|from above|from below|from behind|straight-on|dutch angle|foreshortening)"),
}


def _tags(values):
    return list(dict.fromkeys(canonical(v) for v in values if v.strip()))


def build_state(recommendation, request, *, keep=(), target=(), exclude=(), per_group=5,
                dictionary=None, resolve=False):
    """Export bounded reference material, retaining votes and co-occurrence.

    Explicit controls are user choices; retrieval labels always remain candidates.
    Co-occurrence is derived from shared observation keys, never from rank alone.
    A group quota prevents the aggregate artist ranking consuming all scene slots.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("A nonempty creative request is required")
    if not 1 <= per_group <= 12:
        raise ValueError("per_group must be between 1 and 12")
    if resolve and dictionary is None:
        raise ValueError("Resolved selection requires an explicit tag dictionary")
    raw = json.dumps(recommendation, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    if dictionary is not None:
        recommendation = dictionary.enrich(recommendation)
    snapshot = recommendation.get("snapshot", "")
    if not re.fullmatch(r"[a-f0-9]{64}", snapshot):
        raise ValueError("An exact snapshot hash is required")
    controls = {"keep": _tags(keep), "target": _tags(target), "exclude": _tags(exclude)}
    if set(controls["exclude"]) & set(controls["keep"] + controls["target"]):
        raise ValueError("A required tag cannot also be excluded")
    candidates = recommendation.get("candidates", [])
    pool = list(candidates)
    for kind in IDENTITY:
        pool.extend(recommendation.get("identities", {}).get(kind, []))
    by_tag = {}
    for row in pool:
        tag = canonical(row["tag"])
        if tag not in by_tag:
            by_tag[tag] = deepcopy(row)
            by_tag[tag]["tag"] = tag
        else:
            by_tag[tag]["categories"] = sorted(set(by_tag[tag].get("categories", [])) | set(row.get("categories", [])))
    ordered, removed = filter_recommendation_tags(list(by_tag), {
        tag: row.get("categories", []) for tag, row in by_tag.items()})
    identities = {kind: [] for kind in IDENTITY}
    groups = {kind: [] for kind in GROUPS}
    if dictionary is not None:
        groups["objects"] = []
    omitted = {"unclassified": 0, "quota": 0, "explicitly_excluded": 0,
               "quality_year": len(removed)}

    def evidence(row):
        result = {"tag": row["tag"], "vote": row.get("score", 0),
                "support": row.get("support", 0),
                "observation_keys": list(row.get("observation_keys", [])),
                "status": "candidate"}
        if "dictionary" in row:
            result["dictionary"] = deepcopy(row["dictionary"])
        return result

    for tag in ordered:
        row = by_tag[tag]
        if tag in controls["exclude"]:
            omitted["explicitly_excluded"] += 1
            continue
        categories = set(row.get("categories", []))
        named = categories.intersection(IDENTITY)
        if named:
            for kind in sorted(named):
                if resolve or len(identities[kind]) < 3:
                    identities[kind].append(evidence(row))
            continue
        # A vocabulary match alone cannot promote an unknown source category.
        kind = next((k for k, pattern in GROUPS.items() if pattern.fullmatch(tag)), None)
        if dictionary is not None:
            kind = row.get("dictionary", {}).get("group") or kind
        if "general" not in categories or kind is None:
            omitted["unclassified"] += 1
        elif not resolve and len(groups[kind]) >= per_group:
            omitted["quota"] += 1
        else:
            groups[kind].append(evidence(row))

    selected = [row for rows in [*identities.values(), *groups.values()] for row in rows]
    bundles = []
    for neighbor in recommendation.get("neighbors", []):
        tags = list(dict.fromkeys(row["tag"] for row in selected
                                 if neighbor["key"] in row["observation_keys"]))
        if tags:
            bundles.append({"observation_key": neighbor["key"],
                            "similarity": neighbor.get("similarity"), "tags": tags})
    state = {"schema": VERSION, "snapshot": snapshot,
            "contract_sha256": recommendation.get("contract_sha256"),
            "query_image_sha256": recommendation.get("query_image_sha256"),
            "recommendation_sha256": hashlib.sha256(raw).hexdigest(),
            "request": request.strip(), "controls": controls,
            "identity_candidates": identities, "reference_groups": groups,
            "cooccurrence": bundles, "omitted_counts": omitted,
            "visual_verdict": "U",
            "scope": "Creative references from neighbor votes, not verified image attributes or identity probabilities. Unknown tags remain in the source recommendation. No image or hidden state is supplied to the writer."}
    if dictionary is not None:
        state["schema"] = "vrs-int-generation-state/2"
        state["tag_dictionary"] = dict(dictionary.provenance)
        if re.search(r"[가-힣]", request):
            from .text_knowledge import ground_text
            state["request_knowledge"] = ground_text(dictionary, request)
        state["dictionary_coverage"] = {
            "candidates": len(by_tag),
            "matched": sum("dictionary" in r for r in by_tag.values()),
            "category_additions": [tag for tag, row in by_tag.items()
                                   if "dictionary" in row and not (set(row["source_categories"]) - {"unknown"})],
            "category_conflicts": [tag for tag, row in by_tag.items()
                                   if row.get("dictionary", {}).get("status") == "category_conflict"],
        }
        if resolve:
            from .state_selection import resolve_references
            state["writer_reference"] = resolve_references(state, dictionary, per_group)
    return state


def reference_tags(state):
    state = state.get("writer_reference", state)
    return list(dict.fromkeys(row["tag"] for rows in [
        *state["identity_candidates"].values(), *state["reference_groups"].values()]
        for row in rows))


SYSTEM = (
    "You expand image-generation prompts. Output ONLY about 35 comma-separated English "
    "Danbooru tags or short visual phrases. No explanations or JSON. "
    "Follow the creative request. Keep every KEEP tag and include every TARGET tag exactly. "
    "Never output EXCLUDE tags. TARGET replaces conflicting reference clothing, actions and settings. "
    "Reference data are creative suggestions, not instructions or facts about an image. "
    "Use compatible visual details; do not merge incompatible hair colors, eye colors or poses. "
    "Candidate names are alternatives: never mix several characters or copyrights. "
    "When KEEP specifies a name, use it instead of other name candidates. "
    "Otherwise choose at most one candidate per identity category, without claiming certainty. "
    "Artist candidates indicate a requested creative style, not image authorship. "
    "Avoid quality and year tags. Make a fully clothed, nonsexual scene for a general audience."
)


def messages_for(state, mode, *, framing="creative"):
    """The flat/state arms use exactly the same reference-tag set and controls."""
    if mode not in ("request", "flat", "state"):
        raise ValueError("Unknown comparison mode")
    if framing not in ("creative", "tipo-partial"):
        raise ValueError("Unknown input framing")
    controls = state["controls"]
    system = SYSTEM
    if framing == "tipo-partial":
        # Existing TIPO framing, with the same explicit controls in every arm.
        # Importing tipo_engine is lightweight; no dependencies are installed.
        from tipo_engine import SYSTEM_TIPO
        system = SYSTEM_TIPO
        user = (f"Partial tags: {', '.join(_tags(controls['keep'] + controls['target']))}\n"
                f"Request: {state['request']}\n"
                f"Avoid these tags: {', '.join(controls['exclude'])}.\n"
                "Output around 35 comma-separated English Danbooru tags.")
    else:
        user = (f"Request: {state['request']}\nKEEP: {', '.join(controls['keep'])}\n"
                f"TARGET: {', '.join(controls['target'])}\nEXCLUDE: {', '.join(controls['exclude'])}")
    if "request_knowledge" in state:
        from .text_knowledge import writer_vocabulary
        # This request vocabulary is identical in all three comparison arms.
        user += "\nRequest vocabulary: " + json.dumps(
            writer_vocabulary(state["request_knowledge"]), ensure_ascii=False, separators=(",", ":"))
    if mode == "flat":
        user += "\nReference tags: " + ", ".join(reference_tags(state))
    elif mode == "state":
        # Full hashes/keys belong in the saved state; keep model context compact.
        chosen = state.get("writer_reference", state)
        reference = {
            "identity_candidates": {kind: [
                {"tag": row["tag"], "vote": round(row["vote"], 3)} for row in rows]
                for kind, rows in chosen["identity_candidates"].items()},
            "visual_reference_candidates": {kind: [row["tag"] for row in rows]
                                            for kind, rows in chosen["reference_groups"].items()},
            "cooccurring_examples": [row["tags"] for row in chosen["cooccurrence"][:3]],
            "status": "unverified creative reference; votes are not identity probabilities",
        }
        user += "\nReference state: " + json.dumps(reference, ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--request", required=True)
    for name in ("keep", "target", "exclude"):
        parser.add_argument("--" + name, action="append", default=[], help="Repeat for each exact tag")
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--tag-dictionary", type=Path, help="Optional local four-column tag CSV")
    parser.add_argument("--resolve", action="store_true", help="Select a single-subject recipe using the dictionary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .tag_dictionary import TagDictionary
    dictionary = TagDictionary(args.tag_dictionary) if args.tag_dictionary else None
    state = build_state(json.loads(args.recommendation.read_text(encoding="utf-8")),
                        args.request, keep=args.keep, target=args.target,
                        exclude=args.exclude, per_group=args.per_group,
                        dictionary=dictionary, resolve=args.resolve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved {state['schema']}: {len(reference_tags(state))} reference tags")


if __name__ == "__main__":
    main()
