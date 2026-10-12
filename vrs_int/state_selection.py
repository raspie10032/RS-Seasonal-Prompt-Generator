"""Conservative single-subject recipe selection, separate from stored evidence."""
import re

from .tag_policy import IDENTITY

VERSION = "single-subject-reference-selection/1"
COLORS = "black|blonde|blue|brown|green|grey|gray|orange|pink|purple|red|silver|white|yellow|aqua"


def exclusive_slot(tag):
    for part in ("hair", "eyes"):
        if re.fullmatch(rf"(?:{COLORS}) {part}", tag):
            return part + "_color"
    for slot, values in {
        "season": ("spring", "summer", "autumn", "winter"),
        "environment": ("indoors", "outdoors"),
        "time": ("day", "night", "sunrise", "sunset"),
        "pose": ("standing", "sitting", "lying", "walking", "running"),
        "framing": ("close-up", "upper body", "cowboy shot", "full body", "wide shot"),
    }.items():
        if tag in values:
            return slot
    return None


def resolve_references(state, dictionary, per_group):
    controls = state["controls"]
    required = list(dict.fromkeys(controls["keep"] + controls["target"]))
    if any(re.fullmatch(r"(?:[2-9]|\d{2,})(?:girls|boys|others)|multiple (?:girls|boys|others)", t)
           for t in required):
        raise ValueError("Resolved selection is a single-subject recipe; use candidate mode for multiple subjects")
    relaxed = set()
    if "heterochromia" in required:
        relaxed.add("eyes_color")
    if set(required) & {"multicolored hair", "two-tone hair", "gradient hair"}:
        relaxed.add("hair_color")
    occupied = {}
    for tag in required:
        slot = exclusive_slot(tag)
        if not slot or slot in relaxed:
            continue
        if slot in occupied and occupied[slot] != tag:
            raise ValueError(f"Conflicting single-subject controls: {occupied[slot]} / {tag}")
        occupied[slot] = tag

    # Prefer an example supporting the chosen character, then other KEEP traits.
    character_names = {r["tag"] for r in state["identity_candidates"].get("character", [])}
    def anchor_score(bundle):
        tags = set(bundle["tags"])
        return (len(tags & set(controls["keep"]) & character_names),
                len(tags & set(controls["keep"])))
    anchor = max(state["cooccurrence"], key=anchor_score, default=None)
    anchor_key = anchor["observation_key"] if anchor else None
    replacement_groups = set()
    for tag in controls["target"]:
        entry = dictionary.lookup(tag)
        if entry and entry.group in ("clothing", "action_expression", "setting"):
            replacement_groups.add(entry.group)
    identities = {kind: [] for kind in IDENTITY}
    groups = {kind: [] for kind in state["reference_groups"]}
    rejected = []
    # Alternative names stay available for inspection but never become chosen
    # identities merely because they are popular neighbor labels.
    for kind, rows in state["identity_candidates"].items():
        for row in rows:
            if row["tag"] in required:
                identities[kind].append(row)
            else:
                rejected.append({"tag": row["tag"], "reason": "identity_not_selected"})
    for kind, rows in state["reference_groups"].items():
        if kind == "appearance":
            # Keep distinguishing hair/eye traits ahead of generic body labels.
            # Python's stable sort preserves neighbor-vote order within a tier.
            rows = sorted(rows, key=lambda row: (
                0 if exclusive_slot(row["tag"]) else
                1 if re.search(r"(?:hair|eyes|twintails|sidelocks|ahoge)", row["tag"]) else 2))
        for row in rows:
            tag = row["tag"]
            reason = None
            slot = exclusive_slot(tag)
            if tag in required:
                reason = "already_in_controls"
            elif anchor_key and anchor_key not in row["observation_keys"]:
                reason = "different_reference_example"
            elif kind in replacement_groups:
                reason = "group_replaced_by_target"
            elif slot and slot not in relaxed and slot in occupied and occupied[slot] != tag:
                reason = "conflicting_attribute"
            elif len(groups[kind]) >= per_group:
                reason = "group_quota"
            if reason:
                rejected.append({"tag": tag, "reason": reason})
                continue
            groups[kind].append(row)
            if slot and slot not in relaxed:
                occupied[slot] = tag
    selected = {r["tag"] for rows in [*identities.values(), *groups.values()] for r in rows}
    bundles = [{**b, "tags": [t for t in b["tags"] if t in selected]}
               for b in state["cooccurrence"] if b["observation_key"] == anchor_key]
    return {"policy": VERSION, "identity_candidates": identities, "reference_groups": groups,
            "cooccurrence": [b for b in bundles if b["tags"]],
            "anchor_observation_key": anchor_key, "replaced_groups": sorted(replacement_groups),
            "rejected": rejected,
            "scope": "Single-subject creative recipe; selected names require explicit controls. "
                     "Co-occurrence and a few exact conflict rules do not prove visual consistency."}
