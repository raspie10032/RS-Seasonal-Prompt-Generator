"""Reviewed synthetic instruction tasks for a Gemma 4 TIPO pilot.

These are constructed prompt-writing targets, not captions or human preferences.
Scene families are split BEFORE variations, and no VRS store is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from vrs_int.generation_state import messages_for
from vrs_int.state_selection import exclusive_slot
from vrs_int.tag_policy import canonical, filter_recommendation_tags
from tipo_prompt import row_messages, instruction_messages

SCENES = (
    ("spring_garden", "봄 정원에서 서서 꽃을 바라보는 장면", ["spring", "garden", "standing", "flower", "outdoors"], ["winter", "indoors", "sitting"]),
    ("summer_beach", "여름 해변에서 산책하는 장면", ["summer", "beach", "walking", "ocean", "outdoors"], ["winter", "indoors", "sitting"]),
    ("library_reading", "도서관 안에 앉아 책을 읽는 장면", ["library", "indoors", "sitting", "reading", "book"], ["outdoors", "running", "standing"]),
    ("rain_city", "비 오는 도시에서 우산을 들고 서 있는 장면", ["rain", "city", "umbrella", "standing", "outdoors"], ["indoors", "sitting", "snow"]),
    ("snow_forest", "겨울 숲에서 코트와 목도리를 두르고 걷는 장면", ["winter", "forest", "snow", "coat", "scarf", "walking", "outdoors"], ["summer", "indoors", "sitting"]),
    ("cafe_tea", "카페 안에서 앉아 차를 마시는 장면", ["cafe", "indoors", "sitting", "tea", "cup"], ["outdoors", "running", "standing"]),
    ("autumn_park", "가을 공원에서 산책하는 장면", ["autumn", "park", "walking", "autumn leaves", "outdoors"], ["spring", "indoors", "sitting"]),
    ("winter_market", "겨울 야시장에서 코트를 입고 컵을 들고 서 있는 장면", ["winter", "night market", "night", "coat", "holding cup", "standing", "outdoors"], ["scarf", "summer", "indoors", "sitting"]),
    ("rain_bookstore", "비 오는 날 우산을 들고 책방 밖에 서 있는 장면", ["rain", "umbrella", "bookstore", "standing", "outdoors"], ["scarf", "snow", "sitting", "indoors"]),
)
HAIR = (("금발", "blonde hair"), ("빨간 머리", "red hair"), ("검은 머리", "black hair"), ("은발", "silver hair"))
EYES = (("파란 눈", "blue eyes"), ("초록 눈", "green eyes"), ("갈색 눈", "brown eyes"))
NAMES = ("sample traveler", "sample painter", "sample gardener", "sample singer")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def output_tags(raw):
    return list(dict.fromkeys(canonical(t.strip().strip('`')) for t in raw.split(',') if t.strip()))


def score_output(raw, required, excluded):
    tags = output_tags(raw)
    slots = {}
    for tag in tags:
        slot = exclusive_slot(tag)
        if slot:
            slots.setdefault(slot, set()).add(tag)
    _, filtered = filter_recommendation_tags(tags, {})
    missing = [t for t in required if t not in tags]
    forbidden = [t for t in excluded if t in tags]
    conflicts = {s: sorted(v) for s, v in slots.items() if len(v) > 1}
    return {"tags": tags, "missing": missing, "forbidden": forbidden,
            "conflicts": conflicts, "quality_year": filtered,
            "pass": not (missing or forbidden or conflicts or filtered)}


def validate_row(row):
    messages = row_messages(row)
    expected = ["user", "assistant"] if row.get("prompt_contract") else ["system", "user", "assistant"]
    if [m["role"] for m in messages] != expected:
        raise ValueError("Unexpected instruction/assistant exchange")
    if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
        raise ValueError("Empty or non-text message")
    if set(row["required"]) & set(row["excluded"]):
        raise ValueError("Contradictory supervision")
    if not score_output(messages[-1]["content"], row["required"], row["excluded"])["pass"]:
        raise ValueError("Target violates its annotated constraints")


def validate_splits(train, evaluation):
    for row in [*train, *evaluation]:
        validate_row(row)
    if {r["family"] for r in train} & {r["family"] for r in evaluation}:
        raise ValueError("Scene-family leakage")
    if {digest(instruction_messages(r)) for r in train} & {digest(instruction_messages(r)) for r in evaluation}:
        raise ValueError("Prompt leakage")


def make_rows():
    train, evaluation = [], []
    for scene_index, (family, scene_ko, target, excluded) in enumerate(SCENES):
        for color_index, (hair_ko, hair) in enumerate(HAIR):
            for eye_index, (eyes_ko, eyes) in enumerate(EYES):
                for mode in ("request", "flat", "state"):
                    # Fictional names require exact copying, not prior model knowledge.
                    name = NAMES[(scene_index + color_index + eye_index) % len(NAMES)]
                    artist, series = "sample creator", "sample tales"
                    keep = [name, artist, series, hair, eyes]
                    wrong_hair = "blue hair" if hair != "blue hair" else "pink hair"
                    wrong_eyes = "red eyes"
                    avoid = list(dict.fromkeys([*excluded, wrong_hair, wrong_eyes, "alternative character"]))
                    request = (f"{name}, {artist}, {series} 이름은 그대로 유지하고 {hair_ko}, {eyes_ko}도 유지해줘. "
                               f"{scene_ko}으로 바꿔줘. {', '.join(excluded)}는 빼줘.")
                    refs = {"appearance": [hair, eyes, wrong_hair, wrong_eyes, "long hair"],
                            "clothing": ["scarf"], "setting": ["indoors", "summer"],
                            "action_expression": ["sitting", "smile"], "style_composition": ["full body"]}
                    def candidate(tag):
                        return {"tag": tag, "vote": 0.5}
                    state = {"request": request, "controls": {"keep": keep, "target": target, "exclude": avoid},
                             "identity_candidates": {"character": [candidate(name), candidate("alternative character")],
                                                     "artist": [candidate(artist)], "copyright": [candidate(series)]},
                             "reference_groups": {k: [candidate(t) for t in v] for k, v in refs.items()},
                             "cooccurrence": [{"tags": [hair, eyes, "long hair", "smile"]}]}
                    required = list(dict.fromkeys([*keep, *target]))
                    answer = list(dict.fromkeys(["1girl", "solo", *required, "long hair", "smile", "full body"]))
                    row = {"family": family, "mode": mode, "source": "constructed instruction target; not an image caption",
                           "required": required, "excluded": avoid,
                           "messages": [*messages_for(state, mode), {"role": "assistant", "content": ", ".join(answer)}]}
                    (evaluation if scene_index >= 6 else train).append(row)
    # Dedicated Korean-only negation pairs. Entire pair families stay in one split.
    for group_index, (noun, tag) in enumerate((("목도리", "scarf"), ("모자", "hat"), ("장갑", "gloves"), ("안경", "glasses"))):
        for hair_ko, hair in HAIR:
            for positive in (False, True):
                request = f"{hair_ko} 인물. {noun}는 {'넣어줘' if positive else '빼줘'}."
                state = {"request": request, "controls": {"keep": [], "target": [], "exclude": []}}
                messages = messages_for(state, "request")
                # Explicit bilingual data, same payload for both signs. Role comes from Korean.
                messages[-1]["content"] += "\nVocabulary (meanings only): " + json.dumps({hair_ko: hair, noun: tag}, ensure_ascii=False)
                required = [hair] + ([tag] if positive else [])
                excluded = [] if positive else [tag]
                row = {"family": "korean_" + tag, "mode": "korean_only", "source": "constructed paired Korean instruction",
                       "required": required, "excluded": excluded,
                       "messages": [*messages, {"role": "assistant", "content": ", ".join(["1girl", "solo", *required, "standing", "long hair", "smile"])}]}
                (evaluation if group_index == 3 else train).append(row)
    validate_splits(train, evaluation)
    random.Random(42).shuffle(train)
    return train, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train, evaluation = make_rows()
    manifest = {"schema": "gemma4-tipo-pilot-data/1", "scope": "Synthetic instruction pilot; no image captions, private store or preference labels.", "splits": {}}
    for name, rows in (("train", train), ("eval", evaluation)):
        path = args.output / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "families": sorted({r["family"] for r in rows})}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
