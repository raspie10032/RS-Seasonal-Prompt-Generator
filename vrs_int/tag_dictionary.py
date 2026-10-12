"""Optional local tag semantics. Dictionary descriptions are data, not instructions.

Reads the headerless tag, category ID, post count, Korean description CSV format.
No network, image model, sentence translation, or memory mutation is involved.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re

from .tag_policy import canonical

VERSION = "vrs-int-tag-dictionary/1"
CATEGORIES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}


def semantic_group(path):
    """Normalize common taxonomy variants; unmapped paths stay unknown."""
    if not path:
        return None
    root, *rest = path
    detail = " ".join(rest)
    if root in ("행위", "행동", "자세", "포즈", "표정") or (
        root in ("신체", "표현") and any(s in detail for s in ("자세", "표정", "움직임"))
    ):
        return "action_expression"
    if root in ("패션", "의상", "복장"):
        return "appearance" if any(s in detail for s in ("헤어", "머리카락")) else "clothing"
    if root in ("인물", "신체", "체형", "특징") or path == ("캐릭터", "유형"):
        return "appearance"
    if root in ("배경", "건물", "장소", "자연", "식물", "날씨"):
        return "setting"
    if root in ("구도", "구성", "스타일", "효과", "연출", "미디어"):
        return "style_composition"
    if root in ("사물", "소품", "음식", "동물", "무기"):
        return "objects"
    return None


@dataclass(frozen=True)
class TagEntry:
    tag: str
    category: str
    post_count: int
    description_ko: str
    semantic_path: tuple
    keywords_ko: tuple

    @property
    def group(self):
        return semantic_group(self.semantic_path) if self.category == "general" else None


class TagDictionary:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.entries = {}
        for number, row in enumerate(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), 1):
            if len(row) != 4:
                raise ValueError(f"Dictionary row {number}: expected four columns (no header)")
            tag, category, count, description = row
            tag = canonical(tag)
            try:
                category, count = CATEGORIES[int(category)], int(count)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Dictionary row {number}: invalid category/count") from exc
            if not tag or count < 0 or tag in self.entries:
                raise ValueError(f"Dictionary row {number}: empty/duplicate tag or negative count")
            match = re.match(r"\s*\[([^\]]+)\]", description)
            path_parts = tuple(p.strip() for p in match[1].split(">") if p.strip()) if match else ()
            keyword_text = description.partition("키워드:")[2]
            # Angle-bracket terms denote a broad class, not a translation alias.
            aliases = tuple(dict.fromkeys(canonical(s.strip().strip("[]"))
                            for s in re.sub(r"<[^>]*>", "", keyword_text).split(",") if s.strip()))
            self.entries[tag] = TagEntry(tag, category, count, description, path_parts, aliases)
        if not self.entries:
            raise ValueError("Empty tag dictionary")
        self.provenance = {"schema": VERSION, "sha256": hashlib.sha256(raw).hexdigest(),
                           "bytes": len(raw), "rows": len(self.entries),
                           "categories": dict(Counter(e.category for e in self.entries.values()))}

    def lookup(self, tag):
        return self.entries.get(canonical(tag))

    def search(self, text, limit=10):
        """Exact English/alias first, then substring; returns alternatives, not a decision."""
        query = canonical(text)
        if not query or not 1 <= limit <= 100:
            raise ValueError("Nonempty query and limit in 1..100 required")
        matches = []
        for entry in self.entries.values():
            exact = query == entry.tag or query in entry.keywords_ko
            if exact or query in entry.tag or query in canonical(entry.description_ko):
                matches.append((not exact, -entry.post_count, entry.tag, entry))
        return [dict(asdict(e), match="exact" if not rank else "substring")
                for rank, _, _, e in sorted(matches)[:limit]]

    def annotate(self, row):
        result = deepcopy(row)
        entry = self.lookup(row["tag"])
        if entry is None:
            return result
        source = sorted(set(row.get("source_categories", row.get("categories", []))))
        known = set(source) - {"unknown"}
        conflict = bool(known and known != {entry.category})
        result["source_categories"] = source
        result["categories"] = sorted(known) if known else [entry.category]
        result["dictionary"] = {"category": entry.category, "semantic_path": list(entry.semantic_path),
                                "description_ko": entry.description_ko,
                                "status": "category_conflict" if conflict else "matched",
                                "group": None if conflict else entry.group}
        return result

    def enrich(self, recommendation):
        """Annotate saved candidates without changing votes, neighbors or source input."""
        prior = recommendation.get("tag_dictionary", {})
        if prior and prior.get("sha256") != self.provenance["sha256"]:
            raise ValueError("Recommendation already uses a different dictionary; start from its source")
        result = deepcopy(recommendation)
        result["candidates"] = [self.annotate(row) for row in result.get("candidates", [])]
        result["identities"] = {kind: [self.annotate(row) for row in rows]
                                for kind, rows in result.get("identities", {}).items()}
        result["tag_dictionary"] = dict(self.provenance)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--query", help="English tag or Korean keyword; omitted prints dictionary summary")
    query.add_argument("--text", help="Find bilingual term candidates in a sentence; no image memory needed")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    dictionary = TagDictionary(args.csv)
    if args.text is not None:
        from .text_knowledge import ground_text
        print(json.dumps(ground_text(dictionary, args.text, limit=args.limit), ensure_ascii=False, indent=2))
        return
    print(json.dumps({"dictionary": dictionary.provenance,
                      "matches": dictionary.search(args.query, args.limit) if args.query else []},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
