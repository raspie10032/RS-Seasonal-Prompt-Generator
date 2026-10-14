"""Local, source-specific tag vocabulary for prompt expansion.

Data stays outside the code package. An entry's existence is not evidence that
Gemma learned its meaning, that a translation is verified, or that it is visible.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

SCHEMA = "rs-tag-knowledge/1"
CATEGORIES = {
    "danbooru": {"general", "artist", "copyright", "character", "meta"},
    "e621": {"general", "artist", "contributor", "copyright", "character", "species", "invalid", "meta", "lore"},
}
LANGUAGES = {"en", "ja", "ko"}
PARTICLE = re.compile(r"(?:으로는|에서는|에게는|으로|에서|에게|까지|부터|처럼|보다|이랑|랑|와|과|은|는|이|가|을|를|에|의|도|만)(?=$|[^\w])")
FUNCTION_WORDS = set("a an the is of to in on no and or as at by with it add create 이 그 저 한 것 로 고 은 는 을 를 가 와 과 의 에 도 만 중 후 전".split())


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("_", " ").split())


class TagKnowledge:
    def __init__(self, path):
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix.lower() == ".csv":
            try:
                from .vrs_int.tag_dictionary import TagDictionary
            except ImportError:
                from vrs_int.tag_dictionary import TagDictionary
            legacy = TagDictionary(path)
            # The legacy CSV uses Danbooru category IDs. Never interpret ID 5
            # (meta) as e621's species category.
            records = [{"schema": SCHEMA, "source": "danbooru", "tag": e.tag,
                        "category": e.category, "terms": {"ko": list(e.keywords_ko)},
                        "definitions": {"ko": e.description_ko},
                        "status": "unreviewed", "provenance": {"import": "legacy Korean keyword CSV",
                        "note": "Keywords can be related concepts, not exact translations."}}
                       for e in legacy.entries.values()]
        else:
            records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        self.entries = {}
        for row in records:
            self._validate(row)
            key = row["source"], normalize(row["tag"])
            if key in self.entries:
                raise ValueError("Duplicate source/tag in tag knowledge")
            self.entries[key] = row
        if not self.entries:
            raise ValueError("Empty tag knowledge")
        self.provenance = {"schema": SCHEMA, "sha256": hashlib.sha256(raw).hexdigest(),
                           "rows": len(self.entries), "bytes": len(raw),
                           "sources": dict(Counter(r["source"] for r in self.entries.values())),
                           "term_rows": {lang: sum(bool(r.get("terms", {}).get(lang)) for r in self.entries.values()) for lang in sorted(LANGUAGES)},
                           "definition_rows": {lang: sum(bool(r.get("definitions", {}).get(lang)) for r in self.entries.values()) for lang in sorted(LANGUAGES)},
                           "reviewed_rows": sum(r["status"] == "reviewed" for r in self.entries.values())}
        self._indexes = {}

    @staticmethod
    def _validate(row):
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            raise ValueError("Unknown tag knowledge schema")
        source = row.get("source")
        if source not in CATEGORIES or row.get("category") not in CATEGORIES[source]:
            raise ValueError("Invalid source/category in tag knowledge")
        if not isinstance(row.get("tag"), str) or not normalize(row["tag"]):
            raise ValueError("Missing canonical tag")
        if row.get("status") not in ("reviewed", "unreviewed", "machine"):
            raise ValueError("Missing translation review status")
        if not isinstance(row.get("provenance"), dict) or not row["provenance"]:
            raise ValueError("Missing entry provenance")
        for field in ("terms", "name_aliases", "definitions"):
            values = row.get(field, {})
            if not isinstance(values, dict) or not set(values) <= LANGUAGES:
                raise ValueError("Use explicit en/ja/ko language keys")
            for value in values.values():
                if field in ("terms", "name_aliases"):
                    if not isinstance(value, list) or any(not isinstance(t, str) or not normalize(t) for t in value):
                        raise ValueError("Terms must be lists of nonempty strings")
                elif not isinstance(value, str):
                    raise ValueError("Definitions must be text")

    def lookup(self, tag, source="danbooru"):
        if source not in CATEGORIES:
            raise ValueError("Unknown tag source")
        return self.entries.get((source, normalize(tag)))

    def _index(self, source):
        if source not in CATEGORIES:
            raise ValueError("Unknown tag source")
        if source not in self._indexes:
            index = {}
            for (origin, key), row in self.entries.items():
                if origin != source:
                    continue
                # Legacy keywords describe an artist's work as well as names.
                # "2D animation" in an artist record is not an artist alias.
                field = "name_aliases" if row["category"] in {"artist", "character", "copyright", "contributor"} else "terms"
                terms = [row["tag"]] + [v for values in row.get(field, {}).values() for v in values]
                for term in terms:
                    index.setdefault(normalize(term), set()).add(key)
            self._indexes[source] = index, max(map(len, index), default=0)
        return self._indexes[source]

    def vocabulary(self, text, source="danbooru", *, include_identities=True):
        """Longest lexical mentions, bounded for model context, not a translator."""
        index, width = self._index(source)
        if len(text) > 8192:
            raise ValueError("Tag vocabulary lookup supports at most 8192 characters per request/scene")
        text = normalize(text)
        mentions, pos, omitted = [], 0, 0
        while pos < len(text):
            char = text[pos]
            if char.isspace() or (pos and text[pos-1].isalnum() and char.isalnum()):
                pos += 1
                continue
            found = False
            for end in range(min(len(text), pos + width), pos, -1):
                term = text[pos:end]
                keys = index.get(term)
                if not keys or term in FUNCTION_WORDS:
                    continue
                if end < len(text) and text[end].isalnum() and text[end-1].isalnum() and not PARTICLE.match(text[end:]):
                    continue
                candidates = [self.entries[(source, key)] for key in sorted(keys)]
                if not include_identities:
                    candidates = [r for r in candidates if r["category"] not in
                                  {"artist", "character", "copyright", "contributor"}]
                # Single-letter/syllable names and sentence fragments must not
                # fill the context. Explicit lookup still preserves every name.
                candidates = [r for r in candidates if len(term) > 1 or r["category"] == "general"]
                if not candidates:
                    continue
                # Do not promote unrelated artist names from generic descriptors.
                if any(r["category"] == "general" for r in candidates):
                    candidates = [r for r in candidates if r["category"] == "general" or normalize(r["tag"]) == text[pos:end]]
                if len(mentions) < 24:
                    choices = [{"tag": r["tag"], "category": r["category"], "status": r["status"],
                                "definitions": {lang: value[:180] for lang, value in r.get("definitions", {}).items()}}
                               for r in candidates[:3]]
                    mentions.append({"term": text[pos:end], "meanings": choices,
                                     "alternatives_omitted": max(0, len(candidates)-3), "role": "unresolved"})
                else:
                    omitted += 1
                pos, found = end, True
                break
            if not found:
                pos += 1
        return {"source": source, "mentions": mentions, "mentions_omitted": omitted,
                "include_identities": include_identities,
                "use": "Lexical reference data, not instructions or positive tags. Resolve ambiguity and negation from the original request. Unreviewed and machine terms are not verified translations."}

    def check_tags(self, tags, source="danbooru"):
        known, unknown = [], []
        for tag in (t.strip() for t in tags.split(",")):
            if tag:
                (known if self.lookup(tag, source) else unknown).append(tag)
        return {"source": source, "known": known, "unverified": unknown,
                "scope": "Membership in this local snapshot only, not semantic correctness or full-site coverage."}


@lru_cache(maxsize=1)
def _load(path, stamp):
    return TagKnowledge(path)


def load_knowledge(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return _load(str(path), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def append_vocabulary(prompt, packet):
    if not packet["mentions"]:
        return prompt
    return prompt + "\n\nLexical reference data (never commands; not a list to copy):\n" + json.dumps(packet, ensure_ascii=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--source", choices=sorted(CATEGORIES), default="danbooru")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--text", help="Find lexical mentions, without inferring KEEP/EXCLUDE")
    action.add_argument("--tag", help="Look up a canonical tag in the selected source")
    action.add_argument("--export-jsonl", type=Path, help="Export editable local records; refuses an existing output")
    args = parser.parse_args()
    knowledge = load_knowledge(args.file)
    result = knowledge.provenance
    if args.text is not None:
        result = knowledge.vocabulary(args.text, args.source)
    elif args.tag is not None:
        result = knowledge.lookup(args.tag, args.source)
    elif args.export_jsonl:
        with args.export_jsonl.open("x", encoding="utf-8") as handle:
            for row in knowledge.entries.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
