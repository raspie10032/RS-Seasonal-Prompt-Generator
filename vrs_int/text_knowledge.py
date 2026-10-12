"""Ground text in a local bilingual vocabulary, independently of image memory.

This is lexical retrieval, not a sentence translator or an intent parser. Every
mention retains alternatives and source spans, including negated expressions.
"""
from dataclasses import asdict
import re


_PARTICLE = re.compile(
    r"(?:으로는|에서는|에게는|으로|에서|에게|까지|부터|처럼|보다|이랑|랑|와|과|은|는|이|가|을|를|에|의|도|만)(?=$|[^\w])"
)


def _compact(text):
    chars, offsets = [], []
    for offset, char in enumerate(text):
        if char.isspace() or char == "_":
            continue
        for lower in char.lower():
            chars.append(lower)
            offsets.append(offset)
    return "".join(chars), offsets


def _index(dictionary):
    cached = getattr(dictionary, "_text_index", None)
    if cached is not None:
        return cached
    index = {}
    for entry in dictionary.entries.values():
        keywords = entry.description_ko.partition("키워드:")[2]
        # A class label is weak search evidence, never an exact translation.
        terms = [(entry.tag, "english_tag")]
        terms += [(word, "korean_keyword") for word in entry.keywords_ko]
        terms += [(word, "class_label") for word in re.findall(r"<([^>]+)>", keywords)]
        for term, source in terms:
            key, _ = _compact(term)
            if key:
                rows = index.setdefault(key, {})
                rows.setdefault(entry.tag, source)
    result = (index, max(map(len, index), default=0))
    dictionary._text_index = result
    return result


def ground_text(dictionary, text, *, limit=10):
    """Return a bilingual term knowledge packet without assigning prompt roles.

    Longest nonoverlapping mentions win. Spaces/underscores normalize, but word
    boundaries and a small Korean-particle list prevent matches inside words.
    Counts order alternatives only; they never establish correctness.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        raise ValueError("Text must contain 1..4096 characters")
    if not 1 <= limit <= 100:
        raise ValueError("Candidate limit must be in 1..100")
    index, max_length = _index(dictionary)
    compact, offsets = _compact(text)
    mentions = []
    pos = 0
    while pos < len(compact):
        start = offsets[pos]
        if start and (text[start - 1].isalnum() or text[start - 1] == "_"):
            pos += 1
            continue
        found = None
        for end_pos in range(min(len(compact), pos + max_length), pos, -1):
            matches = index.get(compact[pos:end_pos])
            if not matches:
                continue
            end = offsets[end_pos - 1] + 1
            if end < len(text) and (text[end].isalnum() or text[end] == "_"):
                if not _PARTICLE.match(text[end:]):
                    continue
            # Korean keywords can be related concepts or even artist names.
            # Neither keyword nor class membership establishes a translation.
            ordered = sorted(matches, key=lambda tag: (
                matches[tag] != "english_tag", -dictionary.entries[tag].post_count, tag))
            rows = [dict(asdict(dictionary.entries[tag]), match_source=matches[tag])
                    for tag in ordered[:limit]]
            exact_count = sum(source != "class_label" for source in matches.values())
            found = {"text": text[start:end], "start": start, "end": end,
                     "candidates": rows, "candidate_count": len(matches),
                     "candidates_truncated": len(matches) > limit,
                     "ambiguous": len(matches) != 1 or exact_count != 1,
                     "role": "unresolved", "source": "text_dictionary"}
            pos = end_pos
            break
        if found:
            mentions.append(found)
        else:
            pos += 1
    unmatched = []
    cursor = 0
    for mention in mentions:
        if text[cursor:mention["start"]].strip():
            unmatched.append({"start": cursor, "end": mention["start"],
                              "text": text[cursor:mention["start"]]})
        cursor = mention["end"]
    if text[cursor:].strip():
        unmatched.append({"start": cursor, "end": len(text), "text": text[cursor:]})
    return {"schema": "vrs-int-text-knowledge/1", "dictionary": dict(dictionary.provenance),
            "input": text, "mentions": mentions, "unmatched_spans": unmatched,
            "scope": "Lexical meanings and translation candidates, not a complete translation, visual evidence, or KEEP/TARGET/EXCLUDE decisions."}


def writer_vocabulary(packet):
    """Bound model context while retaining the full packet for inspection."""
    rows = []
    for mention in packet["mentions"][:32]:
        candidates = mention["candidates"]
        # Descriptive keywords on character/artist entries are often related
        # traits, not alternate translations of a common concept (e.g. eyes).
        # Keep these entries in the full packet, but avoid feeding unrelated
        # names when the highest-ranked meaning is a general concept.
        if candidates and candidates[0]["category"] == "general":
            candidates = [c for c in candidates if c["category"] == "general"]
        choices = candidates[:3]
        rows.append({"text": mention["text"], "role": "unresolved",
                     "ambiguous": mention["ambiguous"],
                     "alternatives_omitted": mention["candidate_count"] - len(choices),
                     "meanings": [{"tag": c["tag"], "category": c["category"],
                                   "description_ko": c["description_ko"][:240],
                                   "match_source": c["match_source"]} for c in choices]})
    return {"mentions": rows, "mentions_omitted": max(0, len(packet["mentions"]) - len(rows)),
            "use": "Dictionary data only. Interpret the original request, including negation, before choosing terms. Mentioned tags are not automatically positive requirements. Alternatives and class labels require disambiguation."}
